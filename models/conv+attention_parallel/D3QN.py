import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers
import numpy as np
import random
from collections import deque

# --- Step 3: Prioritized Experience Replay (PER) ---

class SumTree:
    """
    A SumTree implementation for efficient prioritized sampling.
    Tree structure:
        0
       / \
      0   0
     / \ / \
    0 0 0 0  (leaf nodes store priorities)
    """
    write = 0 # Current write index

    def __init__(self, capacity):
        self.capacity = capacity
        # Tree is a 1D array
        # Parent nodes are at index 0 to capacity - 1
        # Leaf nodes (priorities) are at index capacity - 1 to 2*capacity - 2
        self.tree = np.zeros(2 * capacity - 1)
        # Data (transitions) are stored separately
        self.data = np.empty(capacity, dtype=object)
        self.n_entries = 0

    def _propagate(self, idx, change):
        """Propagate priority changes up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Find the sample index for a given priority value `s`."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx # Leaf node

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Total priority."""
        return self.tree[0]

    def add(self, p, data):
        """Store new transition in the tree."""
        idx = self.write + self.capacity - 1
        
        self.data[self.write] = data
        self.update(idx, p) # Update priority

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0 # Ring buffer
            
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        """Update priority for a given tree index."""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """Get leaf index, priority, and data for a given value `s`."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])

class PERMemory:
    """Prioritized Experience Replay Memory."""
    def __init__(self, capacity, per_alpha, per_beta_start, per_beta_anneal_steps, per_epsilon):
        self.tree = SumTree(capacity)
        self.alpha = per_alpha
        self.beta = per_beta_start
        self.beta_increment = (1.0 - per_beta_start) / per_beta_anneal_steps
        self.epsilon = per_epsilon
        self.abs_err_upper = 1.0 # Max possible priority

    def store(self, transition):
        """Store a new transition with max priority."""
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)

    def sample(self, n):
        """Sample a batch of transitions."""
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        # Anneal beta
        self.beta = np.min([1.0, self.beta + self.beta_increment])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            (idx, p, data) = self.tree.get(s)
            if data == 0: # Handle edge case where data might not be populated
                continue
                
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        
        # Calculate Importance Sampling (IS) weights
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max() # Normalize weights

        return batch, idxs, is_weight

    def update_priorities(self, tree_idxs, abs_errors):
        """Update priorities of sampled transitions."""
        # Add epsilon to errors to ensure non-zero priority
        abs_errors += self.epsilon
        # Clip errors to max value
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        # Convert errors to priorities (P(i) = |TD-error| ^ alpha)
        priorities = np.power(clipped_errors, self.alpha)

        for ti, p in zip(tree_idxs, priorities):
            self.tree.update(ti, p)

    def __len__(self):
        return self.tree.n_entries

# --- End Step 3 ---


# --- HYBRID (Conv1D + GRU+Attention) Model ---
class QNetwork(Model):
    def __init__(self, n_actions, n_features, n_lstm, n_l1, n_gru_units, n_conv_filters, dueling):
        super(QNetwork, self).__init__()
        self.dueling = dueling
        self.n_actions = n_actions
        self.n_l1 = n_l1
        self.n_gru_units = n_gru_units

        # --- Stream 1: Current State 's' ---
        self.s_l1 = layers.Dense(n_l1, activation='relu', name='State_L1')
        
        # --- FIX: Add a projection layer for the query ---
        # This layer will project the s_out (dim 20) to match the GRU output (dim 10)
        # for the attention mechanism.
        self.query_proj = layers.Dense(n_gru_units, activation='relu', name='Query_Projection')
        # --- END FIX ---

        # --- Stream 2: Conv1D Path for History 'lstm_s' ---
        self.conv1 = layers.Conv1D(filters=n_conv_filters, kernel_size=3, padding='same', activation='relu', name='Conv1D_Path')
        self.pool1 = layers.GlobalAveragePooling1D(name='Conv_Pool')

        # --- Stream 3: GRU+Attention Path for History 'lstm_s' ---
        self.gru1 = layers.GRU(n_gru_units, return_sequences=True, name='GRU_Path')
        # Note: Keras Attention layer computes dot-product, so query/key dims must match
        self.attention1 = layers.Attention(name='Attention_Path')
        
        # --- Dueling Heads ---
        self.l12 = layers.Dense(n_l1, activation='relu', name='Combined_L12')
        self.V = layers.Dense(1, name='Value')
        self.A = layers.Dense(n_actions, name='Advantage')

    def call(self, inputs):
        s, lstm_s = inputs

        # --- Stream 1: Process Current State 's' ---
        s_out = self.s_l1(s) # Shape: (batch, n_l1=20)

        # --- Stream 2: Process History 'lstm_s' with Conv1D ---
        conv_out = self.conv1(lstm_s)
        conv_pooled = self.pool1(conv_out) # Shape: (batch, n_conv_filters)

        # --- Stream 3: Process History 'lstm_s' with GRU+Attention ---
        gru_out = self.gru1(lstm_s) # Shape: (batch, 10, n_gru_units=10)
        
        # --- FIX: Use the *projected* s_out as the query ---
        # Project s_out from (batch, 20) to (batch, 10)
        query = self.query_proj(s_out)
        # Add a time dimension to match the GRU output: (batch, 1, 10)
        query = tf.expand_dims(query, axis=1)
        
        # Attention: query=(batch, 1, 10), value=gru_out=(batch, 10, 10)
        context_vector = self.attention1(inputs=[query, gru_out]) # Shape: (batch, 1, 10)
        context_vector = tf.squeeze(context_vector, axis=1) # Shape: (batch, 10)
        # --- END FIX ---

        # --- Merge All Streams ---
        concat = tf.concat([s_out, conv_pooled, context_vector], axis=1)
        
        # --- Dueling Q-Value Calculation ---
        l12_out = self.l12(concat)
        V = self.V(l12_out)
        A = self.A(l12_out)
        out = V + (A - tf.reduce_mean(A, axis=1, keepdims=True))
            
        return out, s_out, context_vector # Return extras for debugging if needed

# --- End Hybrid Model ---


class DuelingDoubleDeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 n_lstm_state,
                 n_time,
                 # --- Step 1: LR Schedule ---
                 learning_rate_start=1e-4,
                 learning_rate_end=1e-6,
                 learning_rate_decay_steps=1000000,
                 batch_size=64,
                 # --- End Step 1 ---
                 reward_decay=0.9,
                 # --- Reverted Step 4: Epsilon-Greedy ---
                 e_greedy=0.99,
                 e_greedy_increment=0.00025,
                 # --- End Revert ---
                 replace_target_iter=200,
                 memory_size=500,
                 # --- Step 3: PER Parameters ---
                 per_alpha=0.6,
                 per_beta_start=0.4,
                 per_beta_anneal_steps=500000,
                 per_epsilon=1e-6,
                 # --- End Step 3 ---
                 n_lstm_step=10,
                 dueling=True,
                 double_q=True,
                 N_L1=20,
                 N_lstm=20):

        self.n_actions = n_actions
        self.n_features = n_features
        self.n_time = n_time
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.batch_size = batch_size
        self.dueling = dueling
        self.double_q = double_q
        self.learn_step_counter = 0

        # --- Reverted Step 4: Epsilon-Greedy ---
        self.epsilon_max = e_greedy
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        # --- End Revert ---

        # lstm
        self.N_lstm = N_lstm
        self.n_lstm_step = n_lstm_step
        self.n_lstm_state = n_lstm_state

        # --- Step 3: PER Memory ---
        self.memory = PERMemory(memory_size, per_alpha, per_beta_start, per_beta_anneal_steps, per_epsilon)
        self.beta = self.memory.beta # To be read by main
        # --- End Step 3 ---

        # [s, lstm_s, a, r, s_, lstm_s_]
        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for _ in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))

        # --- Build Networks ---
        self.N_L1 = N_L1
        self.q_network = QNetwork(n_actions, n_features, n_lstm_state, n_lstm_step, N_L1, N_lstm)
        self.target_q_network = QNetwork(n_actions, n_features, n_lstm_state, n_lstm_step, N_L1, N_lstm)
        
        # Initial copy
        self.update_target_network()

        # --- Step 1: LR Schedule & Optimizer ---
        self.lr_schedule = optimizers.schedules.PolynomialDecay(
            initial_learning_rate=learning_rate_start,
            decay_steps=learning_rate_decay_steps,
            end_learning_rate=learning_rate_end,
            power=1.0 # Linear decay
        )
        self.optimizer = optimizers.Adam(learning_rate=self.lr_schedule)
        # --- End Step 1 ---
        
        self.loss_fn = tf.keras.losses.MeanSquaredError()

        self.reward_store = []
        self.action_store = []
        self.delay_store = []
        self.energy_store = []

    def update_target_network(self):
        """Copy weights from q_network to target_q_network."""
        self.target_q_network.set_weights(self.q_network.get_weights())
        # print('Network_parameter_updated\n')

    def store_transition(self, s, lstm_s, a, r, s_, lstm_s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        
        transition = [s, lstm_s, a, r, s_, lstm_s_]
        
        # --- Step 3: Store in PER ---
        self.memory.store(transition)
        # --- End Step 3 ---
        
        self.memory_counter += 1

    def update_lstm(self, lstm_s):
        self.lstm_history.append(lstm_s)
        
    def get_current_lstm_history(self):
        return np.array(self.lstm_history)

    def choose_action(self, observation):
        # Add batch dimension
        observation = observation[np.newaxis, :]
        lstm_observation = self.get_current_lstm_history()[np.newaxis, :, :]

        # --- Reverted Step 4: Epsilon-Greedy ---
        if np.random.uniform() < self.epsilon:
            # Exploit: Choose best action from Q-network
            actions_value, _, _ = self.q_network.predict([observation, lstm_observation], verbose=0)
            action = np.argmax(actions_value)
        else:
            # Explore: Choose random action
            action = np.random.randint(0, self.n_actions)
        return action
        # --- End Revert ---

    @tf.function
    def train_step(self, s_batch, lstm_s_batch, a_batch, r_batch, s_next_batch, lstm_s_next_batch, is_weights):
        # --- Double DQN Logic ---
        # 1. Get next actions from *online* network
        q_next_online, _, _ = self.q_network([s_next_batch, lstm_s_next_batch])
        max_act_next = tf.argmax(q_next_online, axis=1, output_type=tf.int32)
        
        # 2. Get Q-values for next actions from *target* network
        q_next_target, _, _ = self.target_q_network([s_next_batch, lstm_s_next_batch])
        
        # Create mask for batch indexing
        batch_indices = tf.range(self.batch_size, dtype=tf.int32)
        action_indices = tf.stack([batch_indices, max_act_next], axis=1)
        
        selected_q_next = tf.gather_nd(q_next_target, action_indices)
        
        # 3. Calculate target Q-value
        q_target = r_batch + self.gamma * selected_q_next
        
        # 4. Train *online* network
        with tf.GradientTape() as tape:
            q_values, _, _ = self.q_network([s_batch, lstm_s_batch])
            # Get Q-value for the action that was actually taken
            action_indices_taken = tf.stack([batch_indices, a_batch], axis=1)
            q_eval = tf.gather_nd(q_values, action_indices_taken)
            
            # --- Step 3: PER ---
            # Calculate TD-Error for priority update
            abs_errors = tf.abs(q_target - q_eval)
            # Weight the loss by Importance Sampling (IS) weights
            loss = self.loss_fn(q_target, q_eval, sample_weight=is_weights)
            # --- End Step 3 ---

        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        return loss, abs_errors


    def learn(self):
        # Check if it's time to update the target network
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.update_target_network()

        # --- Step 3: Sample from PER ---
        # Don't learn if memory is not full enough
        if len(self.memory) < self.batch_size:
            return 0.0 # Return 0 loss
            
        self.beta = self.memory.beta # Update beta for main
        
        batch, tree_idxs, is_weights = self.memory.sample(self.batch_size)
        
        s_batch = np.array([transition[0] for transition in batch], dtype=np.float32)
        lstm_s_batch = np.array([transition[1] for transition in batch], dtype=np.float32)
        a_batch = np.array([transition[2] for transition in batch], dtype=np.int32)
        r_batch = np.array([transition[3] for transition in batch], dtype=np.float32)
        s_next_batch = np.array([transition[4] for transition in batch], dtype=np.float32)
        lstm_s_next_batch = np.array([transition[5] for transition in batch], dtype=np.float32)
        is_weights = tf.convert_to_tensor(is_weights, dtype=tf.float32)
        # --- End Step 3 ---

        # Run training step
        loss, abs_errors = self.train_step(s_batch, lstm_s_batch, a_batch, r_batch, s_next_batch, lstm_s_next_batch, is_weights)

        # --- Step 3: Update Priorities ---
        self.memory.update_priorities(tree_idxs, abs_errors.numpy())
        # --- End Step 3 ---

        # --- Reverted Step 4: Increment Epsilon ---
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        # --- End Revert ---
        
        self.learn_step_counter += 1
        
        return loss.numpy()

    def do_store_reward(self, episode, time, reward):
        while episode >= len(self.reward_store):
            self.reward_store.append(np.zeros([self.n_time]))
        self.reward_store[episode][time] = reward

    def do_store_action(self, episode, time, action):
        while episode >= len(self.action_store):
            self.action_store.append(-np.ones([self.n_time]))
        self.action_store[episode][time] = action

    def do_store_delay(self, episode, time, delay):
        while episode >= len(self.delay_store):
            self.delay_store.append(np.zeros([self.n_time]))
        self.delay_store[episode][time] = delay

    def do_store_energy(self, episode, time, energy, energy2, energy3, energy4):
        fog_energy = 0
        for i in range(len(energy3)):
            if energy3[i] != 0:
                fog_energy = energy3[i]

        idle_energy = 0
        for i in range(len(energy4)):
            if energy4[i] != 0:
                idle_energy = energy4[i]

        while episode >= len(self.energy_store):
            self.energy_store.append(np.zeros([self.n_time]))
        self.energy_store[episode][time] = energy + energy2 + fog_energy + idle_energy

    def save_model(self, model_prefix):
        """Saves the Q-network and Target Q-network weights."""
        self.q_network.save_weights(f"{model_prefix}_q_network.weights.h5")
        self.target_q_network.save_weights(f"{model_prefix}_target_q_network.weights.h5")

    def load_model(self, model_prefix):
        """Loads the Q-network and Target Q-network weights."""
        try:
            self.q_network.load_weights(f"{model_prefix}_q_network.weights.h5")
            self.target_q_network.load_weights(f"{model_prefix}_target_q_network.weights.h5")
            print(f"Successfully loaded model weights from {model_prefix}")
        except Exception as e:
            print(f"Error loading model weights: {e}. Starting from scratch.")