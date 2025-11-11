import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Input
from collections import deque
import os

# --- TensorFlow 2.x Keras Model Definition ---
# We define our network as a tf.keras.Model
# This replaces the old `build_layers` function.
class QNetwork(Model):
    def __init__(self, n_actions, n_lstm, n_l1, dueling):
        super(QNetwork, self).__init__()
        self.dueling = dueling
        self.n_actions = n_actions

        # LSTM layer for temporal features (congestion)
        self.lstm = LSTM(n_lstm)
        
        # Dense layers
        self.l1 = Dense(n_l1, activation='relu', name='l1')
        
        # This architecture is based on your original code, where
        # the dueling network is one layer deeper than the non-dueling one.
        if self.dueling:
            self.l12 = Dense(n_l1, activation='relu', name='l12')
            self.V = Dense(1, name='Value')
            self.A = Dense(n_actions, name='Advantage')
        else:
            self.Q = Dense(n_actions, name='Q')

    def call(self, inputs):
        """
        Defines the forward pass.
        `inputs` is a list/tuple: [s, lstm_s]
        """
        s, lstm_s = inputs

        # Process LSTM input
        # Input shape for LSTM is (batch_size, timesteps, features)
        lstm_output = self.lstm(lstm_s)

        # Concatenate LSTM output with current state
        concat = tf.concat([lstm_output, s], axis=1)
        
        # Pass through dense layers
        l1_out = self.l1(concat)

        if self.dueling:
            l12_out = self.l12(l1_out)
            V = self.V(l12_out)  # State Value
            A = self.A(l12_out)  # Action Advantage
            # Combine V and A to get Q
            out = V + (A - tf.reduce_mean(A, axis=1, keepdims=True))
        else:
            out = self.Q(l1_out)  # Direct Q-value output
            
        return out

# --- STEP 3: SumTree Class for PER ---
class SumTree:
    """
    A SumTree data structure used for efficient priority sampling.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # Tree is a 1D array.
        # Parent node index: (i - 1) // 2
        # Left child index: 2 * i + 1
        # Right child index: 2 * i + 2
        self.tree = np.zeros(2 * capacity - 1)
        # Data is stored in a separate array, not in the tree
        self.n_entries = 0
        self.data_pointer = 0

    def _propagate(self, idx, change):
        """Propagates a change in priority up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Finds the leaf node (sample) for a given priority value `s`."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Returns the total priority (root node value)."""
        return self.tree[0]

    def add(self, p):
        """Stores a new priority and returns the index to store the data."""
        idx = self.data_pointer + self.capacity - 1
        
        # We store the transition in a separate data array at self.data_pointer
        data_pointer = self.data_pointer
        
        self.update(idx, p) # Update the tree
        
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0 # Ring buffer

        if self.n_entries < self.capacity:
            self.n_entries += 1
            
        return data_pointer # Return the index where data should be stored

    def update(self, idx, p):
        """Updates the priority of a leaf node."""
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """Gets the leaf index, priority, and data index for a given sample value `s`."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], data_idx)

# --- STEP 3: PER Replay Buffer Class ---
class PERMemory:
    def __init__(self, capacity, alpha):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        # We store transitions in a simple list, indexed by data_pointer from SumTree
        self.data = np.zeros(capacity, dtype=object) 
        self.is_full = False
        self.data_pointer = 0 # Duplicates sumtree.data_pointer, but easier to track

    def store(self, transition):
        """
        Stores a new transition. We give it the max priority to ensure
        it gets sampled at least once.
        """
        max_p = np.max(self.tree.tree[-self.capacity:])
        if max_p == 0:
            max_p = 1.0 # Initial max priority
        
        # Add priority to SumTree, get data index
        data_idx = self.tree.add(max_p ** self.alpha)
        # Store transition data at that index
        self.data[data_idx] = transition
        self.data_pointer = self.tree.data_pointer # Sync pointer

    def sample(self, batch_size, beta, n_features, n_lstm_step, n_lstm_state):
        """
        Samples a batch, calculating Importance-Sampling (IS) weights.
        """
        # Create arrays for the batch
        batch_s = np.zeros((batch_size, n_features), dtype=np.float32)
        batch_lstm_s = np.zeros((batch_size, n_lstm_step, n_lstm_state), dtype=np.float32)
        batch_a = np.zeros(batch_size, dtype=np.int32)
        batch_r = np.zeros(batch_size, dtype=np.float32)
        batch_s_ = np.zeros((batch_size, n_features), dtype=np.float32)
        batch_lstm_s_ = np.zeros((batch_size, n_lstm_step, n_lstm_state), dtype=np.float32)

        tree_indices = np.zeros(batch_size, dtype=np.int32)
        is_weights = np.zeros(batch_size, dtype=np.float32)

        total_p = self.tree.total()
        segment = total_p / batch_size
        
        # Calculate N (current number of entries) for IS weight
        N = self.tree.n_entries

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            (tree_idx, p, data_idx) = self.tree.get(s)

            # (s, lstm_s, a, r, s_, lstm_s_)
            transition = self.data[data_idx]
            
            batch_s[i, :] = transition[0]
            batch_lstm_s[i, :, :] = transition[1]
            batch_a[i] = transition[2]
            batch_r[i] = transition[3]
            batch_s_[i, :] = transition[4]
            batch_lstm_s_[i, :, :] = transition[5]
            
            tree_indices[i] = tree_idx

            # Calculate Importance-Sampling (IS) Weight
            prob = p / total_p
            is_weights[i] = (N * prob) ** -beta

        # Normalize weights for stability
        is_weights /= np.max(is_weights)
        
        return (batch_s, batch_lstm_s, batch_a, batch_r, batch_s_, 
                batch_lstm_s_, tree_indices, is_weights)

    def update(self, tree_indices, abs_td_errors, epsilon):
        """Updates the priorities in the SumTree after a learning step."""
        priorities = (abs_td_errors + epsilon) ** self.alpha
        for idx, p in zip(tree_indices, priorities):
            self.tree.update(idx, p)

# --- Main D3QN Agent Class (Refactored for TF2) ---
class DuelingDoubleDeepQNetwork:
    def __init__(self,
                 n_actions,
                 n_features,
                 n_lstm_features,
                 n_time,
                 # --- Step 1: Updated Hyperparameters ---
                 learning_rate_start=5e-4,
                 learning_rate_end=1e-6,
                 learning_rate_decay_steps=100000,
                 batch_size=64,
                 # --- End Step 1 ---
                 # --- Step 3: PER Parameters ---
                 per_alpha=0.6,
                 per_beta_start=0.4,
                 per_beta_anneal_steps=100000,
                 per_epsilon=1e-6,
                 # --- End Step 3 ---
                 reward_decay=0.9,
                 e_greedy=0.99,
                 replace_target_iter=200,
                 memory_size=500,
                 e_greedy_increment=0.00025,
                 n_lstm_step=10,
                 dueling=True,
                 double_q=True,
                 N_L1=20,
                 N_lstm=20):

        self.n_actions = n_actions
        self.n_features = n_features
        self.n_time = n_time
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.dueling = dueling
        self.double_q = double_q
        self.learn_step_counter = 0
        self.N_L1 = N_L1

        # lstm
        self.N_lstm = N_lstm
        self.n_lstm_step = n_lstm_step
        self.n_lstm_state = n_lstm_features

        # --- STEP 3: Replace old memory with PERMemory ---
        self.memory = PERMemory(memory_size, per_alpha)
        self.beta = per_beta_start
        self.beta_increment = (1.0 - per_beta_start) / per_beta_anneal_steps
        self.per_epsilon = per_epsilon
        # --- End Memory Replacement ---

        # --- TF2/Keras Setup ---
        # 1. Build Keras models (replaces _build_net)
        self.eval_net = QNetwork(self.n_actions, self.N_lstm, self.N_L1, self.dueling)
        self.target_net = QNetwork(self.n_actions, self.N_lstm, self.N_L1, self.dueling)

        # 2. Setup Optimizer with LR Schedule (Step 1)
        self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=learning_rate_start,
            decay_steps=learning_rate_decay_steps,
            end_learning_rate=learning_rate_end,
            power=1.0  # Linear decay
        )
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr_schedule)
        # --- STEP 3: Use Huber loss and NO reduction ---
        # We need per-element loss for IS weights
        self.loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        # --- End TF2 Setup ---

        # 3. Initialize Target Network
        # We must call the models on dummy data to build them before loading/setting weights
        self._build_models_on_dummy_data()
        self.target_net.set_weights(self.eval_net.get_weights())

        self.reward_store = []
        self.action_store = []
        self.delay_store = []
        self.energy_store = []

        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for _ in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))

        self.store_q_value = []

    def _build_models_on_dummy_data(self):
        """
        Calls the models on dummy Tensors to initialize their weights
        (a required step in TF2 Keras functional/subclass models).
        """
        s_dummy = tf.zeros((1, self.n_features), dtype=tf.float32)
        lstm_s_dummy = tf.zeros((1, self.n_lstm_step, self.n_lstm_state), dtype=tf.float32)
        try:
            self.eval_net([s_dummy, lstm_s_dummy])
            self.target_net([s_dummy, lstm_s_dummy])
        except Exception as e:
            print(f"Error building models: {e}")

    def store_transition(self, s, lstm_s_history, a, r, s_, lstm_s_history_):
        # --- STEP 3: Store transition in PERMemory ---
        # We store the full histories as provided
        transition = (s, lstm_s_history, a, r, s_, lstm_s_history_)
        self.memory.store(transition)
        # --- End Store ---

    def update_lstm(self, lstm_s):
        """lstm_s is the observation for the *current* step"""
        self.lstm_history.append(lstm_s)
        
    def get_current_lstm_history(self):
        """Returns the current history as a numpy array"""
        return np.array(self.lstm_history)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :] # Shape (1, n_features)

        if np.random.uniform() < self.epsilon:
            # Get current LSTM history and reshape for batch size 1
            lstm_observation = self.get_current_lstm_history()
            lstm_observation = lstm_observation.reshape(1, self.n_lstm_step, self.n_lstm_state)

            # Convert to Tensors
            s_tensor = tf.convert_to_tensor(observation, dtype=tf.float32)
            lstm_s_tensor = tf.convert_to_tensor(lstm_observation, dtype=tf.float32)

            # Get Q-values (no sess.run needed)
            actions_value = self.eval_net([s_tensor, lstm_s_tensor])
            actions_value_np = actions_value.numpy() # Convert Tensor to Numpy

            self.store_q_value.append({'observation': observation, 'q_value': actions_value_np})
            action = np.argmax(actions_value_np)
        else:
            action = np.random.randint(0, self.n_actions) # Fixed to be 0-indexed
            
        # The environment expects action 1 for local, 1+ for offload
        # But our network is 0-indexed. 
        # Let's check env.n_actions. It's 1 + self.n_edge
        # Action 0 = local, Action 1...n_edge = offload
        # The original code's random.randint(1, self.n_actions) was a bug.
        # It would never explore action 0 (local).
        
        return action

    @tf.function # Compiles this function into a high-performance graph
    def _train_step(self, s, lstm_s, a, q_target_tensor, is_weights_tensor):
        """
        A single training step, compiled with tf.function
        Now accepts actions and IS weights.
        Returns the final loss and the absolute TD errors.
        """
        with tf.GradientTape() as tape:
            # Get Q-values for *all* actions
            q_eval_pred = self.eval_net([s, lstm_s], training=True)
            
            # --- PER Loss Calculation ---
            # We need the Q values for the actions *actually taken*
            # Create (batch_size, 2) tensor of indices
            action_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), a], axis=1)
            
            # Gather the Q-values for the actions taken
            q_eval_pred_taken = tf.gather_nd(q_eval_pred, action_indices)
            
            # Gather the Target Q-values for the actions taken
            # (q_target_tensor already has r + gamma*Q in the right spot)
            q_target_taken = tf.gather_nd(q_target_tensor, action_indices)

            # Calculate TD-Error for the chosen actions
            abs_td_errors = tf.abs(q_target_taken - q_eval_pred_taken)
            
            # Calculate Huber loss for the chosen actions
            loss = self.loss_fn(q_target_taken, q_eval_pred_taken) # Shape: (batch_size,)

            # Apply Importance-Sampling (IS) weights
            weighted_loss = is_weights_tensor * loss
            
            # Compute the final, mean loss to backpropagate
            final_loss = tf.reduce_mean(weighted_loss)
            # --- End PER Loss Calculation ---
            
        gradients = tape.gradient(final_loss, self.eval_net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.eval_net.trainable_variables))
        
        # Return final loss for reporting, and TD errors for tree update
        return final_loss, abs_td_errors

    def learn(self):
        # --- STEP 3: Update Guard Clause ---
        if self.memory.tree.n_entries < self.batch_size:
            return  # Not enough memories to sample from
        # --- END FIX ---

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.set_weights(self.eval_net.get_weights())
            print('Network_parameter_updated\n')

        # --- STEP 3: Sample from PERMemory ---
        # Anneal beta
        self.beta = np.min([1.0, self.beta + self.beta_increment])
        
        (s, lstm_s, a, r, s_, lstm_s_, 
         tree_indices, is_weights) = self.memory.sample(self.batch_size, self.beta,
                                                        self.n_features, self.n_lstm_step,
                                                        self.n_lstm_state)
        # --- End Sample ---

        # Convert all to Tensors
        s = tf.convert_to_tensor(s, dtype=tf.float32)
        s_ = tf.convert_to_tensor(s_, dtype=tf.float32)
        lstm_s = tf.convert_to_tensor(lstm_s, dtype=tf.float32)
        lstm_s_ = tf.convert_to_tensor(lstm_s_, dtype=tf.float32)
        a_tensor = tf.convert_to_tensor(a, dtype=tf.int32)
        r_tensor = tf.convert_to_tensor(r, dtype=tf.float32)
        is_weights_tensor = tf.convert_to_tensor(is_weights, dtype=tf.float32)


        # Get Q-values from networks (Eager execution)
        q_next_target = self.target_net([s_, lstm_s_]).numpy()
        q_eval_next = self.eval_net([s_, lstm_s_]).numpy()
        q_eval = self.eval_net([s, lstm_s]).numpy()
        
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = a # Actions are already sampled
        reward = r # Rewards are already sampled

        if self.double_q:
            max_act4next = np.argmax(q_eval_next, axis=1)
            selected_q_next = q_next_target[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next_target, axis=1)

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        # --- Perform the training step ---
        q_target_tensor = tf.convert_to_tensor(q_target, dtype=tf.float32)
        
        self.cost, abs_td_errors = self._train_step(s, lstm_s, a_tensor, 
                                                    q_target_tensor, is_weights_tensor)
        
        # --- STEP 3: Update priorities in SumTree ---
        self.memory.update(tree_indices, abs_td_errors.numpy(), self.per_epsilon)
        # --- End Update ---

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        return self.cost.numpy()

    # --- Storage methods (unchanged, they are fine) ---
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

    # --- Keras-based Save/Load ---
    def save_model(self, file_path_prefix):
        """Saves model weights using Keras H5 format."""
        try:
            self.eval_net.save_weights(f"{file_path_prefix}_eval_net.h5")
            self.target_net.save_weights(f"{file_path_prefix}_target_net.h5")
            print(f"Model weights saved to {file_path_prefix}_[eval/target]_net.h5")
        except Exception as e:
            print(f"Error saving model weights: {e}")

    def load_model(self, file_path_prefix):
        """Loads model weights from Keras H5 format."""
        try:
            # Must build models first
            self._build_models_on_dummy_data() 
            
            self.eval_net.load_weights(f"{file_path_prefix}_eval_net.h5")
            self.target_net.load_weights(f"{file_path_prefix}_target_net.h5")
            print(f"Model weights loaded from {file_path_prefix}")
        except Exception as e:
            print(f"Error loading model weights: {e}. Starting from scratch.")