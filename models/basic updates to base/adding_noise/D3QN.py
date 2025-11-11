import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Input
from collections import deque
import os
import math

# --- STEP 4: NoisyDense Layer ---
# This layer replaces tf.keras.layers.Dense for exploration
class NoisyDense(tf.keras.layers.Layer):
    """
    A dense layer with factorized Gaussian noise.
    The noise is trainable and allows for state-dependent exploration.
    """
    def __init__(self, units, std_init=0.5, **kwargs):
        super(NoisyDense, self).__init__(**kwargs)
        self.units = units
        self.std_init = std_init

    def build(self, input_shape):
        in_dim = input_shape[-1]
        
        # Factorized noise parameters
        mu_init = tf.random_uniform_initializer(
            minval=-1/math.sqrt(in_dim), maxval=1/math.sqrt(in_dim)
        )
        sigma_init = tf.constant_initializer(self.std_init / math.sqrt(in_dim))
        
        # Weight parameters (mu and sigma)
        self.w_mu = self.add_weight(
            name='w_mu',
            shape=(in_dim, self.units),
            initializer=mu_init,
            trainable=True,
        )
        self.w_sigma = self.add_weight(
            name='w_sigma',
            shape=(in_dim, self.units),
            initializer=sigma_init,
            trainable=True,
        )
        
        # Bias parameters (mu and sigma)
        self.b_mu = self.add_weight(
            name='b_mu',
            shape=(self.units,),
            initializer=mu_init,
            trainable=True,
        )
        self.b_sigma = self.add_weight(
            name='b_sigma',
            shape=(self.units,),
            initializer=sigma_init,
            trainable=True,
        )
        super(NoisyDense, self).build(input_shape)

    def call(self, inputs, training=True):
        # Sample noise for weights and biases
        # This noise is shared across the batch
        in_dim = inputs.shape[-1]
        
        # Sample factorized noise
        epsilon_in = self._f(tf.random.normal([in_dim, 1]))
        epsilon_out = self._f(tf.random.normal([1, self.units]))
        
        # Combine noise
        w_epsilon = epsilon_in * epsilon_out
        b_epsilon = epsilon_out
        
        # Create noisy weights and biases
        w = self.w_mu + self.w_sigma * w_epsilon
        b = self.b_mu + self.b_sigma * b_epsilon
        
        return tf.matmul(inputs, w) + b

    def _f(self, x):
        """Scaling function for noise sampling."""
        return tf.sign(x) * tf.sqrt(tf.abs(x))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

# --- TensorFlow 2.x Keras Model Definition ---
class QNetwork(Model):
    def __init__(self, n_actions, n_lstm, n_l1, dueling):
        super(QNetwork, self).__init__()
        self.dueling = dueling
        self.n_actions = n_actions

        # LSTM layer for temporal features (congestion)
        self.lstm = LSTM(n_lstm)
        
        # Dense layers
        # L1 is shared, so it doesn't need to be noisy
        self.l1 = Dense(n_l1, activation='relu', name='l1')
        
        if self.dueling:
            # --- STEP 4: Use NoisyDense for decision layers ---
            self.l12 = NoisyDense(n_l1, name='l12_noisy')
            self.V = NoisyDense(1, name='Value_noisy')
            self.A = NoisyDense(n_actions, name='Advantage_noisy')
            # --- END STEP 4 ---
        else:
            # --- STEP 4: Use NoisyDense for decision layer ---
            self.Q = NoisyDense(n_actions, name='Q_noisy')
            # --- END STEP 4 ---

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
            # --- STEP 4: Noisy layers don't need activation here ---
            l12_out = tf.nn.relu(self.l12(l1_out))
            V = self.V(l12_out)  # State Value
            A = self.A(l12_out)  # Action Advantage
            # --- END STEP 4 ---
            # Combine V and A to get Q
            out = V + (A - tf.reduce_mean(A, axis=1, keepdims=True))
        else:
            out = self.Q(l1_out)  # Direct Q-value output
            
        return out

# --- STEP 3: SumTree Class for PER ---
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.n_entries = 0
        self.data_pointer = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p):
        idx = self.data_pointer + self.capacity - 1
        data_pointer = self.data_pointer
        self.update(idx, p)
        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1
        return data_pointer

    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], data_idx)

# --- STEP 3: PER Replay Buffer Class ---
class PERMemory:
    def __init__(self, capacity, alpha):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.data = np.zeros(capacity, dtype=object) 
        self.data_pointer = 0

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.capacity:])
        if max_p == 0:
            max_p = 1.0
        data_idx = self.tree.add(max_p ** self.alpha)
        self.data[data_idx] = transition
        self.data_pointer = self.tree.data_pointer

    def sample(self, batch_size, beta, n_features, n_lstm_step, n_lstm_state):
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
        N = self.tree.n_entries

        for i in range(batch_size):
            s = np.random.uniform(segment * i, segment * (i + 1))
            (tree_idx, p, data_idx) = self.tree.get(s)
            transition = self.data[data_idx]
            
            batch_s[i, :] = transition[0]
            batch_lstm_s[i, :, :] = transition[1]
            batch_a[i] = transition[2]
            batch_r[i] = transition[3]
            batch_s_[i, :] = transition[4]
            batch_lstm_s_[i, :, :] = transition[5]
            tree_indices[i] = tree_idx

            prob = p / total_p
            is_weights[i] = (N * prob) ** -beta

        is_weights /= np.max(is_weights)
        
        return (batch_s, batch_lstm_s, batch_a, batch_r, batch_s_, 
                batch_lstm_s_, tree_indices, is_weights)

    def update(self, tree_indices, abs_td_errors, epsilon):
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
                 replace_target_iter=200,
                 memory_size=500,
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
        self.memory_size = memory_size
        self.batch_size = batch_size
        
        # --- STEP 4: Remove Epsilon ---
        # self.epsilon_increment = e_greedy_increment (REMOVED)
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max (REMOVED)
        # --- END STEP 4 ---
        
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
        self.eval_net = QNetwork(self.n_actions, self.N_lstm, self.N_L1, self.dueling)
        self.target_net = QNetwork(self.n_actions, self.N_lstm, self.N_L1, self.dueling)

        self.lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=learning_rate_start,
            decay_steps=learning_rate_decay_steps,
            end_learning_rate=learning_rate_end,
            power=1.0
        )
        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr_schedule)
        self.loss_fn = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.NONE)
        # --- End TF2 Setup ---

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
        s_dummy = tf.zeros((1, self.n_features), dtype=tf.float32)
        lstm_s_dummy = tf.zeros((1, self.n_lstm_step, self.n_lstm_state), dtype=tf.float32)
        try:
            self.eval_net([s_dummy, lstm_s_dummy])
            self.target_net([s_dummy, lstm_s_dummy])
        except Exception as e:
            print(f"Error building models: {e}")

    def store_transition(self, s, lstm_s_history, a, r, s_, lstm_s_history_):
        transition = (s, lstm_s_history, a, r, s_, lstm_s_history_)
        self.memory.store(transition)

    def update_lstm(self, lstm_s):
        self.lstm_history.append(lstm_s)
        
    def get_current_lstm_history(self):
        return np.array(self.lstm_history)

    def choose_action(self, observation):
        # --- STEP 4: Always exploit (noise provides exploration) ---
        observation = observation[np.newaxis, :] # Shape (1, n_features)
        
        # Get current LSTM history and reshape for batch size 1
        lstm_observation = self.get_current_lstm_history()
        lstm_observation = lstm_observation.reshape(1, self.n_lstm_step, self.n_lstm_state)

        # Convert to Tensors
        s_tensor = tf.convert_to_tensor(observation, dtype=tf.float32)
        lstm_s_tensor = tf.convert_to_tensor(lstm_observation, dtype=tf.float32)

        # Get Q-values (no sess.run needed)
        # The noise in NoisyDense layers provides exploration
        actions_value = self.eval_net([s_tensor, lstm_s_tensor], training=True)
        actions_value_np = actions_value.numpy() # Convert Tensor to Numpy

        self.store_q_value.append({'observation': observation, 'q_value': actions_value_np})
        action = np.argmax(actions_value_np)
        # --- END STEP 4 ---
        
        return action

    @tf.function
    def _train_step(self, s, lstm_s, a, q_target_tensor, is_weights_tensor):
        with tf.GradientTape() as tape:
            q_eval_pred = self.eval_net([s, lstm_s], training=True)
            
            action_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), a], axis=1)
            q_eval_pred_taken = tf.gather_nd(q_eval_pred, action_indices)
            q_target_taken = tf.gather_nd(q_target_tensor, action_indices)

            abs_td_errors = tf.abs(q_target_taken - q_eval_pred_taken)
            loss = self.loss_fn(q_target_taken, q_eval_pred_taken)
            weighted_loss = is_weights_tensor * loss
            final_loss = tf.reduce_mean(weighted_loss)
            
        gradients = tape.gradient(final_loss, self.eval_net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.eval_net.trainable_variables))
        
        return final_loss, abs_td_errors

    def learn(self):
        if self.memory.tree.n_entries < self.batch_size:
            return

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.set_weights(self.eval_net.get_weights())
            print('Network_parameter_updated\n')

        self.beta = np.min([1.0, self.beta + self.beta_increment])
        
        (s, lstm_s, a, r, s_, lstm_s_, 
         tree_indices, is_weights) = self.memory.sample(self.batch_size, self.beta,
                                                        self.n_features, self.n_lstm_step,
                                                        self.n_lstm_state)

        s = tf.convert_to_tensor(s, dtype=tf.float32)
        s_ = tf.convert_to_tensor(s_, dtype=tf.float32)
        lstm_s = tf.convert_to_tensor(lstm_s, dtype=tf.float32)
        lstm_s_ = tf.convert_to_tensor(lstm_s_, dtype=tf.float32)
        a_tensor = tf.convert_to_tensor(a, dtype=tf.int32)
        r_tensor = tf.convert_to_tensor(r, dtype=tf.float32)
        is_weights_tensor = tf.convert_to_tensor(is_weights, dtype=tf.float3s)

        # Get Q-values from networks
        # We use training=True for target_net as well, to sample noise.
        # This is a common practice in Noisy DQN.
        q_next_target = self.target_net([s_, lstm_s_], training=True).numpy()
        q_eval_next = self.eval_net([s_, lstm_s_], training=True).numpy()
        q_eval = self.eval_net([s, lstm_s], training=True).numpy()
        
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = a
        reward = r

        if self.double_q:
            max_act4next = np.argmax(q_eval_next, axis=1)
            selected_q_next = q_next_target[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next_target, axis=1)

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        q_target_tensor = tf.convert_to_tensor(q_target, dtype=tf.float32)
        
        self.cost, abs_td_errors = self._train_step(s, lstm_s, a_tensor, 
                                                    q_target_tensor, is_weights_tensor)
        
        self.memory.update(tree_indices, abs_td_errors.numpy(), self.per_epsilon)

        # --- STEP 4: Remove Epsilon update ---
        # self.epsilon = ... (REMOVED)
        # --- END STEP 4 ---
        self.learn_step_counter += 1

        return self.cost.numpy()

    # --- Storage methods (unchanged) ---
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
        try:
            self.eval_net.save_weights(f"{file_path_prefix}_eval_net.h5")
            self.target_net.save_weights(f"{file_path_prefix}_target_net.h5")
            print(f"Model weights saved to {file_path_prefix}_[eval/target]_net.h5")
        except Exception as e:
            print(f"Error saving model weights: {e}")

    def load_model(self, file_path_prefix):
        try:
            self._build_models_on_dummy_data() 
            self.eval_net.load_weights(f"{file_path_prefix}_eval_net.h5")
            self.target_net.load_weights(f"{file_path_prefix}_target_net.h5")
            print(f"Model weights loaded from {file_path_prefix}")
        except Exception as e:
            print(f"Error loading model weights: {e}. Starting from scratch.")