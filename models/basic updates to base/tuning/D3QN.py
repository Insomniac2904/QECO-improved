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

        # initialize zero memory [s, a, r, s_, lstm_s, lstm_s_]
        # We need to store the LSTM state for both s and s_
        self.memory = np.zeros((self.memory_size, self.n_features + 1 + 1 + self.n_features))
        # Store LSTM history separately in memory for simplicity
        self.lstm_memory_s = np.zeros((self.memory_size, self.n_lstm_step, self.n_lstm_state))
        self.lstm_memory_s_ = np.zeros((self.memory_size, self.n_lstm_step, self.n_lstm_state))
        
        self.memory_counter = 0

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
        self.loss_fn = tf.keras.losses.MeanSquaredError()
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
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        
        self.memory[index, :] = transition
        # Store bulky LSTM histories in their own arrays
        self.lstm_memory_s[index, :, :] = lstm_s_history
        self.lstm_memory_s_[index, :, :] = lstm_s_history_

        self.memory_counter += 1

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
    def _train_step(self, s, lstm_s, q_target_tensor):
        """A single training step, compiled with tf.function"""
        with tf.GradientTape() as tape:
            q_eval_pred = self.eval_net([s, lstm_s], training=True)
            loss = self.loss_fn(q_target_tensor, q_eval_pred)
            
        gradients = tape.gradient(loss, self.eval_net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.eval_net.trainable_variables))
        return loss

    def learn(self):
        # --- FIX: Guard clause to prevent sampling from an empty or small buffer ---
        if self.memory_counter < self.batch_size:
            return  # Not enough memories to sample from
        # --- END FIX ---

        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.set_weights(self.eval_net.get_weights())
            print('Network_parameter_updated\n')

        # Sample batch
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size, replace=False)
        else:
            # We are now guaranteed self.memory_counter >= self.batch_size
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size, replace=False)

        batch_memory = self.memory[sample_index, :]
        lstm_batch_s = self.lstm_memory_s[sample_index]
        lstm_batch_s_ = self.lstm_memory_s_[sample_index]

        # Convert all to Tensors
        s = tf.convert_to_tensor(batch_memory[:, :self.n_features], dtype=tf.float32)
        s_ = tf.convert_to_tensor(batch_memory[:, -self.n_features:], dtype=tf.float32)
        lstm_s = tf.convert_to_tensor(lstm_batch_s, dtype=tf.float32)
        lstm_s_ = tf.convert_to_tensor(lstm_batch_s_, dtype=tf.float32)

        # Get Q-values from networks (Eager execution)
        q_next_target = self.target_net([s_, lstm_s_]).numpy()
        q_eval_next = self.eval_net([s_, lstm_s_]).numpy()
        q_eval = self.eval_net([s, lstm_s]).numpy()
        
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval_next, axis=1)
            selected_q_next = q_next_target[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next_target, axis=1)

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next

        # --- Perform the training step ---
        q_target_tensor = tf.convert_to_tensor(q_target, dtype=tf.float32)
        self.cost = self._train_step(s, lstm_s, q_target_tensor)
        # --- End training step ---

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