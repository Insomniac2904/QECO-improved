# filename: D3QN.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, initializers
from collections import deque
from Config import Config

class DuelingDoubleDeepQNetwork(tf.keras.Model):
    def __init__(self,
                 n_actions,
                 n_features,
                 n_temporal_features,
                 n_time,
                 learning_rate=Config.LEARNING_RATE,
                 reward_decay=Config.REWARD_DECAY,
                 e_greedy=Config.E_GREEDY,
                 replace_target_iter=Config.N_NETWORK_UPDATE,
                 memory_size=Config.MEMORY_SIZE,
                 batch_size=Config.BATCH_SIZE,
                 e_greedy_increment=0.00025,
                 dueling=True,
                 double_q=True,
                 N_L1=64):
        super(DuelingDoubleDeepQNetwork, self).__init__()

        # core env / RL params
        self.n_actions = n_actions
        self.n_features = n_features
        self.n_time = n_time
        self.lr = learning_rate
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

        # temporal / sequence params
        self.sequence_length = Config.SEQUENCE_LENGTH
        self.n_temporal_features = n_temporal_features
        self.lstm_units = getattr(Config, "LSTM_UNITS", 64)
        self.attn_units = getattr(Config, "ATTN_UNITS", 64)

        # memory row layout:
        # [ s (n_features) | a (1) | r (1) | s_ (n_features) | temporal (n_temporal_features) ]
        self.memory_row_len = self.n_features + 1 + 1 + self.n_features + self.n_temporal_features
        self.memory = np.zeros((self.memory_size, self.memory_row_len), dtype=np.float32)
        self.memory_counter = 0

        # networks: eval & target
        self.eval_net = self._build_net()
        self.target_net = self._build_net()
        # initialize target weights
        self.target_net.set_weights(self.eval_net.get_weights())

        # optimizer
        self.optimizer = optimizers.RMSprop(learning_rate=self.lr)

        # logging arrays
        self.reward_store = []
        self.action_store = []
        self.delay_store = []
        self.energy_store = []
        self.store_q_value = []

        # local temporal history (for choose_action fallback)
        self.temporal_history = deque(maxlen=self.sequence_length)
        for _ in range(self.sequence_length):
            self.temporal_history.append(np.zeros([self.n_temporal_features], dtype=np.float32))

    def _build_net(self):
        # -------------------------
        # LSTM + Attention encoder
        # -------------------------
        # state_input: static (current) features (n_features,)
        # temporal_input: sequence (sequence_length, n_temporal_features)
        state_input = layers.Input(shape=(self.n_features,), name='state_input')
        temporal_input = layers.Input(shape=(self.sequence_length, self.n_temporal_features), name='temporal_input')

        # LSTM encoder (return sequences to attend over timesteps)
        x = layers.LSTM(self.lstm_units, return_sequences=True, kernel_initializer='glorot_uniform')(temporal_input)
        # Optional small dropout
        x = layers.Dropout(0.1)(x)

        # Attention mechanism (Bahdanau style):
        # compute attention scores u_t = v^T tanh(W1*h_t + W2*h_final)
        h_final = layers.Lambda(lambda z: z[:, -1, :])(x)  # last hidden (batch, lstm_units)
        # expand h_final to time dimension
        h_final_expanded = layers.RepeatVector(self.sequence_length)(h_final)  # (batch, seq_len, units)
        # score
        score = layers.Dense(self.attn_units, activation='tanh')(layers.Concatenate()([x, h_final_expanded]))
        score = layers.Dense(1)(score)  # (batch, seq_len, 1)
        attention_weights = layers.Softmax(axis=1)(score)  # across time dimension
        # context vector: weighted sum of x
        context = layers.Multiply()([attention_weights, x])
        context = layers.Lambda(lambda z: tf.reduce_sum(z, axis=1))(context)  # (batch, lstm_units)

        # Combine context with state input
        concat = layers.Concatenate()([state_input, context])

        # Dense head (dueling)
        l1 = layers.Dense(self.N_L1, activation='relu', kernel_initializer='he_normal')(concat)
        l2 = layers.Dense(self.N_L1, activation='relu', kernel_initializer='he_normal')(l1)

        if self.dueling:
            value = layers.Dense(1, name='value')(l2)
            advantage = layers.Dense(self.n_actions, name='advantage')(l2)
            # q = V + (A - mean(A))
            q_values = layers.Lambda(
                lambda x: x[0] + (x[1] - tf.reduce_mean(x[1], axis=1, keepdims=True)),
                name='q_values')([value, advantage])
        else:
            q_values = layers.Dense(self.n_actions, name='q_values')(l2)

        model = models.Model(inputs=[state_input, temporal_input], outputs=q_values)
        return model

    #################################################
    # Storage & utility
    #################################################
    def store_transition(self, s, temporal_s, a, r, s_, temporal_s_):
        """
        Store one time-step transition as a single row.
        - s, s_ : vectors of length n_features
        - temporal_s : vector of length n_temporal_features for the current time-step
        """
        s = np.asarray(s).reshape(-1)
        s_ = np.asarray(s_).reshape(-1)
        temporal_s = np.asarray(temporal_s).reshape(-1)
        # store: [s, a, r, s_, temporal_s]
        transition = np.hstack((s, [a, r], s_, temporal_s)).astype(np.float32)
        index = self.memory_counter % self.memory_size
        self.memory[index, :transition.shape[0]] = transition
        self.memory_counter += 1

    def update_temporal_history(self, temporal_s):
        self.temporal_history.append(np.asarray(temporal_s).reshape(-1))

    def choose_action(self, observation):
        """
        NOTE: keep same epsilon semantics as original code:
        original used: if random < epsilon -> use network (exploit).
        We keep that to preserve behavior between runs.
        """
        observation = observation[np.newaxis, :].astype(np.float32)
        if np.random.uniform() < self.epsilon:
            temporal_observation = np.array(self.temporal_history)[np.newaxis, :, :].astype(np.float32)
            actions_value = self.eval_net([observation, temporal_observation], training=False)
            action = np.argmax(actions_value.numpy())
        else:
            action = np.random.randint(0, self.n_actions)
        return int(action)

    #################################################
    # Training helpers
    #################################################
    @tf.function
    def _train_step(self, s, temporal_s, a_indices, q_target_values):
        """
        s: (batch, n_features)
        temporal_s: (batch, seq_len, n_temporal_features)
        a_indices: (batch,) int32 actions to update
        q_target_values: (batch,) float32 targets
        """
        with tf.GradientTape() as tape:
            q_eval_all = self.eval_net([s, temporal_s], training=True)  # (batch, n_actions)
            # gather corresponding predicted q for taken actions
            a_idx = tf.stack([tf.range(tf.shape(q_eval_all)[0], dtype=tf.int32), a_indices], axis=1)
            q_eval = tf.gather_nd(q_eval_all, a_idx)  # (batch,)
            # MSE loss
            loss = tf.reduce_mean(tf.square(q_target_values - q_eval))
        gradients = tape.gradient(loss, self.eval_net.trainable_variables)
        # gradient clipping
        gradients, _ = tf.clip_by_global_norm(gradients, getattr(Config, "GRAD_CLIP_NORM", 10.0))
        self.optimizer.apply_gradients(zip(gradients, self.eval_net.trainable_variables))
        return loss

    def learn(self):
        # don't train until burn-in steps
        if self.memory_counter < getattr(Config, "BURN_IN_STEPS", 2000):
            return None

        # soft target update every learn step
        tau = getattr(Config, "TARGET_TAU", 5e-4)
        online_vars = self.eval_net.trainable_variables
        target_vars = self.target_net.trainable_variables
        for (t, o) in zip(target_vars, online_vars):
            t.assign((1.0 - tau) * t + tau * o)

        # sample batches of sequence_length: choose an end index for each sequence
        min_memory = min(self.memory_counter, self.memory_size)
        if min_memory < self.sequence_length + 1:
            return None

        valid_idx_start = self.sequence_length - 1
        valid_idx_end = min_memory - 1
        indices = np.random.choice(np.arange(valid_idx_start, valid_idx_end + 1), size=self.batch_size)

        # prepare batches
        s_batch = np.zeros((self.batch_size, self.n_features), dtype=np.float32)
        s__batch = np.zeros((self.batch_size, self.n_features), dtype=np.float32)
        temporal_s_batch = np.zeros((self.batch_size, self.sequence_length, self.n_temporal_features), dtype=np.float32)
        temporal_s__batch = np.zeros((self.batch_size, self.sequence_length, self.n_temporal_features), dtype=np.float32)
        rewards = np.zeros(self.batch_size, dtype=np.float32)
        actions = np.zeros(self.batch_size, dtype=np.int32)

        for i, end_idx in enumerate(indices):
            start_idx = end_idx - (self.sequence_length - 1)
            seq_rows = []
            for row_idx in range(start_idx, end_idx + 1):
                # use modulo to allow circular buffer
                seq_rows.append(self.memory[row_idx % self.memory_size])
            seq_rows = np.vstack(seq_rows)  # shape: [sequence_length, memory_row_len]

            # state (first row), next-state (last row's s_)
            s_batch[i, :] = seq_rows[0, :self.n_features]
            s__batch[i, :] = seq_rows[-1, (self.n_features + 1 + 1):(self.n_features + 1 + 1 + self.n_features)]

            temporal_field_start = self.n_features + 1 + 1 + self.n_features
            for t in range(self.sequence_length):
                temporal_s_batch[i, t, :] = seq_rows[t, temporal_field_start:temporal_field_start + self.n_temporal_features]
                # For temporal_s_ we can shift by one step (optionally); keep same for simplicity:
                temporal_s__batch[i, t, :] = seq_rows[t, temporal_field_start:temporal_field_start + self.n_temporal_features]

            # action & reward from the last row (we use last time-step's a/r as label)
            actions[i] = int(seq_rows[-1, self.n_features])
            rewards[i] = seq_rows[-1, self.n_features + 1]

        # convert to tensors
        s = tf.constant(s_batch, dtype=tf.float32)
        s_ = tf.constant(s__batch, dtype=tf.float32)
        temporal_s = tf.constant(temporal_s_batch, dtype=tf.float32)
        temporal_s_ = tf.constant(temporal_s__batch, dtype=tf.float32)

        # compute q_next via target_net and q_eval4next via eval_net for Double DQN
        q_next = self.target_net([s_, temporal_s_], training=False).numpy()  # (batch, n_actions)
        q_eval4next = self.eval_net([s_, temporal_s_], training=False).numpy()

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_next[np.arange(self.batch_size), max_act4next]
        else:
            selected_q_next = np.max(q_next, axis=1)

        q_target_values = rewards + self.gamma * selected_q_next  # (batch,)

        # train step
        loss = self._train_step(
            s, temporal_s,
            tf.convert_to_tensor(actions, dtype=tf.int32),
            tf.convert_to_tensor(q_target_values, dtype=tf.float32)
        )

        # update epsilon (keeps original semantics)
        if self.epsilon_increment is not None:
            self.epsilon = min(self.epsilon_max, self.epsilon + self.epsilon_increment)

        self.learn_step_counter += 1
        return loss

    # --- helper logging methods (unchanged) ---
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

    def save_model(self, filepath):
        # save weights (eval and target)
        self.eval_net.save_weights(filepath + "_eval.weights.h5")
        self.target_net.save_weights(filepath + "_target.weights.h5")

    def load_model(self, filepath):
        try:
            self.eval_net.load_weights(filepath + "_eval.weights.h5")
            self.target_net.load_weights(filepath + "_target.weights.h5")
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Failed to load model: {e}")

    def Initialize(self, iot=None):
        if iot is not None:
            self.load_model(f"./TrainedModel_20UE_2EN_PerformanceMode/800/{iot}_X_model")
