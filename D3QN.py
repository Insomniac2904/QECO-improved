import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
import matplotlib.pyplot as plt

class DuelingDoubleDeepQNetwork(tf.keras.Model):
    def __init__(self,
                 n_actions,               
                 n_features,
                 n_lstm_features,
                 n_time,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.99,
                 replace_target_iter=200,  
                 memory_size=500,  
                 batch_size=32,
                 e_greedy_increment=0.00025,
                 n_lstm_step=10,
                 dueling=True,
                 double_q=True,
                 N_L1=20,
                 N_lstm=20):
        
        super().__init__()
        
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

        # LSTM parameters
        self.N_lstm = N_lstm
        self.n_lstm_step = n_lstm_step
        self.n_lstm_state = n_lstm_features

        # Initialize memory
        self.memory = np.zeros((self.memory_size, self.n_features + 1 + 1 + 
                               self.n_features + self.n_lstm_state + self.n_lstm_state))
        self.memory_counter = 0

        # Build networks
        self.eval_net = self._build_net()
        self.target_net = self._build_net()
        
        # Copy weights to target network
        self.target_net.set_weights(self.eval_net.get_weights())
        
        self.optimizer = optimizers.RMSprop(learning_rate=self.lr)
        
        # Storage lists
        self.reward_store = list()
        self.action_store = list()
        self.delay_store = list()
        self.energy_store = list()
        self.store_q_value = list()

        # LSTM history
        self.lstm_history = deque(maxlen=self.n_lstm_step)
        for _ in range(self.n_lstm_step):
            self.lstm_history.append(np.zeros([self.n_lstm_state]))

    def _build_net(self):
        # Input layers
        state_input = layers.Input(shape=(self.n_features,), name='state')
        lstm_input = layers.Input(shape=(self.n_lstm_step, self.n_lstm_state), name='lstm_state')
        
        # LSTM layer
        lstm_out = layers.LSTM(self.N_lstm)(lstm_input)
        
        # Concatenate state and LSTM output
        concat = layers.Concatenate()([state_input, lstm_out])
        
        # Dense layers - Fixed initializer issues
        l1 = layers.Dense(self.N_L1, activation='relu', 
                         kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.3),
                         bias_initializer=tf.keras.initializers.Constant(value=0.1))(concat)
        l12 = layers.Dense(self.N_L1, activation='relu',
                          kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.3),
                          bias_initializer=tf.keras.initializers.Constant(value=0.1))(l1)
        
        if self.dueling:
            # Dueling architecture
            value = layers.Dense(1, 
                               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.3),
                               bias_initializer=tf.keras.initializers.Constant(value=0.1),
                               name='value')(l12)
            advantage = layers.Dense(self.n_actions, 
                                   kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.3),
                                   bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                   name='advantage')(l12)
            
            # Combine value and advantage
            q_values = layers.Lambda(
                lambda x: x[0] + (x[1] - tf.reduce_mean(x[1], axis=1, keepdims=True)),
                name='q_values')([value, advantage])
        else:
            q_values = layers.Dense(self.n_actions, 
                                  kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.3),
                                  bias_initializer=tf.keras.initializers.Constant(value=0.1),
                                  name='q_values')(l12)
        
        model = models.Model(inputs=[state_input, lstm_input], outputs=q_values)
        return model

    def store_transition(self, s, lstm_s, a, r, s_, lstm_s_):
        transition = np.hstack((s, [a, r], s_, lstm_s, lstm_s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def update_lstm(self, lstm_s):
        self.lstm_history.append(lstm_s)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        
        if np.random.uniform() < self.epsilon:
            lstm_observation = np.array(self.lstm_history)
            lstm_observation = lstm_observation.reshape(1, self.n_lstm_step, self.n_lstm_state)
            
            actions_value = self.eval_net([observation, lstm_observation])
            self.store_q_value.append({'observation': observation, 'q_value': actions_value.numpy()})
            action = np.argmax(actions_value.numpy())
        else:
            action = np.random.randint(0, self.n_actions)
        
        return action

    @tf.function
    def _train_step(self, s, lstm_s, q_target):
        with tf.GradientTape() as tape:
            q_eval = self.eval_net([s, lstm_s])
            loss = tf.reduce_mean(tf.square(q_target - q_eval))
        
        gradients = tape.gradient(loss, self.eval_net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.eval_net.trainable_variables))
        return loss

    def learn(self):
        # Update target network
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.set_weights(self.eval_net.get_weights())
            print('Network_parameter_updated\n')

        # Sample batch
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size - self.n_lstm_step, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter - self.n_lstm_step, size=self.batch_size)

        # Prepare batch data
        batch_memory = self.memory[sample_index, :self.n_features + 1 + 1 + self.n_features]
        lstm_batch_memory = np.zeros([self.batch_size, self.n_lstm_step, self.n_lstm_state * 2])
        
        for ii in range(len(sample_index)):
            for jj in range(self.n_lstm_step):
                lstm_batch_memory[ii, jj, :] = self.memory[sample_index[ii] + jj,
                                                          self.n_features + 1 + 1 + self.n_features:]

        # Extract components
        s = tf.constant(batch_memory[:, :self.n_features], dtype=tf.float32)
        s_ = tf.constant(batch_memory[:, -self.n_features:], dtype=tf.float32)
        lstm_s = tf.constant(lstm_batch_memory[:, :, :self.n_lstm_state], dtype=tf.float32)
        lstm_s_ = tf.constant(lstm_batch_memory[:, :, self.n_lstm_state:], dtype=tf.float32)

        # Get Q values
        q_next = self.target_net([s_, lstm_s_])
        q_eval4next = self.eval_net([s_, lstm_s_])
        q_eval = self.eval_net([s, lstm_s])

        # Prepare target
        q_target = q_eval.numpy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval4next.numpy(), axis=1)
            selected_q_next = q_next.numpy()[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next.numpy(), axis=1)

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
        q_target = tf.constant(q_target, dtype=tf.float32)

        # Train
        cost = self._train_step(s, lstm_s, q_target)

        # Update epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        return cost.numpy()

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
        self.eval_net.save_weights(filepath + "_eval")
        self.target_net.save_weights(filepath + "_target")

    def load_model(self, filepath):
        try:
            self.eval_net.load_weights(filepath + "_eval")
            self.target_net.load_weights(filepath + "_target")
            print(f"Model loaded from {filepath}")
        except Exception as e:
            print(f"Failed to load model: {e}")

    def Initialize(self, iot=None):
        if iot is not None:
            self.load_model(f"./TrainedModel_20UE_2EN_PerformanceMode/800/{iot}_X_model")
