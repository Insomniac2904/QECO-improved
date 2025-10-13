# filename: D3QN.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from collections import deque
from Config import Config

# CORRECTED: Custom Layer using the correct patching function
class PatchingLayer(layers.Layer):
    def __init__(self, patch_length, **kwargs):
        super(PatchingLayer, self).__init__(**kwargs)
        self.patch_length = patch_length

    def call(self, inputs):
        # Temporarily expand the input to 4D to use extract_patches
        # Shape becomes: (batch, 1, sequence_length, features)
        inputs_expanded = tf.expand_dims(inputs, axis=1)

        # Create patches using the correct function
        patches = tf.image.extract_patches(
            images=inputs_expanded,
            sizes=[1, 1, self.patch_length, 1],    # Patch size: 1xpatch_length
            strides=[1, 1, self.patch_length, 1], # Non-overlapping patches
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        
        # Reshape the output to the desired 3D format
        # Shape becomes: (batch, num_patches, patch_length * features)
        input_shape = tf.shape(inputs)
        batch_size, _, _, features = patches.shape
        patches = tf.reshape(patches, [batch_size, -1, self.patch_length * input_shape[2]])
        return patches

    def compute_output_shape(self, input_shape):
        # Helper method for Keras to infer the shape
        num_patches = (input_shape[1] - self.patch_length) // self.patch_length + 1
        return (input_shape[0], num_patches, self.patch_length * input_shape[2])


class DuelingDoubleDeepQNetwork(tf.keras.Model):
    def __init__(self,
                 n_actions,
                 n_features,
                 n_temporal_features,
                 n_time,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.99,
                 replace_target_iter=200,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=0.00025,
                 dueling=True,
                 double_q=True,
                 N_L1=20):

        super(DuelingDoubleDeepQNetwork, self).__init__()

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

        # PatchTST parameters from Config
        self.sequence_length = Config.SEQUENCE_LENGTH
        self.patch_length = Config.PATCH_LENGTH
        self.d_model = Config.D_MODEL
        self.n_heads = Config.N_HEADS
        self.n_encoder_layers = Config.N_ENCODER_LAYERS
        self.n_temporal_features = n_temporal_features
        
        assert (self.sequence_length % self.patch_length) == 0, "Sequence length must be divisible by patch length"
        self.num_patches = self.sequence_length // self.patch_length

        self.memory = np.zeros((self.memory_size, self.n_features + 1 + 1 + 
                                self.n_features + self.n_temporal_features + self.n_temporal_features))
        self.memory_counter = 0

        self.eval_net = self._build_net()
        self.target_net = self._build_net()

        self.target_net.set_weights(self.eval_net.get_weights())
        self.optimizer = optimizers.RMSprop(learning_rate=self.lr)

        self.reward_store = []
        self.action_store = []
        self.delay_store = []
        self.energy_store = []
        self.store_q_value = []

        self.temporal_history = deque(maxlen=self.sequence_length)
        for _ in range(self.sequence_length):
            self.temporal_history.append(np.zeros([self.n_temporal_features]))

    def _build_net(self):
        def transformer_encoder(inputs, head_size, num_heads, ff_dim):
            x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=0.1)(inputs, inputs)
            x = layers.Dropout(0.1)(x)
            x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)
            ff_out = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
            ff_out = layers.Dropout(0.1)(ff_out)
            ff_out = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(ff_out)
            return layers.LayerNormalization(epsilon=1e-6)(x + ff_out)

        state_input = layers.Input(shape=(self.n_features,), name='state_input')
        temporal_input = layers.Input(shape=(self.sequence_length, self.n_temporal_features), name='temporal_input')

        # 1. Patching (Using our corrected custom layer)
        patches = PatchingLayer(self.patch_length)(temporal_input)
        
        # 2. Patch Embedding
        patch_embedding = layers.Dense(self.d_model)(patches)

        # 3. Positional Encoding
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        pos_embedding = layers.Embedding(input_dim=self.num_patches, output_dim=self.d_model)(positions)
        x = patch_embedding + pos_embedding
        
        # 4. Transformer Encoder Stack
        for _ in range(self.n_encoder_layers):
            x = transformer_encoder(x, head_size=self.d_model, num_heads=self.n_heads, ff_dim=self.d_model)
            
        # 5. Aggregation
        transformer_output = layers.GlobalAveragePooling1D()(x)

        # --- D3QN Head ---
        concat = layers.Concatenate()([state_input, transformer_output])
        
        l1 = layers.Dense(self.N_L1, activation='relu', kernel_initializer='he_normal')(concat)
        l2 = layers.Dense(self.N_L1, activation='relu', kernel_initializer='he_normal')(l1)

        if self.dueling:
            value = layers.Dense(1, name='value')(l2)
            advantage = layers.Dense(self.n_actions, name='advantage')(l2)
            q_values = layers.Lambda(
                lambda x: x[0] + (x[1] - tf.reduce_mean(x[1], axis=1, keepdims=True)),
                name='q_values')([value, advantage])
        else:
            q_values = layers.Dense(self.n_actions, name='q_values')(l2)

        model = models.Model(inputs=[state_input, temporal_input], outputs=q_values)
        return model
        
    def store_transition(self, s, temporal_s, a, r, s_, temporal_s_):
        transition = np.hstack((s, [a, r], s_, temporal_s, temporal_s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def update_temporal_history(self, temporal_s):
        self.temporal_history.append(temporal_s)

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]
        if np.random.uniform() < self.epsilon:
            temporal_observation = np.array(self.temporal_history)[np.newaxis, :, :]
            actions_value = self.eval_net([observation, temporal_observation], training=False)
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)
        return action
    
    @tf.function
    def _train_step(self, s, temporal_s, q_target):
        with tf.GradientTape() as tape:
            q_eval_all = self.eval_net([s, temporal_s], training=True)
            a_indices = tf.stack([tf.range(self.batch_size, dtype=tf.int32), tf.cast(q_target[:, 1], dtype=tf.int32)], axis=1)
            q_eval = tf.gather_nd(q_eval_all, a_indices)
            q_target_values = q_target[:, 0]
            loss = tf.keras.losses.MSE(q_target_values, q_eval)
        
        gradients = tape.gradient(loss, self.eval_net.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.eval_net.trainable_variables))
        return loss

    def learn(self):
        if self.learn_step_counter > 0 and self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.set_weights(self.eval_net.get_weights())

        min_memory = self.memory_size if self.memory_counter > self.memory_size else self.memory_counter
        if min_memory < self.sequence_length:
            return

        sample_index = np.random.choice(min_memory - self.sequence_length, size=self.batch_size)
        
        batch_memory = self.memory[sample_index, :self.n_features + 1 + 1 + self.n_features]
        
        temporal_s_batch = np.zeros((self.batch_size, self.sequence_length, self.n_temporal_features))
        temporal_s__batch = np.zeros((self.batch_size, self.sequence_length, self.n_temporal_features))
        temporal_mem_start = self.n_features + 1 + 1 + self.n_features
        
        for i, idx in enumerate(sample_index):
            mem_slice = self.memory[idx:idx+self.sequence_length, temporal_mem_start:]
            temporal_s_batch[i, :, :] = mem_slice[:, :self.n_temporal_features]
            temporal_s__batch[i, :, :] = mem_slice[:, self.n_temporal_features:]

        s = tf.constant(batch_memory[:, :self.n_features], dtype=tf.float32)
        s_ = tf.constant(batch_memory[:, -self.n_features:], dtype=tf.float32)
        temporal_s = tf.constant(temporal_s_batch, dtype=tf.float32)
        temporal_s_ = tf.constant(temporal_s__batch, dtype=tf.float32)
        
        q_next = self.target_net([s_, temporal_s_], training=False)
        q_eval4next = self.eval_net([s_, temporal_s_], training=False)
        
        q_target = np.zeros((self.batch_size, 2))
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:
            max_act4next = np.argmax(q_eval4next, axis=1)
            selected_q_next = q_next.numpy()[batch_index, max_act4next]
        else:
            selected_q_next = np.max(q_next.numpy(), axis=1)
            
        q_target[:, 0] = reward + self.gamma * selected_q_next
        q_target[:, 1] = eval_act_index
        q_target = tf.constant(q_target, dtype=tf.float32)
        
        cost = self._train_step(s, temporal_s, q_target)
        self.epsilon = min(self.epsilon_max, self.epsilon + self.epsilon_increment)
        self.learn_step_counter += 1
        return cost

    # --- KEEP ALL ORIGINAL HELPER METHODS BELOW ---
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