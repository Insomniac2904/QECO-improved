# filename: main.py
from MEC_Env import MEC
from D3QN import DuelingDoubleDeepQNetwork
from Config import Config
import numpy as np
import os

model_dir_prefix = "models/patchTST_performance_mode"

def normalize(parameter, minimum, maximum):
    """Normalizes a parameter to a value between 0 and 1."""
    return (parameter - minimum) / (maximum - minimum)

def QoE_Function(delay, max_delay, unfinish_task, ue_energy_state, ue_comp_energy, ue_trans_energy, edge_comp_energy, ue_idle_energy):
    """Calculates the Quality of Experience (QoE) as the reward."""
    energy_cons = ue_comp_energy + ue_trans_energy + next((e for e in edge_comp_energy if e != 0), 0) + next((e for e in ue_idle_energy if e != 0), 0)
    scaled_energy = normalize(energy_cons, 0, 20) * 10
    cost = 2 * ((ue_energy_state * delay) + ((1 - ue_energy_state) * scaled_energy))
    reward = max_delay * 4
    return reward - cost if not unfinish_task else -cost

def Drop_Count(env):
    """Counts the number of dropped tasks."""
    return sum(env.unfinish_task.flatten())

def Cal_QoE(ue_RL_list, episode):
    """Calculates the average QoE for an episode."""
    rewards = [sum(ue_RL.reward_store[episode]) for ue_RL in ue_RL_list if episode < len(ue_RL.reward_store)]
    return sum(rewards) / len(ue_RL_list) if rewards else 0

def Cal_Delay(ue_RL_list, episode):
    """Calculates the average delay for an episode."""
    all_delays = [
        delay for ue_RL in ue_RL_list if episode < len(ue_RL.delay_store)
        for delay in ue_RL.delay_store[episode] if delay != 0
    ]
    return sum(all_delays) / len(all_delays) if all_delays else 0

def Cal_Energy(ue_RL_list, episode):
    """Calculates the average energy consumption for an episode."""
    energy_ue_list = [sum(ue_RL.energy_store[episode]) for ue_RL in ue_RL_list if episode < len(ue_RL.energy_store)]
    return sum(energy_ue_list) / len(energy_ue_list) if energy_ue_list else 0

def train(env, ue_RL_list, NUM_EPISODE):
    """Main training loop."""
    rl_step = 0
    for episode in range(NUM_EPISODE):
        print("\n" + "-*" * 20)
        print(f"Episode: {episode}, Epsilon: {ue_RL_list[0].epsilon:.4f}")

        bitarrive_size = np.random.uniform(env.min_arrive_size, env.max_arrive_size, size=[env.n_time, env.n_ue])
        bitarrive_size *= (np.random.uniform(0, 1, size=[env.n_time, env.n_ue]) < env.task_arrive_prob)
        bitarrive_size[-env.max_delay:, :] = 0
        
        bitarrive_dens = np.zeros_like(bitarrive_size)
        task_indices = np.where(bitarrive_size != 0)
        bitarrive_dens[task_indices] = np.random.choice(Config.TASK_COMP_DENS, size=len(task_indices[0]))
        print(f"Num_Task_Arrive: {np.count_nonzero(bitarrive_size)}")

        history = [[{
            'observation': np.zeros(env.n_features), 'temporal': np.zeros(env.n_temporal_features),
            'action': np.nan,
            'observation_': np.zeros(env.n_features), 'temporal_': np.zeros(env.n_temporal_features)
        } for _ in range(env.n_ue)] for _ in range(env.n_time)]
        reward_indicator = np.zeros([env.n_time, env.n_ue])
        
        observation_all, temporal_state_all = env.reset(bitarrive_size, bitarrive_dens)

        # --- Training Loop ---
        while True:
            action_all = np.zeros(env.n_ue)
            for ue_index in range(env.n_ue):
                observation = np.squeeze(observation_all[ue_index, :])
                if observation.any():
                    action_all[ue_index] = ue_RL_list[ue_index].choose_action(observation)
                    if observation[0] != 0:
                        ue_RL_list[ue_index].do_store_action(episode, env.time_count, action_all[ue_index])
                else:
                    action_all[ue_index] = 0

            observation_all_, temporal_state_all_, done = env.step(action_all)

            for ue_index in range(env.n_ue):
                ue_RL_list[ue_index].update_temporal_history(temporal_state_all_[ue_index, :])

            time_counter = env.time_count - 1
            for ue_index in range(env.n_ue):
                history[time_counter][ue_index].update({
                    'observation': observation_all[ue_index, :], 'temporal': np.squeeze(temporal_state_all[ue_index, :]),
                    'action': action_all[ue_index], 'observation_': observation_all_[ue_index, :],
                    'temporal_': np.squeeze(temporal_state_all_[ue_index, :])
                })

                update_indices = np.where((1 - reward_indicator[:, ue_index]) * env.process_delay[:, ue_index] > 0)[0]
                for time_idx in update_indices:
                    reward = QoE_Function(
                        env.process_delay[time_idx, ue_index], env.max_delay, env.unfinish_task[time_idx, ue_index],
                        env.ue_energy_state[ue_index], env.ue_comp_energy[time_idx, ue_index],
                        env.ue_tran_energy[time_idx, ue_index], env.edge_comp_energy[time_idx, ue_index, :],
                        env.ue_idle_energy[time_idx, ue_index, :]
                    )
                    hist = history[time_idx][ue_index]
                    if not np.isnan(hist['action']):
                        ue_RL_list[ue_index].store_transition(
                            hist['observation'], hist['temporal'], hist['action'], reward, hist['observation_'], hist['temporal_']
                        )
                        ue_RL_list[ue_index].do_store_reward(episode, time_idx, reward)
                        ue_RL_list[ue_index].do_store_delay(episode, time_idx, env.process_delay[time_idx, ue_index])
                        ue_RL_list[ue_index].do_store_energy(
                            episode, time_idx, env.ue_comp_energy[time_idx, ue_index], env.ue_tran_energy[time_idx, ue_index],
                            env.edge_comp_energy[time_idx, ue_index, :], env.ue_idle_energy[time_idx, ue_index, :]
                        )
                        reward_indicator[time_idx, ue_index] = 1

            rl_step += 1
            observation_all = observation_all_
            temporal_state_all = temporal_state_all_

            if rl_step > 200 and rl_step % 10 == 0:
                for ue in range(env.n_ue):
                    ue_RL_list[ue].learn()
            
            if done:
                break

        for ue_index in range(env.n_ue):
            update_indices = np.where((1 - reward_indicator[:, ue_index]) * env.process_delay[:, ue_index] > 0)[0]
            for time_idx in update_indices:
                reward = QoE_Function(
                    env.process_delay[time_idx, ue_index], env.max_delay, env.unfinish_task[time_idx, ue_index],
                    env.ue_energy_state[ue_index], env.ue_comp_energy[time_idx, ue_index],
                    env.ue_tran_energy[time_idx, ue_index], env.edge_comp_energy[time_idx, ue_index, :],
                    env.ue_idle_energy[time_idx, ue_index, :]
                )
                hist = history[time_idx][ue_index]
                if not np.isnan(hist['action']):
                    ue_RL_list[ue_index].store_transition(
                        hist['observation'], hist['temporal'], hist['action'], reward, hist['observation_'], hist['temporal_']
                    )
                    ue_RL_list[ue_index].do_store_reward(episode, time_idx, reward)
                    ue_RL_list[ue_index].do_store_delay(episode, time_idx, env.process_delay[time_idx, ue_index])
                    ue_RL_list[ue_index].do_store_energy(
                        episode, time_idx, env.ue_comp_energy[time_idx, ue_index], env.ue_tran_energy[time_idx, ue_index],
                        env.edge_comp_energy[time_idx, ue_index, :], env.ue_idle_energy[time_idx, ue_index, :]
                    )
                    reward_indicator[time_idx, ue_index] = 1

        avg_delay = Cal_Delay(ue_RL_list, episode)
        avg_energy = Cal_Energy(ue_RL_list, episode)
        avg_QoE = Cal_QoE(ue_RL_list, episode)
        num_dropped = Drop_Count(env)
        
        print(f"Num_Dropped: {num_dropped} [Trans: {env.drop_trans_count}, Edge: {env.drop_edge_count}, UE: {env.drop_ue_count}]")
        print(f"Avg_Delay: {avg_delay:.4f}, Avg_Energy: {avg_energy:.4f}, Avg_QoE: {avg_QoE:.4f}")

        with open("Delay_modified_random.txt", 'a') as f: f.write(f'{avg_delay}\n')
        with open("Energy_modified_random.txt", 'a') as f: f.write(f'{avg_energy}\n')
        with open("QoE_modified_random.txt", 'a') as f: f.write(f'{avg_QoE}\n')
        with open("Drop_modified_random.txt", 'a') as f: f.write(f'{num_dropped}\n')

        if episode > 0 and episode % 800 == 0:
            model_dir = os.path.join(model_dir_prefix, str(episode))
            os.makedirs(model_dir, exist_ok=True)
            print(f"\n--- Saving models at episode {episode} ---")
            for ue_idx, ue_rl in enumerate(ue_RL_list):
                model_path = os.path.join(model_dir, f"{ue_idx}_X_model")
                ue_rl.save_model(model_path)
            print(f"--- Models saved to {model_dir} ---\n")

if __name__ == "__main__":
    env = MEC(Config.N_UE, Config.N_EDGE, Config.N_TIME, Config.N_COMPONENT, Config.MAX_DELAY)

    ue_RL_list = [
        DuelingDoubleDeepQNetwork(
            n_actions=env.n_actions, 
            n_features=env.n_features, 
            n_temporal_features=env.n_temporal_features, 
            n_time=env.n_time,
            learning_rate=Config.LEARNING_RATE,
            reward_decay=Config.REWARD_DECAY,
            e_greedy=Config.E_GREEDY,
            replace_target_iter=Config.N_NETWORK_UPDATE,
            memory_size=Config.MEMORY_SIZE
        ) for _ in range(Config.N_UE)
    ]
    
    train(env, ue_RL_list, Config.N_EPISODE)