from MEC_Env import MEC
from D3QN import DuelingDoubleDeepQNetwork
from Config import Config
# import matplotlib.pyplot as plt
import numpy as np
import random
import os
import shutil

# --- Step 2: Reward Shaping Function ---
def calculate_immediate_penalty(env, ue_index, action, observation, lstm_history):
    """
    Calculates a small, immediate penalty for an action based on the current state.
    This penalty is added to the final reward to "shape" it.
    
    observation: [task_size, t_ue_comp, t_ue_tran, [b_edge], ue_energy_state]
    lstm_history: (n_steps, n_edge)
    """
    penalty = 0.0
    
    # Penalty 1: Processing locally when local queue is already backlogged
    if action == 0:
        local_queue_delay_slots = observation[1] # t_ue_comp (feature index 1)
        if local_queue_delay_slots > 1: # Already has a task
            # Penalize based on how long the queue is, but keep it small
            penalty -= 0.1 * local_queue_delay_slots 
    
    # Penalty 2: Offloading to a congested server
    elif action > 0:
        edge_index = int(action - 1)
        if edge_index < env.n_edge:
            # Get *current* congestion from the *last* step of the LSTM history
            current_congestion = lstm_history[-1, edge_index] 
            
            # Penalize if congestion is higher than the average load
            # (e.g., if more than 10 UEs are on this server)
            avg_load_per_server = env.n_ue / env.n_edge
            if current_congestion > avg_load_per_server: 
                penalty -= 0.2 * (current_congestion - avg_load_per_server) # Penalize based on congestion level
        
    return penalty
# --- End Step 2 Function ---


def normalize(parameter, minimum, maximum):
    # Handle division by zero if max == min
    if (maximum - minimum) == 0:
        return 0
    normalized_parameter = (parameter - minimum) / (maximum - minimum)
    return normalized_parameter


def QoE_Function(delay, max_delay, unfinish_task, ue_energy_state, ue_comp_energy, ue_trans_energy, edge_comp_energy, ue_idle_energy):
    edge_energy = next((e for e in edge_comp_energy if e != 0), 0)
    idle_energy = next((e for e in ue_idle_energy if e != 0), 0)

    energy_cons = ue_comp_energy + ue_trans_energy  # + edge_energy + idle_energy
    
    # Normalize energy and delay
    # Max energy can be high, let's set a reasonable upper bound
    scaled_energy = normalize(energy_cons, 0, 20) * 10 
    scaled_delay = normalize(delay, 0, max_delay) * 10
    
    # Cost function with energy state weighting
    # We use (1 - ue_energy_state) for energy because 0.25 (power-saving) should care *more* about energy
    # But your original formula seems to be (0.25 * delay) + (0.75 * energy)
    # Let's stick to your original logic:
    cost = 2 * ((ue_energy_state * scaled_delay) + ((1 - ue_energy_state) * scaled_energy))

    Reward = max_delay * 4 # Max possible reward if cost is 0

    if unfinish_task:
        QoE = -cost # Penalty
    else:
        QoE = Reward - cost

    return QoE


def Drop_Count(ue_RL_list, episode):
    drrop = 0
    # We check the actual recorded delays
    for i in range(len(ue_RL_list)):
        if episode < len(ue_RL_list[i].delay_store):
            for j in range(len(ue_RL_list[i].delay_store[episode])):
                # Check for max delay, which indicates a drop
                if ue_RL_list[i].delay_store[episode][j] == Config.MAX_DELAY:
                    drrop = drrop + 1
    return drrop


def Cal_QoE(ue_RL_list, episode):
    episode_sum_reward = 0
    num_ues_with_rewards = 0
    for ue_RL in ue_RL_list:
        if episode < len(ue_RL.reward_store):
            episode_sum_reward += sum(ue_RL.reward_store[episode])
            num_ues_with_rewards += 1
            
    if num_ues_with_rewards == 0:
        return 0
        
    avg_episode_sum_reward = episode_sum_reward / num_ues_with_rewards
    return avg_episode_sum_reward


def Cal_Delay(ue_RL_list, episode):
    all_delays_in_episode = []
    for i in range(len(ue_RL_list)):
        if episode < len(ue_RL_list[i].delay_store):
            for j in range(len(ue_RL_list[i].delay_store[episode])):
                if ue_RL_list[i].delay_store[episode][j] > 0: # Only count processed tasks
                    all_delays_in_episode.append(ue_RL_list[i].delay_store[episode][j])
                    
    if not all_delays_in_episode: # Avoid division by zero
        return 0
        
    avg_delay_in_episode = (sum(all_delays_in_episode) / len(all_delays_in_episode))
    return avg_delay_in_episode


def Cal_Energy(ue_RL_list, episode):
    energy_ue_list = []
    for ue_RL in ue_RL_list:
        if episode < len(ue_RL.energy_store):
            energy_ue_list.append(sum(ue_RL.energy_store[episode]))
            
    if not energy_ue_list: # Avoid division by zero
        return 0
        
    avg_energy_in_episode = sum(energy_ue_list) / len(energy_ue_list)
    return avg_energy_in_episode


def train(ue_RL_list, NUM_EPISODE):
    avg_QoE_list = []
    avg_delay_list = []
    energy_cons_list = []
    num_drop_list = []
    
    RL_step = 0
    n_lstm_step = ue_RL_list[0].n_lstm_step # Get from agent

    for episode in range(NUM_EPISODE):
        print("\n-*-**-***-*****-********-*************-********-*****-***-**-*-")
        print(f"Episode  : {episode}")
        # --- Restore Epsilon-Greedy Print ---
        print(f"Epsilon  : {ue_RL_list[0].epsilon:.4f}")
        # --- End Restore ---
        print(f"LR       : {ue_RL_list[0].lr_schedule(RL_step).numpy():.7f}")
        print(f"PER Beta : {ue_RL_list[0].beta:.4f}")

        # BITRATE ARRIVAL
        bitarrive_size = np.random.uniform(env.min_arrive_size, env.max_arrive_size, size=[env.n_time, env.n_ue])
        task_prob = env.task_arrive_prob
        bitarrive_size = bitarrive_size * (np.random.uniform(0, 1, size=[env.n_time, env.n_ue]) < task_prob)
        bitarrive_size[-env.max_delay:, :] = np.zeros([env.max_delay, env.n_ue])

        bitarrive_dens = np.zeros([env.n_time, env.n_ue])
        for i in range(len(bitarrive_size)):
            for j in range(len(bitarrive_size[i])):
                if bitarrive_size[i][j] != 0:
                    bitarrive_dens[i][j] = Config.TASK_COMP_DENS[np.random.randint(0, len(Config.TASK_COMP_DENS))]

        test = np.count_nonzero(bitarrive_size)
        print(f"Num_Task_Arrive: {test}")

        # OBSERVATION MATRIX SETTING
        history = list()
        for time_index in range(env.n_time):
            history.append(list())
            for ue_index in range(env.n_ue):
                tmp_dict = {'observation': np.zeros(env.n_features),
                            'lstm_history': np.zeros((n_lstm_step, env.n_lstm_state)), # Store the full history
                            'action': np.nan,
                            'observation_': np.zeros(env.n_features),
                            'lstm_history_': np.zeros((n_lstm_step, env.n_lstm_state)),
                            'immediate_penalty': 0.0 # --- Step 2: Add penalty field ---
                            }
                history[time_index].append(tmp_dict)
        reward_indicator = np.zeros([env.n_time, env.n_ue])

        # INITIALIZE OBSERVATION
        observation_all, lstm_state_all = env.reset(bitarrive_size, bitarrive_dens)
        
        # Initialize LSTM history for all agents
        for ue_index in range(env.n_ue):
            ue_RL_list[ue_index].lstm_history.clear()
            for _ in range(n_lstm_step):
                ue_RL_list[ue_index].lstm_history.append(np.zeros(env.n_lstm_state))
            ue_RL_list[ue_index].update_lstm(lstm_state_all[ue_index, :])


        # TRAIN DRL
        while True:
            # PERFORM ACTION
            action_all = np.zeros([env.n_ue], dtype=int)
            for ue_index in range(env.n_ue):
                observation = np.squeeze(observation_all[ue_index, :])
                
                # Get the agent's current LSTM history
                current_lstm_history = ue_RL_list[ue_index].get_current_lstm_history()
                
                if np.sum(observation) == 0:
                    action_all[ue_index] = 0 # No task, default action is local
                else:
                    action_all[ue_index] = ue_RL_list[ue_index].choose_action(observation)
                    
                    # Store action if a task is arriving
                    if observation[0] != 0:
                        ue_RL_list[ue_index].do_store_action(episode, env.time_count, action_all[ue_index])
                        
                        # --- STEP 2: Calculate and store immediate penalty ---
                        time_now_action = env.time_count
                        penalty = calculate_immediate_penalty(env, ue_index, action_all[ue_index], 
                                                              observation, current_lstm_history)
                        
                        # Store penalty at the time the task arrives
                        if time_now_action < env.n_time:
                            history[time_now_action][ue_index]['immediate_penalty'] = penalty
                        # --- END STEP 2 ---


            # OBSERVE THE NEXT STATE AND PROCESS DELAY (REWARD)
            observation_all_, lstm_state_all_, done = env.step(action_all)

            # Update LSTM history for all agents
            for ue_index in range(env.n_ue):
                ue_RL_list[ue_index].update_lstm(lstm_state_all_[ue_index, :])

            process_delay = env.process_delay
            unfinish_task = env.unfinish_task

            # STORE MEMORY
            for ue_index in range(env.n_ue):
                time_now_store = env.time_count - 1
                if time_now_store < 0: continue
                
                # Store history for *this* step
                history[time_now_store][ue_index]['observation'] = observation_all[ue_index, :]
                history[time_now_store][ue_index]['action'] = action_all[ue_index]
                history[time_now_store][ue_index]['observation_'] = observation_all_[ue_index]
                
                # Store the LSTM history *before* the action and *after*
                # To do this, we need to go back one step for the 's' history
                if time_now_store > 0:
                     history[time_now_store][ue_index]['lstm_history'] = history[time_now_store - 1][ue_index]['lstm_history_']
                else:
                     # For the very first step, history is all zeros
                     history[time_now_store][ue_index]['lstm_history'] = np.zeros((n_lstm_step, env.n_lstm_state))
                
                history[time_now_store][ue_index]['lstm_history_'] = ue_RL_list[ue_index].get_current_lstm_history()
                
                # Check if any tasks *finished* this step and store their transitions
                update_index = np.where((1 - reward_indicator[:, ue_index]) * process_delay[:, ue_index] > 0)[0]

                if len(update_index) != 0:
                    for update_ii in range(len(update_index)):
                        time_index = update_index[update_ii] # This is the time the task *arrived*

                        # --- Step 2: Get final QoE and add immediate penalty ---
                        final_reward = QoE_Function(process_delay[time_index, ue_index],
                                                     env.max_delay,
                                                     unfinish_task[time_index, ue_index],
                                                     env.ue_energy_state[ue_index],
                                                     env.ue_comp_energy[time_index, ue_index],
                                                     env.ue_tran_energy[time_index, ue_index],
                                                     env.edge_comp_energy[time_index, ue_index],
                                                     env.ue_idle_energy[time_index, ue_index])
                        
                        immediate_penalty = history[time_index][ue_index]['immediate_penalty']
                        reward = final_reward + immediate_penalty # Total shaped reward
                        # --- End Step 2 ---

                        ue_RL_list[ue_index].store_transition(
                            history[time_index][ue_index]['observation'],
                            history[time_index][ue_index]['lstm_history'],
                            history[time_index][ue_index]['action'],
                            reward, # Store the combined, shaped reward
                            history[time_index][ue_index]['observation_'],
                            history[time_index][ue_index]['lstm_history_'])
                        
                        ue_RL_list[ue_index].do_store_reward(episode, time_index, reward)
                        
                        ue_RL_list[ue_index].do_store_delay(episode, time_index,
                                                             process_delay[time_index, ue_index])

                        ue_RL_list[ue_index].do_store_energy(
                            episode,
                            time_index,
                            env.ue_comp_energy[time_index, ue_index],
                            env.ue_tran_energy[time_index, ue_index],
                            env.edge_comp_energy[time_index, ue_index],
                            env.ue_idle_energy[time_index, ue_index])

                        reward_indicator[time_index, ue_index] = 1

            # ADD STEP
            RL_step += 1

            # UPDATE OBSERVATION
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            # CONTROL LEARNING START TIME AND FREQUENCY
            # Start learning after 200 steps and learn every 10 steps
            if (RL_step > 200) and (RL_step % 10 == 0):
                for ue in range(env.n_ue):
                    ue_RL_list[ue].learn()

            # GAME ENDS
            if done:
                # Calculate metrics for this episode
                avg_delay = Cal_Delay(ue_RL_list, episode)
                avg_energy = Cal_Energy(ue_RL_list, episode)
                avg_QoE = Cal_QoE(ue_RL_list, episode)
                num_drop = Drop_Count(ue_RL_list, episode)

                # Store metrics
                avg_QoE_list.append(avg_QoE)
                avg_delay_list.append(avg_delay)
                energy_cons_list.append(avg_energy)
                num_drop_list.append(num_drop)
                
                # Write to files
                with open("Delay.txt", 'a') as f:
                    f.write('\n' + str(avg_delay))
                with open("Energy.txt", 'a') as f:
                    f.write('\n' + str(avg_energy))
                with open("QoE.txt", 'a') as f:
                    f.write('\n' + str(avg_QoE))
                with open("Drop.txt", 'a') as f:
                    f.write('\n' + str(num_drop))

                # Save model checkpoints
                if episode % 200 == 0 and episode != 0:
                    model_dir = os.path.join("models", str(episode))
                    os.makedirs(model_dir, exist_ok=True)
                    for ue in range(env.n_ue):
                        model_prefix = os.path.join(model_dir, f"{ue}_X_model")
                        ue_RL_list[ue].save_model(model_prefix)
                        print(f"UE {ue} Network_model_saved\n")

                # Plotting logic
                # if episode % 10 == 0 and episode != 0:
                #     fig, axs = plt.subplots(4, 1, figsize=(10, 20))
                #     fig.suptitle('Performance Metrics Over Episodes', fontsize=16, y=0.92)

                #     # Subplot for Average QoE
                #     axs[0].plot(avg_QoE_list, marker='o', linestyle='-', color='b', label='Avg QoE')
                #     axs[0].set_ylabel('Average QoE')
                #     axs[0].grid(True, linestyle='--', alpha=0.7)
                #     axs[0].legend()

                #     # Subplot for Average Delay
                #     axs[1].plot(avg_delay_list, marker='s', linestyle='-', color='g', label='Avg Delay')
                #     axs[1].set_ylabel('Average Delay')
                #     axs[1].grid(True, linestyle='--', alpha=0.7)
                #     axs[1].legend()

                #     # Subplot for Energy Consumption
                #     axs[2].plot(energy_cons_list, marker='^', linestyle='-', color='r', label='Energy Cons.')
                #     axs[2].set_ylabel('Energy Consumption')
                #     axs[2].grid(True, linestyle='--', alpha=0.7)
                #     axs[2].legend()

                #     # Subplot for Number of Drops
                #     axs[3].plot(num_drop_list, marker='x', linestyle='-', color='m', label='Num Drops')
                #     axs[3].set_ylabel('Number Drops')
                #     axs[3].set_xlabel('Episode')
                #     axs[3].grid(True, linestyle='--', alpha=0.7)
                #     axs[3].legend()

                #     plt.tight_layout()
                #     plt.subplots_adjust(top=0.9)
                #     plt.savefig('Performance_Chart.png', dpi=100)
                #     plt.close(fig) # Close the figure to save memory


                print("SystemPerformance: ---------------------------------------------------------------------")
                print(f"Num_Dropped     : {num_drop:.0f}")
                print(f"Avg_Delay       : {avg_delay:.1f}")
                print(f"Avg_Energy      : {avg_energy:.1f}")
                print(f"Avg_QoE         : {avg_QoE:.1f}")
                print("----------------------------------------------------------------------------------------")
                
                break # Training Finished


if __name__ == "__main__":
    # Create model directory
    if not os.path.exists("models"):
        os.mkdir("models")
        
    # GENERATE ENVIRONMENT
    env = MEC(Config.N_UE, Config.N_EDGE, Config.N_TIME, Config.N_COMPONENT, Config.MAX_DELAY)

    # GENERATE MULTIPLE CLASSES FOR RL
    ue_RL_list = list()
    for ue in range(Config.N_UE):
        ue_RL_list.append(DuelingDoubleDeepQNetwork(
            env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
            # --- Step 1: LR Schedule ---
            learning_rate_start=Config.LEARNING_RATE_START,
            learning_rate_end=Config.LEARNING_RATE_END,
            learning_rate_decay_steps=Config.LEARNING_RATE_DECAY_STEPS,
            batch_size=Config.BATCH_SIZE,
            # --- End Step 1 ---
            reward_decay=Config.REWARD_DECAY,
            # --- Restored Epsilon-Greedy ---
            e_greedy=Config.E_GREEDY,
            e_greedy_increment=Config.E_GREEDY_INCREMENT,
            # --- End Restore ---
            replace_target_iter=Config.N_NETWORK_UPDATE,
            memory_size=Config.MEMORY_SIZE,
            # --- Step 3: PER Parameters ---
            per_alpha=Config.PER_ALPHA,
            per_beta_start=Config.PER_BETA_START,
            per_beta_anneal_steps=Config.PER_BETA_ANNEAL_STEPS,
            per_epsilon=Config.PER_EPSILON,
            # --- End Step 3 ---
            n_lstm_step=10, # Default n_lstm_step
            dueling=True,
            double_q=True,
            N_L1=20, # Default N_L1
            N_lstm=20 # Default N_lstm
        ))

    # Clean old log files
    with open("Delay.txt", 'w') as f: f.write("Avg_Delay")
    with open("Energy.txt", 'w') as f: f.write("Avg_Energy")
    with open("QoE.txt", 'w') as f: f.write("Avg_QoE")
    with open("Drop.txt", 'w') as f: f.write("Num_Drop")

    # TRAIN THE SYSTEM
    try:
        train(ue_RL_list, Config.N_EPISODE)
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final models...")
        model_dir = os.path.join("models", "final_interrupt")
        os.makedirs(model_dir, exist_ok=True)
        for ue in range(env.n_ue):
            model_prefix = os.path.join(model_dir, f"{ue}_X_model")
            ue_RL_list[ue].save_model(model_prefix)
        print("Final models saved.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()