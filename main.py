from MEC_Env import MEC
from D3QN import DuelingDoubleDeepQNetwork
from Config import Config
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import shutil

def normalize(parameter, minimum, maximum):
    normalized_parameter = (parameter - minimum) / (maximum - minimum)
    return normalized_parameter

def QoE_Function(delay, max_delay, unfinish_task, ue_energy_state, ue_comp_energy, ue_trans_energy, edge_comp_energy, ue_idle_energy):
    edge_energy = next((e for e in edge_comp_energy if e != 0), 0)
    idle_energy = next((e for e in ue_idle_energy if e != 0), 0)

    energy_cons = ue_comp_energy + ue_trans_energy
    scaled_energy = normalize(energy_cons, 0, 20) * 10
    cost = 2 * ((ue_energy_state * delay) + ((1 - ue_energy_state) * scaled_energy))

    Reward = max_delay * 4

    if unfinish_task:
        QoE = -cost
    else:
        QoE = Reward - cost

    return QoE

def Drop_Count(ue_RL_list, episode):
    drrop_delay10 = 0 
    drrop = 0
    for time_index in range(100):   
        drrop = drrop + sum(env.unfinish_task[time_index])

    for i in range(len(ue_RL_list)):
        for j in range(len(ue_RL_list[i].delay_store[episode])):
            if ue_RL_list[i].delay_store[episode][j] == 10:
                drrop_delay10 = drrop_delay10 + 1

    return drrop

def Cal_QoE(ue_RL_list, episode):
    episode_sum_reward = sum(sum(ue_RL.reward_store[episode]) for ue_RL in ue_RL_list)
    avg_episode_sum_reward = episode_sum_reward / len(ue_RL_list)
    return avg_episode_sum_reward

def Cal_Delay(ue_RL_list, episode):
    avg_delay_in_episode = []
    for i in range(len(ue_RL_list)):
        for j in range(len(ue_RL_list[i].delay_store[episode])):
            if ue_RL_list[i].delay_store[episode][j] != 0:
                avg_delay_in_episode.append(ue_RL_list[i].delay_store[episode][j])
    avg_delay_in_episode = (sum(avg_delay_in_episode) / len(avg_delay_in_episode))
    return avg_delay_in_episode

def Cal_Energy(ue_RL_list, episode):
    energy_ue_list = [sum(ue_RL.energy_store[episode]) for ue_RL in ue_RL_list]
    avg_energy_in_episode = sum(energy_ue_list) / len(energy_ue_list)
    return avg_energy_in_episode

def train(ue_RL_list, NUM_EPISODE):
    avg_QoE_list = []
    avg_delay_list = []
    energy_cons_list = []
    num_drop_list = []
    avg_reward_list = []
    avg_reward_list_2 = []
    avg_delay_list_in_episode = []
    avg_energy_list_in_episode = []
    num_task_drop_list_in_episode = []
    RL_step = 0

    for episode in range(NUM_EPISODE):
        print("\n-*-**-***-*****-********-*************-********-*****-***-**-*-")
        print("Episode  :", episode)
        print("Epsilon  :", ue_RL_list[0].epsilon)

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

        test = 0 
        for i in range(len(bitarrive_size)):
            for j in range(len(bitarrive_size[i])):
                if bitarrive_size[i][j] != 0: 
                    test = test + 1

        print("Num_Task_Arrive: ", test)

        Check = []
        for i in range(len(bitarrive_size)):
            Check.append(sum(bitarrive_size[i]))

        # OBSERVATION MATRIX SETTING
        history = list()
        for time_index in range(env.n_time):
            history.append(list())
            for ue_index in range(env.n_ue):
                tmp_dict = {'observation': np.zeros(env.n_features),
                            'lstm': np.zeros(env.n_lstm_state),
                            'action': np.nan,
                            'observation_': np.zeros(env.n_features),
                            'lstm_': np.zeros(env.n_lstm_state)}
                history[time_index].append(tmp_dict)
        reward_indicator = np.zeros([env.n_time, env.n_ue])

        # INITIALIZE OBSERVATION
        observation_all, lstm_state_all = env.reset(bitarrive_size, bitarrive_dens)

        # TRAIN DRL
        while True:
            # PERFORM ACTION
            action_all = np.zeros([env.n_ue])
            for ue_index in range(env.n_ue):
                observation = np.squeeze(observation_all[ue_index, :])
                if np.sum(observation) == 0:
                    action_all[ue_index] = 0
                else:
                    action_all[ue_index] = ue_RL_list[ue_index].choose_action(observation)
                    if observation[0] != 0:
                        ue_RL_list[ue_index].do_store_action(episode, env.time_count, action_all[ue_index])

            # OBSERVE THE NEXT STATE AND PROCESS DELAY (REWARD)
            observation_all_, lstm_state_all_, done = env.step(action_all)

            # Update LSTM for each UE
            for ue_index in range(env.n_ue):
                ue_RL_list[ue_index].update_lstm(lstm_state_all_[ue_index, :])

            process_delay = env.process_delay
            unfinish_task = env.unfinish_task

            # STORE MEMORY; STORE TRANSITION IF THE TASK PROCESS DELAY IS JUST UPDATED
            for ue_index in range(env.n_ue):
                history[env.time_count - 1][ue_index]['observation'] = observation_all[ue_index, :]
                history[env.time_count - 1][ue_index]['lstm'] = np.squeeze(lstm_state_all[ue_index, :])
                history[env.time_count - 1][ue_index]['action'] = action_all[ue_index]
                history[env.time_count - 1][ue_index]['observation_'] = observation_all_[ue_index]
                history[env.time_count - 1][ue_index]['lstm_'] = np.squeeze(lstm_state_all_[ue_index, :])

                update_index = np.where((1 - reward_indicator[:, ue_index]) * process_delay[:, ue_index] > 0)[0]

                if len(update_index) != 0:
                    for update_ii in range(len(update_index)):
                        time_index = update_index[update_ii]
                        reward = QoE_Function(process_delay[time_index, ue_index],
                                            env.max_delay,
                                            unfinish_task[time_index, ue_index],
                                            env.ue_energy_state[ue_index],
                                            env.ue_comp_energy[time_index, ue_index],
                                            env.ue_tran_energy[time_index, ue_index],
                                            env.edge_comp_energy[time_index, ue_index],
                                            env.ue_idle_energy[time_index, ue_index])
                        
                        ue_RL_list[ue_index].store_transition(history[time_index][ue_index]['observation'],
                                                            history[time_index][ue_index]['lstm'],
                                                            history[time_index][ue_index]['action'],
                                                            reward,
                                                            history[time_index][ue_index]['observation_'],
                                                            history[time_index][ue_index]['lstm_'])
                        
                        ue_RL_list[ue_index].do_store_reward(episode, time_index, reward)
                        ue_RL_list[ue_index].do_store_delay(episode, time_index, process_delay[time_index, ue_index])
                        ue_RL_list[ue_index].do_store_energy(episode, time_index,
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
            if (RL_step > 200) and (RL_step % 10 == 0):
                for ue in range(env.n_ue):
                    ue_RL_list[ue].learn()

            # GAME ENDS
            if done:
                with open("Delay.txt", 'a') as f:
                    f.write('\n' + str(Cal_Delay(ue_RL_list, episode)))

                with open("Energy.txt", 'a') as f:
                    f.write('\n' + str(Cal_Energy(ue_RL_list, episode)))

                with open("QoE.txt", 'a') as f:
                    f.write('\n' + str(Cal_QoE(ue_RL_list, episode)))

                with open("Drop.txt", 'a') as f:
                    f.write('\n' + str(Drop_Count(ue_RL_list, episode)))

                # Save models periodically using TF2 methods
                if episode % 200 == 0 and episode != 0:
                    if not os.path.exists("models"):
                        os.mkdir("models")
                    model_dir = os.path.join("models", str(episode))
                    if not os.path.exists(model_dir):
                        os.mkdir(model_dir)
                    
                    for ue in range(env.n_ue):
                        model_path = os.path.join(model_dir, f"{ue}_X_model")
                        ue_RL_list[ue].save_model(model_path)
                        print(f"UE {ue} Network_model_saved\n")
                
                if episode % 999 == 0 and episode != 0:
                    model_dir = os.path.join("models", str(episode))
                    if not os.path.exists(model_dir):
                        os.mkdir(model_dir)
                    
                    for ue in range(env.n_ue):
                        model_path = os.path.join(model_dir, f"{ue}_X_model")
                        ue_RL_list[ue].save_model(model_path)
                        print(f"UE {ue} Network_model_saved\n")

                # Process energy calculations
                ue_bit_processed = sum(sum(env.ue_bit_processed))
                ue_comp_energy = sum(sum(env.ue_comp_energy))
                ue_bit_transmitted = sum(sum(env.ue_bit_transmitted))
                ue_tran_energy = sum(sum(env.ue_tran_energy))
                edge_bit_processed = sum(sum(env.edge_bit_processed))
                edge_comp_energy = sum(sum(env.edge_comp_energy))
                ue_idle_energy = sum(sum(env.ue_idle_energy))

                avg_delay = Cal_Delay(ue_RL_list, episode)
                avg_energy = Cal_Energy(ue_RL_list, episode)
                avg_QoE = Cal_QoE(ue_RL_list, episode)

                avg_QoE_list.append(avg_QoE)
                avg_delay_list.append(avg_delay)
                energy_cons_list.append(avg_energy)
                num_drop_list.append(env.drop_trans_count + env.drop_edge_count + env.drop_ue_count)
                avg_reward_list.append(-(Cal_QoE(ue_RL_list, episode)))

                # Plotting every 10 episodes
                # if episode % 10 == 0:
                #     avg_reward_list_2.append(sum(avg_reward_list[episode-10:episode]) / 10)
                #     avg_delay_list_in_episode.append(Cal_Delay(ue_RL_list, episode))
                #     avg_energy_list_in_episode.append(Cal_Energy(ue_RL_list, episode))

                #     # Create performance plots
                #     fig, axs = plt.subplots(4, 1, figsize=(10, 20))
                #     fig.suptitle('Performance Metrics Over Episodes', fontsize=16, y=0.92)

                #     axs[0].plot(avg_QoE_list, marker='o', linestyle='-', color='b', label='Avg QoE')
                #     axs[0].set_title('', fontsize=14)
                #     axs[0].set_ylabel('Average QoE')
                #     axs[0].set_xlabel('Episode')
                #     axs[0].grid(True, linestyle='--', alpha=0.7)
                #     axs[0].legend()

                #     axs[1].plot(avg_delay_list, marker='s', linestyle='-', color='g', label='Avg Delay')
                #     axs[1].set_title('', fontsize=14)
                #     axs[1].set_ylabel('Average Delay')
                #     axs[1].set_xlabel('Episode')
                #     axs[1].grid(True, linestyle='--', alpha=0.7)
                #     axs[1].legend()

                #     axs[2].plot(energy_cons_list, marker='^', linestyle='-', color='r', label='Energy Cons.')
                #     axs[2].set_title('', fontsize=14)
                #     axs[2].set_ylabel('Energy Consumption')
                #     axs[2].set_xlabel('Episode')
                #     axs[2].grid(True, linestyle='--', alpha=0.7)
                #     axs[2].legend()

                #     axs[3].plot(num_drop_list, marker='x', linestyle='-', color='m', label='Num Drops')
                #     axs[3].set_title('', fontsize=14)
                #     axs[3].set_ylabel('Number Drops')
                #     axs[3].set_xlabel('Episode')
                #     axs[3].grid(True, linestyle='--', alpha=0.7)
                #     axs[3].legend()

                #     plt.tight_layout()
                #     plt.subplots_adjust(top=0.9)
                #     plt.savefig('Performance_Chart.png', dpi=100)

                print("SystemPerformance: ---------------------------------------------------------------------")
                print("Num_Dropped   :  ", env.drop_trans_count+env.drop_edge_count+env.drop_ue_count, 
                      "[Trans_Drop: ", env.drop_trans_count, "Edge_Drop: ", env.drop_edge_count, "UE_Drop: ", env.drop_ue_count, "]")
                print("Avg_Delay     :  ", "%0.1f" % avg_delay)
                print("Avg_Energy    :  ", "%0.1f" % avg_energy)
                print("Avg_QoE       :  ", "%0.1f" % avg_QoE)
                print("EnergyCosumption: ----------------------------------------------------------------------")
                print("Local         :  ", "%0.1f" % ue_comp_energy, "[ue_bit_processed:", int(ue_bit_processed), "]")
                print("Trans         :  ", "%0.1f" % ue_tran_energy, "[ue_bit_transmitted:", int(ue_bit_transmitted), "]")
                print("Edges         :  ", "%0.1f" % sum(ue_idle_energy), "[edge_bit_processed :", int(sum(edge_bit_processed)), "]")

                break

if __name__ == "__main__":
    # GENERATE ENVIRONMENT
    env = MEC(Config.N_UE, Config.N_EDGE, Config.N_TIME, Config.N_COMPONENT, Config.MAX_DELAY)

    # GENERATE MULTIPLE CLASSES FOR RL
    ue_RL_list = list()
    for ue in range(Config.N_UE):
        ue_RL_list.append(DuelingDoubleDeepQNetwork(env.n_actions, env.n_features, env.n_lstm_state, env.n_time,
                                                    learning_rate=Config.LEARNING_RATE,
                                                    reward_decay=Config.REWARD_DECAY,
                                                    e_greedy=Config.E_GREEDY,
                                                    replace_target_iter=Config.N_NETWORK_UPDATE,  
                                                    memory_size=Config.MEMORY_SIZE))

    # LOAD Trained MODEL (if needed)
    # for ue in range(Config.N_UE):
    #     ue_RL_list[ue].Initialize(ue)
    #     ue_RL_list[ue].epsilon = 1

    # Create output files
    with open("Delay.txt", 'w') as f:
        pass
    with open("Energy.txt", 'w') as f:
        pass
    with open("QoE.txt", 'w') as f:
        pass
    with open("Drop.txt", 'w') as f:
        pass

    # TRAIN THE SYSTEM
    train(ue_RL_list, Config.N_EPISODE)
