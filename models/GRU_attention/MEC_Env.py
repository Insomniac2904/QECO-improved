from Config import Config
import numpy as np
import random
import math
import queue

class MEC:
    def __init__(self, num_ue, num_edge, num_time, num_component, max_delay):
        # Initialize variables
        self.n_ue           = num_ue
        self.n_edge         = num_edge
        self.n_time         = num_time
        self.n_component    = num_component
        self.max_delay      = max_delay
        self.duration       = Config.DURATION
        self.ue_p_comp      = Config.UE_COMP_ENERGY
        self.ue_p_tran      = Config.UE_TRAN_ENERGY
        self.ue_p_idle      = Config.UE_IDLE_ENERGY
        self.edge_p_comp    = Config.EDGE_COMP_ENERGY
        

        self.time_count      = 0
        self.task_count_ue   = 0
        self.task_count_edge = 0
        self.n_actions       = 1 + self.n_edge
        self.n_features      = 1 + 1 + 1 + 1 + self.n_edge
        self.n_lstm_state    = self.n_edge

        self.drop_trans_count = 0
        self.drop_edge_count = 0
        self.drop_ue_count = 0
    

        # Computation and transmission capacities
        self.comp_cap_ue     = Config.UE_COMP_CAP * np.ones(self.n_ue) * self.duration
        self.comp_cap_edge   = Config.EDGE_COMP_CAP * np.ones([self.n_edge]) * self.duration
        self.tran_cap_ue     = Config.UE_TRAN_CAP * np.ones([self.n_ue, self.n_edge]) * self.duration
        self.n_cycle = 1
        self.task_arrive_prob = Config.TASK_ARRIVE_PROB
        self.max_arrive_size   = Config.TASK_MAX_SIZE
        self.min_arrive_size   = Config.TASK_MIN_SIZE
        self.arrive_task_size_set    = np.arange(self.min_arrive_size, self.max_arrive_size, 0.1)
        #self.energy_state_set   = np.arange(0.25,1, 0.25) 
        self.ue_energy_state = [Config.UE_ENERGY_STATE[np.random.randint(0,len(Config.UE_ENERGY_STATE))] for ue in range(self.n_ue)]
        self.arrive_task_size    = np.zeros([self.n_time, self.n_ue])
        self.arrive_task_dens    = np.zeros([self.n_time, self.n_ue])




        #print(self.energy_state_set)

        #print(self.ue_energy_state)


        #self.comp_density=0.297



        self.n_task = int(self.n_time * self.task_arrive_prob)

        # Task delay and energy-related arrays
        self.process_delay = np.zeros([self.n_time, self.n_ue])
        self.ue_bit_processed = np.zeros([self.n_time, self.n_ue])
        self.edge_bit_processed = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_bit_transmitted = np.zeros([self.n_time, self.n_ue])
        self.ue_comp_energy = np.zeros([self.n_time, self.n_ue])
        self.edge_comp_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_idle_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_tran_energy = np.zeros([self.n_time, self.n_ue])
        self.unfinish_task = np.zeros([self.n_time, self.n_ue])
        self.process_delay_trans = np.zeros([self.n_time, self.n_ue])
        self.edge_drop = np.zeros([self.n_ue, self.n_edge])

        # Queue information initialization
        self.t_ue_comp = -np.ones([self.n_ue])
        self.t_ue_tran = -np.ones([self.n_ue])
        self.b_edge_comp = np.zeros([self.n_ue, self.n_edge])

        # Queue initialization
        self.ue_computation_queue = [queue.Queue() for _ in range(self.n_ue)]
        self.ue_transmission_queue = [queue.Queue() for _ in range(self.n_ue)]
        self.edge_computation_queue = [[queue.Queue() for _ in range(self.n_edge)] for _ in range(self.n_ue)]
        self.edge_ue_m = np.zeros(self.n_edge)
        self.edge_ue_m_observe = np.zeros(self.n_edge)

        # Task indicator initialization
        self.local_process_task = [{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'TIME': np.nan, 'EDGE': np.nan, 'REMAIN': np.nan, 'DENS': np.nan} for _ in range(self.n_ue)]
        self.local_transmit_task = [{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                     'TIME': np.nan, 'EDGE': np.nan, 'REMAIN': np.nan, 'DENS': np.nan} for _ in range(self.n_ue)]
        self.edge_process_task = [[{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'TIME': np.nan, 'REMAIN': np.nan, 'DENS': np.nan} for _ in range(self.n_edge)] for _ in range(self.n_ue)]

        self.task_history = [[] for _ in range(self.n_ue)]

    def reset(self, arrive_task_size, arrive_task_dens):
    
        self.drop_trans_count = 0
        self.drop_edge_count = 0
        self.drop_ue_count = 0

        # Reset variables and queues
        self.task_history = [[] for _ in range(self.n_ue)]
        self.UE_TASK = [-1] * self.n_ue
        self.drop_edge_count = 0

        self.arrive_task_size = arrive_task_size
        self.arrive_task_dens = arrive_task_dens

        self.time_count = 0

        self.local_process_task = []
        self.local_transmit_task = []
        self.edge_process_task = []



        self.ue_computation_queue = [queue.Queue() for _ in range(self.n_ue)]
        self.ue_transmission_queue = [queue.Queue() for _ in range(self.n_ue)]
        self.edge_computation_queue = [[queue.Queue() for _ in range(self.n_edge)] for _ in range(self.n_ue)]
        
        self.t_ue_comp = -np.ones([self.n_ue])
        self.t_ue_tran = -np.ones([self.n_ue])
        self.b_edge_comp = np.zeros([self.n_ue, self.n_edge])

        self.process_delay = np.zeros([self.n_time, self.n_ue])
        self.ue_bit_processed = np.zeros([self.n_time, self.n_ue])
        self.edge_bit_processed = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_bit_transmitted = np.zeros([self.n_time, self.n_ue])
        self.ue_comp_energy = np.zeros([self.n_time, self.n_ue])
        self.edge_comp_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_idle_energy = np.zeros([self.n_time, self.n_ue, self.n_edge])
        self.ue_tran_energy = np.zeros([self.n_time, self.n_ue])
        self.unfinish_task = np.zeros([self.n_time, self.n_ue])
        self.process_delay_trans = np.zeros([self.n_time, self.n_ue])
        self.edge_drop = np.zeros([self.n_ue, self.n_edge])

        self.local_process_task = [{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'TIME': np.nan, 'EDGE': np.nan, 'REMAIN': np.nan, 'DENS': np.nan} for _ in range(self.n_ue)]
        self.local_transmit_task = [{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                     'TIME': np.nan, 'EDGE': np.nan, 'REMAIN': np.nan, 'DENS': np.nan} for _ in range(self.n_ue)]
        self.edge_process_task = [[{'DIV': np.nan, 'UE_ID': np.nan, 'TASK_ID': np.nan, 'SIZE': np.nan,
                                    'TIME': np.nan, 'REMAIN': np.nan, 'DENS': np.nan} for _ in range(self.n_edge)] for _ in range(self.n_ue)]

        # Initial observation and LSTM state
        UEs_OBS = np.zeros([self.n_ue, self.n_features])
        for ue_index in range(self.n_ue):
            if self.arrive_task_size[self.time_count, ue_index] != 0:
                UEs_OBS[ue_index, :] = np.hstack([
                    self.arrive_task_size[self.time_count, ue_index], self.t_ue_comp[ue_index],
                    self.t_ue_tran[ue_index],
                    np.squeeze(self.b_edge_comp[ue_index, :]),
                    self.ue_energy_state[ue_index]])

        UEs_lstm_state = np.zeros([self.n_ue, self.n_lstm_state])

        return UEs_OBS, UEs_lstm_state

    
    # perform action, observe state and delay (several steps later)
    def step(self, action):
    

        ue_action_local = np.zeros([self.n_ue], np.int32)
        ue_action_offload = np.zeros([self.n_ue], np.int32)

        for ue_index in range(self.n_ue):
            ue_action = action[ue_index]
            ue_action_offload[ue_index] = int(ue_action - 1)
            if ue_action == 0:
                ue_action_local[ue_index] = 1

        # COMPUTATION QUEUE UPDATE ===================
        for ue_index in range(self.n_ue):

            ue_comp_cap = np.squeeze(self.comp_cap_ue[ue_index])
            ue_arrive_task_size = np.squeeze(self.arrive_task_size[self.time_count, ue_index])
            ue_arrive_task_dens = np.squeeze(self.arrive_task_dens[self.time_count, ue_index])
        
            # Handle task arrival
            if ue_arrive_task_size > 0:
                self.UE_TASK[ue_index] += 1 # Increment task ID for this UE
                
                tmp_dict = {
                    'DIV' : 0 , 
                    'UE_ID': ue_index,
                    'TASK_ID': self.UE_TASK[ue_index],
                    'SIZE': ue_arrive_task_size,
                    'DENS': ue_arrive_task_dens,
                    'TIME': self.time_count,
                    'EDGE': ue_action_offload[ue_index],
                }

                if ue_action_local[ue_index] == 1:
                    self.ue_computation_queue[ue_index].put(tmp_dict)
                else:
                    self.ue_transmission_queue[ue_index].put(tmp_dict)


            # LOCAL COMPUTATION PROCESSING
            for cycle in range(self.n_cycle):   
                # TASK ON PROCESS
                if math.isnan(self.local_process_task[ue_index]['REMAIN']) \
                        and (not self.ue_computation_queue[ue_index].empty()):
                    while not self.ue_computation_queue[ue_index].empty():
                        get_task = self.ue_computation_queue[ue_index].get()
                        
                        if get_task['SIZE'] != 0:
                            if self.time_count - get_task['TIME'] + 1 <= self.max_delay:
                                self.local_process_task[ue_index]['UE_ID']   = get_task['UE_ID']
                                self.local_process_task[ue_index]['TASK_ID'] = get_task['TASK_ID']
                                self.local_process_task[ue_index]['SIZE']    = get_task['SIZE']
                                self.local_process_task[ue_index]['DENS']    = get_task['DENS']
                                self.local_process_task[ue_index]['TIME']    = get_task['TIME']
                                self.local_process_task[ue_index]['REMAIN']  = self.local_process_task[ue_index]['SIZE']
                                self.local_process_task[ue_index]['DIV']     = get_task['DIV']
                                break
                            else:
                                # Task is already delayed, drop it
                                self.process_delay[get_task['TIME'], ue_index] = self.max_delay
                                self.unfinish_task[get_task['TIME'], ue_index] = 1
                                self.drop_ue_count += 1
                                
                # PROCESS
                bits_to_process = 0.0
                if self.local_process_task[ue_index]['REMAIN'] > 0:
                    
                    # Calculate how many bits can be processed
                    bits_to_process = ue_comp_cap / self.local_process_task[ue_index]['DENS']
                    
                    if self.local_process_task[ue_index]['REMAIN'] >= bits_to_process:
                        processed_bits = bits_to_process
                        self.local_process_task[ue_index]['REMAIN'] -= bits_to_process
                    else:
                        processed_bits = self.local_process_task[ue_index]['REMAIN']
                        self.local_process_task[ue_index]['REMAIN'] = 0
                    
                    # Store energy and processed bits
                    self.ue_bit_processed[self.local_process_task[ue_index]['TIME'], ue_index] += processed_bits
                    # Energy = Power * Time. Time = Cycles / Freq. Cycles = bits * dens.
                    # Simplified: Energy = bits * dens * k (k is Wh/cycle)
                    # Using original logic: Energy = Power * Duration
                    self.ue_comp_energy[self.local_process_task[ue_index]['TIME'], ue_index] += self.ue_p_comp * self.duration * (processed_bits / bits_to_process)


                    # if no remain, compute processing delay
                    if self.local_process_task[ue_index]['REMAIN'] <= 0: 
                        self.process_delay[self.local_process_task[ue_index]['TIME'], ue_index] \
                            = self.time_count - self.local_process_task[ue_index]['TIME'] + 1
                        self.local_process_task[ue_index]['REMAIN'] = np.nan

                    elif self.time_count - self.local_process_task[ue_index]['TIME'] + 1 == self.max_delay:
                        self.local_process_task[ue_index]['REMAIN'] = np.nan
                        self.process_delay[self.local_process_task[ue_index]['TIME'], ue_index] = self.max_delay
                        self.unfinish_task[self.local_process_task[ue_index]['TIME'], ue_index] = 1
                        self.drop_ue_count += 1

                # OTHER INFO self.t_ue_comp[ue_index]
                # update self.t_ue_comp[ue_index] only when ue_bitrate != 0
                if ue_arrive_task_size != 0 and ue_action_local[ue_index] == 1:
                    tmp_tilde_t_ue_comp = np.max([self.t_ue_comp[ue_index] + 1, self.time_count])
                    self.t_ue_comp[ue_index] = np.min([tmp_tilde_t_ue_comp
                                                       + math.ceil(ue_arrive_task_size
                                                       / (ue_comp_cap / ue_arrive_task_dens)) - 1,
                                                       self.time_count + self.max_delay - 1])

        # edge QUEUE UPDATE =========================
        for ue_index in range(self.n_ue):
            for edge_index in range(self.n_edge):
                
                # Calculate available capacity for this UE on this Edge
                if self.edge_ue_m[edge_index] == 0:
                    edge_cap_per_ue = 0 # No one is using it, no processing
                else:
                    edge_cap_per_ue = self.comp_cap_edge[edge_index] / self.n_cycle / self.edge_ue_m[edge_index]

                for cycle in range(self.n_cycle): 
                    # TASK ON PROCESS
                    if math.isnan(self.edge_process_task[ue_index][edge_index]['REMAIN']) \
                            and (not self.edge_computation_queue[ue_index][edge_index].empty()):
                        while not self.edge_computation_queue[ue_index][edge_index].empty():
                            get_task = self.edge_computation_queue[ue_index][edge_index].get()
                            
                            if self.time_count - get_task['TIME'] + 1 <= self.max_delay:
                                self.edge_process_task[ue_index][edge_index]['UE_ID']   = get_task['UE_ID']
                                self.edge_process_task[ue_index][edge_index]['TASK_ID'] = get_task['TASK_ID']
                                self.edge_process_task[ue_index][edge_index]['SIZE']    = get_task['SIZE']
                                self.edge_process_task[ue_index][edge_index]['DENS']    = get_task['DENS']
                                self.edge_process_task[ue_index][edge_index]['TIME']    = get_task['TIME']
                                self.edge_process_task[ue_index][edge_index]['REMAIN']  = self.edge_process_task[ue_index][edge_index]['SIZE']
                                self.edge_process_task[ue_index][edge_index]['DIV']     = get_task['DIV']
                                break
                            else:
                                # Task is already delayed, drop it
                                self.process_delay[get_task['TIME'], ue_index] = self.max_delay
                                self.unfinish_task[get_task['TIME'], ue_index] = 1
                                self.drop_edge_count += 1
                                
                    # PROCESS
                    self.edge_drop[ue_index, edge_index] = 0

                    if self.edge_process_task[ue_index][edge_index]['REMAIN'] > 0:
                        
                        bits_to_process = edge_cap_per_ue / self.edge_process_task[ue_index][edge_index]['DENS']
                        
                        if self.edge_process_task[ue_index][edge_index]['REMAIN'] >= bits_to_process:
                            processed_bits = bits_to_process
                            self.edge_process_task[ue_index][edge_index]['REMAIN'] -= bits_to_process
                        else:
                            processed_bits = self.edge_process_task[ue_index][edge_index]['REMAIN']
                            self.edge_process_task[ue_index][edge_index]['REMAIN'] = 0
                            
                        # Store energy and processed bits
                        task_time = self.edge_process_task[ue_index][edge_index]['TIME']
                        self.edge_bit_processed[task_time, ue_index, edge_index] += processed_bits
                        # Edge energy
                        self.edge_comp_energy[task_time, ue_index, edge_index] += self.edge_p_comp * self.duration * (processed_bits / bits_to_process if bits_to_process > 0 else 0)
                        # UE idle energy (while waiting for edge)
                        self.ue_idle_energy[task_time, ue_index, edge_index] += self.ue_p_idle * self.duration * (processed_bits / bits_to_process if bits_to_process > 0 else 0)
                        
                        if self.edge_process_task[ue_index][edge_index]['REMAIN'] <= 0:
                            self.process_delay[task_time, ue_index] \
                                = self.time_count - task_time + 1
                            self.edge_process_task[ue_index][edge_index]['REMAIN'] = np.nan

                        elif self.time_count - task_time + 1 == self.max_delay:
                            self.edge_drop[ue_index, edge_index] = self.edge_process_task[ue_index][edge_index]['REMAIN']
                            self.process_delay[task_time, ue_index] = self.max_delay
                            self.unfinish_task[task_time, ue_index] = 1
                            self.edge_process_task[ue_index][edge_index]['REMAIN'] = np.nan
                            self.drop_edge_count += 1

                    # OTHER INFO
                    if edge_cap_per_ue > 0:
                        self.b_edge_comp[ue_index, edge_index] \
                            = np.max([self.b_edge_comp[ue_index, edge_index]
                                      - bits_to_process
                                      - self.edge_drop[ue_index, edge_index], 0])

        # TRANSMISSION QUEUE UPDATE ===================
        for ue_index in range(self.n_ue):
            
            ue_arrive_task_size = np.squeeze(self.arrive_task_size[self.time_count, ue_index])
            ue_arrive_task_dens = np.squeeze(self.arrive_task_dens[self.time_count, ue_index])
            
            for cycle in range(self.n_cycle):

                # TASK ON PROCESS
                if math.isnan(self.local_transmit_task[ue_index]['REMAIN']) \
                        and (not self.ue_transmission_queue[ue_index].empty()):
                    while not self.ue_transmission_queue[ue_index].empty():
                        get_task = self.ue_transmission_queue[ue_index].get()
                        
                        if get_task['SIZE'] != 0:
                            if self.time_count - get_task['TIME'] + 1 <= self.max_delay:
                                self.local_transmit_task[ue_index]['UE_ID'] = get_task['UE_ID']
                                self.local_transmit_task[ue_index]['TASK_ID'] = get_task['TASK_ID']
                                self.local_transmit_task[ue_index]['SIZE'] = get_task['SIZE']
                                self.local_transmit_task[ue_index]['DENS'] = get_task['DENS']
                                self.local_transmit_task[ue_index]['TIME'] = get_task['TIME']
                                self.local_transmit_task[ue_index]['EDGE'] = int(get_task['EDGE'])
                                self.local_transmit_task[ue_index]['REMAIN'] = self.local_transmit_task[ue_index]['SIZE']
                                self.local_transmit_task[ue_index]['DIV'] = get_task['DIV']
                                break
                            else:
                                # Task is already delayed, drop it
                                self.process_delay[get_task['TIME'], ue_index] = self.max_delay
                                self.unfinish_task[get_task['TIME'], ue_index] = 1
                                self.drop_trans_count += 1
                                
                # PROCESS
                if self.local_transmit_task[ue_index]['REMAIN'] > 0:
                    
                    edge_index_trans = self.local_transmit_task[ue_index]['EDGE']
                    ue_tran_cap_edge = np.squeeze(self.tran_cap_ue[ue_index, edge_index_trans]) / self.n_cycle
                    
                    if self.local_transmit_task[ue_index]['REMAIN'] >= ue_tran_cap_edge:
                        transmitted_bits = ue_tran_cap_edge
                        self.local_transmit_task[ue_index]['REMAIN'] -= ue_tran_cap_edge
                    else:
                        transmitted_bits = self.local_transmit_task[ue_index]['REMAIN']
                        self.local_transmit_task[ue_index]['REMAIN'] = 0
                    
                    # Store energy and transmitted bits
                    task_time = self.local_transmit_task[ue_index]['TIME']
                    self.ue_bit_transmitted[task_time, ue_index] += transmitted_bits
                    self.ue_tran_energy[task_time, ue_index] += self.ue_p_tran * self.duration * (transmitted_bits / ue_tran_cap_edge if ue_tran_cap_edge > 0 else 0)


                    # UPDATE edge QUEUE
                    if self.local_transmit_task[ue_index]['REMAIN'] <= 0:
                        tmp_dict = {'UE_ID': self.local_transmit_task[ue_index]['UE_ID'],
                                    'TASK_ID': self.local_transmit_task[ue_index]['TASK_ID'],
                                    'SIZE' : self.local_transmit_task[ue_index]['SIZE'],
                                    'DENS' : self.local_transmit_task[ue_index]['DENS'],
                                    'TIME' : self.local_transmit_task[ue_index]['TIME'],
                                    'EDGE' : self.local_transmit_task[ue_index]['EDGE'],
                                    'DIV'  : self.local_transmit_task[ue_index]['DIV']}

                        
                        self.edge_computation_queue[ue_index][self.local_transmit_task[ue_index]['EDGE']].put(tmp_dict)
                        self.task_count_edge += 1

                        edge_index = self.local_transmit_task[ue_index]['EDGE']
                        self.b_edge_comp[ue_index, edge_index] += self.local_transmit_task[ue_index]['SIZE']
                        
                        self.process_delay_trans[task_time, ue_index] = self.time_count - task_time + 1
                        self.local_transmit_task[ue_index]['REMAIN'] = np.nan


                    elif self.time_count - task_time + 1 == self.max_delay:
                        self.local_transmit_task[ue_index]['REMAIN'] = np.nan
                        self.process_delay[task_time, ue_index] = self.max_delay
                        self.unfinish_task[task_time, ue_index] = 1
                        self.drop_trans_count += 1

                # OTHER INFO
                if ue_arrive_task_size != 0 and ue_action_local[ue_index] == 0:
                    edge_index_trans = ue_action_offload[ue_index]
                    ue_tran_cap_edge = np.squeeze(self.tran_cap_ue[ue_index, edge_index_trans])
                    
                    tmp_tilde_t_ue_tran = np.max([self.t_ue_tran[ue_index] + 1, self.time_count])
                    self.t_ue_tran[ue_index] = np.min([tmp_tilde_t_ue_tran
                                                       + math.ceil(ue_arrive_task_size
                                                       / ue_tran_cap_edge) - 1,
                                                       self.time_count + self.max_delay - 1])
                


        # COMPUTE CONGESTION (FOR NEXT TIME SLOT)
        self.edge_ue_m_observe = self.edge_ue_m
        self.edge_ue_m = np.zeros(self.n_edge)
        for edge_index in range(self.n_edge):
            for ue_index in range(self.n_ue):
                if (not self.edge_computation_queue[ue_index][edge_index].empty()) \
                        or self.edge_process_task[ue_index][edge_index]['REMAIN'] > 0:
                    self.edge_ue_m[edge_index] += 1


        # TIME UPDATE
        self.time_count = self.time_count + 1
        done = False
        if self.time_count >= self.n_time:
            done = True
            
            # set all the tasks' processing delay and unfinished indicator
            for time_index in range(self.n_time):
                for ue_index in range(self.n_ue):
                    if self.process_delay[time_index, ue_index] == 0 and self.arrive_task_size[time_index, ue_index] != 0:
                        self.process_delay[time_index, ue_index] = (self.time_count - 1) - time_index + 1
                        self.unfinish_task[time_index, ue_index] = 1


        # OBSERVATION
        UEs_OBS_ = np.zeros([self.n_ue, self.n_features])
        UEs_lstm_state_ = np.zeros([self.n_ue, self.n_lstm_state])
        if not done:
            for ue_index in range(self.n_ue):
                # observation is zero if there is no task arrival
                if self.arrive_task_size[self.time_count, ue_index] != 0:
                    # state [A, B^{comp}, B^{tran}, [B^{edge}]]
                    UEs_OBS_[ue_index, :] = np.hstack([
                        self.arrive_task_size[self.time_count, ue_index],
                        self.t_ue_comp[ue_index] - self.time_count + 1,
                        self.t_ue_tran[ue_index] - self.time_count + 1,
                        self.b_edge_comp[ue_index, :],
                        self.ue_energy_state[ue_index]])

                UEs_lstm_state_[ue_index, :] = np.hstack(self.edge_ue_m)

        return UEs_OBS_, UEs_lstm_state_, done