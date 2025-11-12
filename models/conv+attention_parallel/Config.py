import tensorflow as tf
import numpy as np

class Config(object):
    
    # System setup
    N_UE              = 20              # Number of Mobile Devices
    N_EDGE            = 2               # Number of Edge Servers
    UE_COMP_CAP       = 2.6             # Mobile Device Computation Capacity
    UE_TRAN_CAP       = 14              # Mobile Device Transmission Capacity
    EDGE_COMP_CAP     = 42              # Edge Servers Computation Capacity

    # Energy consumption settings
    UE_ENERGY_STATE   = [0.25, 0.50, 0.75] # Ultra-power-saving, Power-saving, Performance
    UE_COMP_ENERGY    = 2               # Computation Power of Mobile Device
    UE_TRAN_ENERGY    = 2.3             # Transmission Power of Mobile Device
    UE_IDLE_ENERGY    = 0.1             # Standby power of Mobile Device
    EDGE_COMP_ENERGY  = 5               # Computation Power of Edge Server

    # Task Requrement
    TASK_COMP_DENS    = [0.197, 0.297, 0.397] # Task Computation Density
    TASK_MIN_SIZE     = 1
    TASK_MAX_SIZE     = 7
    N_COMPONENT       = 1               # Number of Task Partitions
    MAX_DELAY         = 10

    # Simulation scenario
    N_EPISODE         = 1000            # Number of Episodes
    N_TIME_SLOT       = 100             # Number of Time Slots
    DURATION          = 0.1             # Time Slot Duration
    TASK_ARRIVE_PROB  = 0.3             # Task Generation Probability
    N_TIME = N_TIME_SLOT + MAX_DELAY

    # --- Step 1: Hyperparameter Tuning ---
    LEARNING_RATE_START = 1e-4          # Lowered initial learning rate
    LEARNING_RATE_END = 1e-6            # Final learning rate
    LEARNING_RATE_DECAY_STEPS = 1000000 # Steps to decay LR (approx 1000 episodes)
    BATCH_SIZE = 64                     # Increased batch size
    # --- End Step 1 ---

    REWARD_DECAY      = 0.9
    N_NETWORK_UPDATE  = 200             # Networks Parameter Replace
    MEMORY_SIZE       = 500             # Replay Buffer Memory Size

    # --- Reverted Step 4: Epsilon-Greedy Parameters ---
    E_GREEDY          = 0.99            # Max epsilon value
    E_GREEDY_INCREMENT = 0.00025        # Epsilon increment per learn step (original value)
    # --- End Revert ---

    # --- Step 3: PER Parameters ---
    PER_ALPHA = 0.6                     # (0.0 - 1.0) How much prioritization to use
    PER_BETA_START = 0.4                # Initial importance sampling weight
    PER_BETA_ANNEAL_STEPS = 500000      # Steps to anneal beta to 1.0
    PER_EPSILON = 1e-6                  # Small value to ensure all priorities > 0
    # --- End Step 3 ---