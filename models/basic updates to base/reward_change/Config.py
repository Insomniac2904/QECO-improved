class Config(object):

    # System setup
    N_UE = 20  # Number of Mobile Devices
    N_EDGE = 2  # Number of Edge Servers
    UE_COMP_CAP = 2.6  # Mobile Device Computation Capacity
    UE_TRAN_CAP = 14  # Mobile Device Transmission Capacity
    EDGE_COMP_CAP = 42  # Edge Servers Computation Capacity

    # Energy consumption settings
    UE_ENERGY_STATE = [0.25, 0.50, 0.75]  # Ultra-power-saving mode, Power-saving mode, Performance mode
    UE_COMP_ENERGY = 2  # Computation Power of Mobile Device
    UE_TRAN_ENERGY = 2.3  # Transmission Power of Mobile Device
    UE_IDLE_ENERGY = 0.1  # Standby power of Mobile Device
    EDGE_COMP_ENERGY = 5  # Computation Power of Edge Server

    # Task Requrement
    TASK_COMP_DENS = [0.197, 0.297, 0.397]  # Task Computation Density
    # TASK_COMP_DENS = 0.297

    TASK_MIN_SIZE = 1
    TASK_MAX_SIZE = 7
    N_COMPONENT = 1  # Number of Task Partitions
    MAX_DELAY = 10

    # Simulation scenario
    N_EPISODE = 1000  # Number of Episodes
    N_TIME_SLOT = 100  # Number of Time Slots
    DURATION = 0.1  # Time Slot Duration
    TASK_ARRIVE_PROB = 0.3  # Task Generation Probability
    N_TIME = N_TIME_SLOT + MAX_DELAY

    # --- Step 1: Hyperparameter Updates ---

    # 1. Learning Rate Schedule
    # We moved from a high, static LR (0.01) to a modern schedule
    # that decays. This is much more stable.
    LEARNING_RATE_START = 5e-4  # Start at a smaller, safer rate
    LEARNING_RATE_END = 1e-6  # Decay to a very small rate
    LEARNING_RATE_DECAY_STEPS = 100000  # Decay over many training steps

    # 2. Batch Size
    # Increased batch size for a more stable gradient.
    BATCH_SIZE = 64  # Was 32

    # --- End of Step 1 Updates ---

    REWARD_DECAY = 0.9
    E_GREEDY = 0.99
    N_NETWORK_UPDATE = 200  # Networks Parameter Replace
    MEMORY_SIZE = 500  # Replay Buffer Memory Size