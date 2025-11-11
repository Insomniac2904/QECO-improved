class Config(object):
    
    # System setup
    N_UE             = 20                   # Number of Mobile Devices
    N_EDGE           = 2                    # Number of Edge Servers
    UE_COMP_CAP      = 2.6                  # Mobile Device Computation Capacity
    UE_TRAN_CAP      = 14                   # Mobile Device Transmission Capacity
    EDGE_COMP_CAP    = 42                   # Edge Servers Computation Capacity

    # Energy consumption settings
    UE_ENERGY_STATE  = [0.25, 0.50, 0.75]   # Ultra-power-saving mode, Power-saving mode, Performance mode
    UE_COMP_ENERGY   = 2                    # Computation Power of Mobile Device
    UE_TRAN_ENERGY   = 2.3                  # Transmission Power of Mobile Device
    UE_IDLE_ENERGY   = 0.1                  # Standby power of Mobile Device
    EDGE_COMP_ENERGY = 5                    # Computation Power of Edge Server

    # Task Requrement
    TASK_COMP_DENS   = [0.197, 0.297, 0.397]      # Task Computation Density
    
    #TASK_COMP_DENS   = 0.297


    TASK_MIN_SIZE    = 1
    TASK_MAX_SIZE    = 7
    N_COMPONENT      = 1                    # Number of Task Partitions
    MAX_DELAY        = 10


    # Simulation scenario
    N_EPISODE        = 801                # Number of Episodes
    N_TIME_SLOT      = 100                  # Number of Time Slots
    DURATION         = 0.1                  # Time Slot Duration
    TASK_ARRIVE_PROB = 0.3                  # Task Generation Probability
    N_TIME = N_TIME_SLOT + MAX_DELAY


    # Algorithm settings
    LEARNING_RATE    = 0.001
    REWARD_DECAY     = 0.9
    E_GREEDY         = 0.99
    N_NETWORK_UPDATE = 200                  # Networks Parameter Replace
    MEMORY_SIZE      = 50000                  # Replay Buffer Memory Size (5000 -> on edge its stored so no problem)
    # add/replace these fields in Config
    # MEMORY_SIZE      = 50_000     # larger than 5k for more stability but still modest
    BATCH_SIZE       = 64
    TARGET_TAU       = 5e-4       # soft target update
    N_STEP           = 3
    GRAD_CLIP_NORM   = 10.0
    BURN_IN_STEPS    = 2000       # fill-replay before learning
    USE_PRIORITIZED_REPLAY = False  # set True later if you implement PER
    LEARNING_RATE    = 1e-4       # modest LR for LSTM encoder
    SEQUENCE_LENGTH  = 16         # keep as original; you can change to 32 later
    LSTM_UNITS       = 64         # hidden units for LSTM encoder
    ATTN_UNITS       = 64         # attention internal size
    BATCH_LEARN_FREQ = 10         # in main loop, you already call `.learn()` every 10 steps
# ---------------------------------------------------------------------------------------------------------------------- #
    # SEQUENCE_LENGTH  = 16              # Length of the history sequence for the transformer
    PATCH_LENGTH     = 8               # The number of time steps in each patch
    D_MODEL          = 32              # The embedding dimension for the transformer
    N_HEADS          = 4               # The number of attention heads
    N_ENCODER_LAYERS = 2               # The number of transformer encoder layers
    