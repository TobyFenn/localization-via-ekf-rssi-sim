import numpy as np

# Time step for each simulation update
DT = 1.0

# Number of simulation steps (frames in the animation)
NUM_STEPS = 600

# Noise standard deviations
RSSI_NOISE_STD = 2.0    # dB noise on RSSI measurements
PROCESS_NOISE_STD = 0.1 # process noise on acceleration

# Path-loss model parameters
# RSSI = A - 10*n*log10(distance)
# Example: A = -30 dB (reference), n ~ 2 to 3 for free space
PL_REF = -30.0
PL_EXP = 2.5

# Initial vehicle state: [x, y, vx, vy]
INIT_STATE = np.array([50.0, 50.0, -1.0, 0.5])

# FAKE INIT STATE
FAKE_INIT_STATE = np.array([50.0, 50.0, 1.0, 0.5])

# Radio station positions (x, y)
STATIONS = np.array([
    [  0.0,   0.0],
    [100.0,   0.0],
    [  0.0, 100.0],
    [100.0, 100.0]
])