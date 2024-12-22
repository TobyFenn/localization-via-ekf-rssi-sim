# main.py
"""
Entry point to run the simulation and visualize results.
Usage:
  python main.py       # without EKF
  python main.py --ekf # with EKF
"""

import sys
import numpy as np
from simulator import simulate_vehicle_motion, generate_rssi_measurements
from ekf import EKFTracker
from visualization import create_animation
from config import INIT_STATE, NUM_STEPS, FAKE_INIT_STATE

def run_simulation(use_ekf=False):
    # 1. Simulate the true vehicle motion
    true_states = simulate_vehicle_motion()
    
    # 2. Generate noisy RSSI measurements
    measurements = generate_rssi_measurements(true_states)
    
    ekf_estimates = None
    if use_ekf:
        # 3. Initialize EKF
        ekf = EKFTracker(init_state=FAKE_INIT_STATE.copy())
        estimates = []
        
        # 4. Loop through each time step
        for k in range(NUM_STEPS):
            # Predict
            ekf.predict()
            
            # Update with the current RSSI measurements
            ekf.update(measurements[k])
            
            estimates.append(ekf.state.copy())
        
        ekf_estimates = np.array(estimates)
    
    # 5. Visualize the results
    create_animation(true_states, measurements, ekf_estimates, use_ekf=use_ekf)

if __name__ == "__main__":
    # Check if user passed --ekf
    if "--ekf" in sys.argv:
        run_simulation(use_ekf=True)
    else:
        run_simulation(use_ekf=False)