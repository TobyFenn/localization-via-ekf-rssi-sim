# simulator.py
"""
Contains classes/functions to simulate the vehicle's true motion
and generate noisy RSSI measurements from radio stations.
"""

import numpy as np
from config import DT, NUM_STEPS, RSSI_NOISE_STD, PROCESS_NOISE_STD
from config import PL_REF, PL_EXP, INIT_STATE, STATIONS

def simulate_vehicle_motion():
    """
    Simulates the 'true' vehicle (robot) motion over time using a simple
    constant-velocity model plus random accelerations.
    Returns an array of shape (NUM_STEPS, 4): [x, y, vx, vy] at each step.
    """
    state = INIT_STATE.copy()
    states = []
    
    # Screen boundaries
    X_MIN, X_MAX = -10, 110
    Y_MIN, Y_MAX = -10, 110
    
    for _ in range(NUM_STEPS):
        states.append(state.copy())
        
        # Random acceleration in x & y
        ax = np.random.randn() * PROCESS_NOISE_STD
        ay = np.random.randn() * PROCESS_NOISE_STD
        
        # Update position
        new_x = state[0] + state[2] * DT
        new_y = state[1] + state[3] * DT
        
        # Bounce off boundaries by reversing velocity
        if new_x < X_MIN or new_x > X_MAX:
            state[2] *= -0.8  # Dampen velocity on bounce
            new_x = np.clip(new_x, X_MIN, X_MAX)
        
        if new_y < Y_MIN or new_y > Y_MAX:
            state[3] *= -0.8  # Dampen velocity on bounce
            new_y = np.clip(new_y, Y_MIN, Y_MAX)
            
        state[0] = new_x
        state[1] = new_y
        
        # Update velocity
        state[2] += ax * DT
        state[3] += ay * DT
        
        # Limit maximum velocity
        speed = np.sqrt(state[2]**2 + state[3]**2)
        if speed > 5.0:
            state[2:4] *= 5.0/speed
    
    return np.array(states)

def distance_to_rssi(distance):
    """
    Given a distance in meters, returns the RSSI in dB using the
    path-loss model: RSSI = PL_REF - 10*PL_EXP*log10(distance)
    """
    # Avoid log(0) by adding a small offset
    return PL_REF - 10.0 * PL_EXP * np.log10(distance + 1e-3)

def generate_rssi_measurements(true_states):
    """
    Generates noisy RSSI measurements from each station at each time step.
    Returns an array of shape (NUM_STEPS, num_stations).
    """
    measurements = []
    
    for state in true_states:
        x, y, vx, vy = state
        step_measurements = []
        for (sx, sy) in STATIONS:
            dist = np.sqrt((x - sx)**2 + (y - sy)**2)
            ideal_rssi = distance_to_rssi(dist)
            noisy_rssi = ideal_rssi + np.random.randn() * RSSI_NOISE_STD
            step_measurements.append(noisy_rssi)
        measurements.append(step_measurements)
    
    return np.array(measurements)