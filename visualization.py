# visualization.py
"""
Creates an animated plot of the true vehicle position, noisy measurements,
and (optionally) EKF estimates over time.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from config import STATIONS, PL_EXP
from simulator import distance_to_rssi

def create_animation(true_states, measurements, ekf_estimates=None, use_ekf=False):
    """
    Animate the simulation results.
    - true_states: (NUM_STEPS, 4) array of [x, y, vx, vy]
    - measurements: (NUM_STEPS, num_stations) array of RSSI
    - ekf_estimates: (NUM_STEPS, 4) array of EKF state estimates (optional)
    - use_ekf: bool indicating if EKF was enabled
    """
    fig, ax = plt.subplots(figsize=(7,7))
    ax.set_title("EKF Radio Position Demo")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_xlim(-20, 120)
    ax.set_ylim(-20, 120)
    ax.grid(True)
    
    # Plot the station locations
    ax.scatter(STATIONS[:,0], STATIONS[:,1], marker='^', color='red', s=100, label='Stations')
    
    # Lines to update
    true_line, = ax.plot([], [], 'b-', label='True Path')
    if use_ekf:
        ekf_line,  = ax.plot([], [], 'g--', label='EKF Estimate')
    else:
        ekf_line = None
    
    # Scatter for current position
    true_pos_scatter = ax.scatter([], [], color='blue', s=50, label='True Pos')
    ekf_pos_scatter  = ax.scatter([], [], color='green', s=50, label='EKF Pos') if use_ekf else None
    
    # Keep track of path points
    true_path_x = []
    true_path_y = []
    ekf_path_x  = []
    ekf_path_y  = []
    
    # For each station, show a circle or ring to indicate approximate range
    # We'll update these circles each frame based on measured RSSI
    station_circles = []
    for _ in STATIONS:
        circle = plt.Circle((0,0), 0, fill=False, color='gray', linestyle=':', alpha=0.5)
        ax.add_patch(circle)
        station_circles.append(circle)
    
    ax.legend(loc='upper left')
    
    def init():
        true_line.set_data([], [])
        if ekf_line:
            ekf_line.set_data([], [])
        return [true_line, ekf_line, true_pos_scatter, ekf_pos_scatter] if ekf_line else [true_line, true_pos_scatter]

    def update(frame):
        # True position
        x_true, y_true, vx_true, vy_true = true_states[frame]
        true_path_x.append(x_true)
        true_path_y.append(y_true)
        true_line.set_data(true_path_x, true_path_y)
        true_pos_scatter.set_offsets([x_true, y_true])
        
        # If EKF is used
        if use_ekf and ekf_estimates is not None:
            x_ekf, y_ekf, vx_ekf, vy_ekf = ekf_estimates[frame]
            ekf_path_x.append(x_ekf)
            ekf_path_y.append(y_ekf)
            ekf_line.set_data(ekf_path_x, ekf_path_y)
            ekf_pos_scatter.set_offsets([x_ekf, y_ekf])
        
        # Update circle radii based on measured RSSI (roughly convert RSSI->distance)
        for i, circle in enumerate(station_circles):
            rssi = measurements[frame, i]
            # Invert the path-loss to get approximate distance
            # rssi = PL_REF - 10*PL_EXP*log10(distance)
            # => distance = 10^((PL_REF - rssi)/(10*PL_EXP))
            dist_approx = 10**((rssi - distance_to_rssi(1e-3)) / (-10.0*PL_EXP))  # or a simpler approach
            sx, sy = STATIONS[i]
            circle.center = (sx, sy)
            circle.radius = dist_approx
        
        return []
    
    ani = animation.FuncAnimation(
        fig, update, frames=len(true_states),
        init_func=init, blit=False, interval=20, repeat=False
    )
    
    plt.show()