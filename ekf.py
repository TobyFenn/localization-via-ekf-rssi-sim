# ekf.py
"""
Implements a basic Extended Kalman Filter for the RSSI-based position tracking.
"""

import numpy as np
from config import DT, PL_REF, PL_EXP, STATIONS, RSSI_NOISE_STD

class EKFTracker:
    def __init__(self, init_state, init_cov=None):
        """
        init_state: [x, y, vx, vy]
        init_cov: 4x4 covariance matrix (optional)
        """
        self.state = init_state
        self.cov = init_cov if init_cov is not None else np.eye(4) * 100.0
        
        # Process noise covariance (tuned for acceleration variance)
        self.Q = np.diag([0.1, 0.1, 0.1, 0.1])  # <-- TUNE THIS
        num_stations = STATIONS.shape[0]

        # Measurement noise covariance
        # Each station measurement has variance (RSSI_NOISE_STD^2)
        self.R = np.eye(num_stations) * (RSSI_NOISE_STD**2)  # <-- TUNE THIS
        
    def predict(self):
        """
        EKF predict step based on constant velocity model.
        """
        # State transition matrix
        F = np.array([
            [1, 0, DT, 0 ],
            [0, 1, 0,  DT],
            [0, 0, 1,  0 ],
            [0, 0, 0,  1 ]
        ])
        
        # Predict state
        self.state = F @ self.state
        
        # Predict covariance
        self.cov = F @ self.cov @ F.T + self.Q

    def rssi_measurement_function(self, state):
        """
        Non-linear measurement function h(x) -> RSSI from each station.
        """
        x, y, vx, vy = state
        rssi_vals = []
        for (sx, sy) in STATIONS:
            dist = np.sqrt((x - sx)**2 + (y - sy)**2) + 1e-3
            # RSSI = PL_REF - 10*PL_EXP*log10(dist)
            rssi = PL_REF - 10.0 * PL_EXP * np.log10(dist)
            rssi_vals.append(rssi)
        return np.array(rssi_vals)
    
    def measurement_jacobian(self, state):
        """
        Jacobian of rssi_measurement_function wrt [x, y, vx, vy].
        """
        x, y, vx, vy = state
        H = []
        for (sx, sy) in STATIONS:
            dist = np.sqrt((x - sx)**2 + (y - sy)**2) + 1e-3
            # dRSSI/dDist = - (10 * PL_EXP) / (dist * ln(10))
            dRSSI_dDist = -(10.0 * PL_EXP) / (dist * np.log(10.0))
            
            dx = (x - sx) / dist
            dy = (y - sy) / dist
            
            dRSSI_dx = dRSSI_dDist * dx
            dRSSI_dy = dRSSI_dDist * dy
            
            # No direct dependence on vx, vy in the measurement
            H.append([dRSSI_dx, dRSSI_dy, 0.0, 0.0])
        return np.array(H)

    def update(self, z):
        """
        EKF update step.
        z: [num_stations] array of measured RSSI values at current time.
        """
        # Predicted measurement
        z_pred = self.rssi_measurement_function(self.state)
        # Residual
        y = z - z_pred
        
        # Jacobian
        H = self.measurement_jacobian(self.state)
        
        # S = H * P * H^T + R
        S = H @ self.cov @ H.T + self.R
        
        # K = P * H^T * inv(S)
        K = self.cov @ H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(4)
        self.cov = (I - K @ H) @ self.cov