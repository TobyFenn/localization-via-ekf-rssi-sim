# EKF Radio Position Demo

This project demonstrates how to estimate a moving vehicle's position using noisy radio signal strength measurements. An Extended Kalman Filter (EKF) refines the estimates.

## Features

- 2D simulation of a robot with a simple constant-velocity motion model  
- Radio stations at fixed known locations  
- Radio signal strength (RSSI) measurements with noise  
- Toggle the EKF on/off to compare raw readings vs. filtered estimates  
- Real-time animation showing:
  - Robot's true path  
  - Noisy position estimates  
  - EKF's refined estimates  
  - Radio stations and circles indicating approximate signal strength  

## Usage

1. **Install Requirements**  
   ```bash
   pip install matplotlib numpy