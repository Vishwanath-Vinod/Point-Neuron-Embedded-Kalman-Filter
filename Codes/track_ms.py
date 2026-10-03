import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.random import default_rng
from Point_Neuron_Learning.point_neuron import PointNeuron
from Point_Neuron_Learning.helper import complex_converter
from pnekf import PNEKF
from tqdm import tqdm

# For 3D Point Neuron Learning stick to Automatic Gradient Descent using autograd functionality of Pytorch
freeze_bias = True

# Parameters
Frequency = 1200
c = 343
k = 2 * np.pi * Frequency / c
TargetR = 0.50
IterN = 100
StepW = 1e-2 #0.005
StepC = [1e-3,1e-3,1e-3] #[0.005, 0.005, 0.005] #0.02
Lambda = 0.01 #0.01
L = (3.2,3.6,2.2) # room_dimensions
eps =1e-8

# Create Results Folder
mode = 'Auto'
pn_dir = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/MICS=95_movingpns/f=1200/PNEKF(B)/'
results_dir = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/MICS=95_movingpns/f=1200/'
data_dir    = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/MICS=95_movingpns/f=1200/'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(pn_dir, exist_ok=True)
# Set random seed for reproducibility
rng = default_rng(1)


# Load microphone coordinates and measurements of sound field
mic_coordinates = loadmat(data_dir +'MIC_DATA_ms (1).mat')
MicCoord = mic_coordinates['MicCoord']
print(MicCoord.shape)

mic_data = loadmat(data_dir +'Measurements (1).mat')
Z = mic_data['MicData_time_noisy']

# Load initial point neuron coordinates
weights_file = pn_dir + 'pn_weights.csv'
scales_file  = pn_dir + 'pn_scale.csv'
bias_file    = pn_dir + 'pn_biases.csv'

# Use the custom converter for each column
PnBias = np.loadtxt(bias_file, delimiter=',')
PnWeight = np.genfromtxt(weights_file,delimiter=',',dtype=complex,converters={0: complex_converter})
PnScale = np.genfromtxt(scales_file,delimiter=',',dtype=complex,converters={0: complex_converter})


# Collapse first 10 PNs into one
n = 10
top_weights = PnWeight[:n]                # shape: (10,)
new_coord =  [1.15,1.85,0.65]#top_coords[indx]
print(new_coord)
# But keep actual complex average for weight (not abs)
new_weight = np.sum(top_weights)
PnBias = np.vstack([new_coord, PnBias[n:]])        
PnWeight = np.concatenate([[new_weight], PnWeight[n:]]) 

# Number of Point Neurons, Timesteps and Microphones
PnN = PnBias.shape[0]
num_steps = Z.shape[1]
MicN = MicCoord.shape[0]
velocity = [1e-3,1e-3,1e-3]
X_t = np.concatenate([PnBias.flatten(), velocity])

acc = loadmat(data_dir +'Process_noise.mat')
acceleration_std = np.sqrt(acc['accel_std_estimated'][0])
R = torch.eye(2 * MicN)*1e-2
state_dim = 3*PnN + 3

error_covariance = np.eye(state_dim)
error_covariance[:3*PnN, :3*PnN] *= 1e-6  # Position variance
error_covariance[-3:, -3:] *= 1e-5     # Velocity variance
ekf = PNEKF(X_t, MicCoord,  error_covariance,acceleration_std, R,PnWeight, Frequency=1200,source_pns_num=1, wall_pns_num=68, dt=1)
estimated_positions = []
period_update_velocity = 1
for t in tqdm(range(num_steps)):
    Xhat_t, Fhat_t = ekf.predict()
    Z_t = Z[:,t]
    NormalFactor = np.linalg.norm(Z_t)
    Z_t = Z_t/ NormalFactor
    coords = X_t[:-3].reshape(ekf.total_pns, 3)
    model = PointNeuron(k,Z_t,MicCoord,coords,ekf.pn_weight,StepW, StepC, Lambda, IterN)
    if freeze_bias:
        _, ekf.pn_weight,_,Loss = model.train_autograd()
    else:
        _, ekf.pn_weight,_,Loss = model.pseudo_inverse()
    X_t, F_t = ekf.update_iekf(Xhat_t, Fhat_t, Z_t)
    
    ekf.X_t[-3:] += (X_t[:3] - Xhat_t[:3]) / ekf.dt
    print("CHANGE IN VELOCITY",(X_t[:3] - Xhat_t[:3]) / ekf.dt)
        
    # Extract position part of state
    pn_positions_flat = ekf.X_t[:3*PnN]         # shape: (3*PnN,)
    pn_positions = pn_positions_flat.reshape(PnN, 3)  # shape: (PnN, 3)
    source_pos = pn_positions[0]
    print(f"SOURCE POSITION at time {t+1} with Frequency {Frequency}",source_pos)
    estimated_positions.append(source_pos)

estimated_positions = torch.stack(estimated_positions)  
estimated_positions = estimated_positions.cpu().numpy()  # shape: (num_steps, 3)
np.savetxt(results_dir +'estimated_positions (1)_lr=1e-3,steps=100.csv', estimated_positions, delimiter=',')

    