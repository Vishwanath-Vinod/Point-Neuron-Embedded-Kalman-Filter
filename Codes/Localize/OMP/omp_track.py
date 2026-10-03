import torch
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.random import default_rng
from tqdm import tqdm
from ekf_baseline import KalmanFilter
from omp import OMP

# Parameters
Frequency = 1200
room_dims = np.array([3.2,3.6,2.2])
num_reverb = 600
num_direct = 1000
dereverb = True
c = 343
k = 2 * np.pi * Frequency / c
velocity = [1e-3, 1e-3, 1e-3]
fs = 16000
grid_type = "3D"
n_grid_cells = 20
estimated_positions = []

results_dir = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/MICS=80 (1)/f=1200/OMP/'
data_dir    = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/MICS=80 (1)/f=1200/'
pn_init_file = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/MICS=80 (1)/PN_INITIALIZATION_omp.mat'

os.makedirs(results_dir, exist_ok=True)
rng = default_rng(1)

# Load microphone coordinates and sound field measurements
mic_coordinates = loadmat(data_dir + 'MIC_DATA (1).mat')
MicCoord = mic_coordinates['MicCoord']
MicData = mic_coordinates['MicData'].flatten()
mic_data = loadmat(data_dir + 'Measurements (1).mat')
Z = mic_data['MicData_time_noisy']  # shape: (num_mics, num_timesteps)
num_steps = Z.shape[1]
MicN = MicCoord.shape[0]


# Load process noise
acc = loadmat(data_dir + 'Process_noise (1).mat')
acceleration_std = np.sqrt(acc['accel_std_estimated'][0])

state_dim = 6
R = torch.eye(3) * 1e-4  # Only 3x3 since we measure position
error_covariance = np.eye(state_dim) * 1e-6


# Localization
pn_data = loadmat(pn_init_file)
InCoord = pn_data['biases']
PnN = InCoord.shape[0]
InWeight = np.random.rand(PnN, MicN)
InWeight = InWeight / InWeight.sum(axis=1, keepdims=True) 
InWeight = InWeight @ MicData  
PnCoord,PnWeight = OMP.initialize(k, MicData, MicCoord, InCoord, InWeight,IterN=10000,freeze_bias=False)
PnCoord_reverb = PnCoord[-num_reverb:]
PnWeight_reverb = PnWeight[-num_reverb:]
PnCoord_direct = PnCoord[:num_direct]
MicDereverb = OMP.dereverb(PnCoord_reverb,PnWeight_reverb, k, MicCoord, MicData)
x, omp_error, c = OMP.localize(MicDereverb, PnCoord_direct,MicCoord, k)
est_position = PnCoord_direct[c[0]]  
print(f'Localization Result: {est_position} ')

# Initial state
X_t = np.concatenate([est_position, velocity])

period = 1
for t in tqdm(range(num_steps)):
    Z_t = Z[:, t]
    if t%period == 0:
        PnCoord,PnWeight = OMP.initialize(k, Z_t, MicCoord, PnCoord, PnWeight,IterN=500,freeze_bias=False)
        PnCoord_reverb = PnCoord[-num_reverb:]
        PnWeight_reverb = PnWeight[-num_reverb:]
        PnCoord_direct = PnCoord[:num_direct]
        MicDereverb = OMP.dereverb(PnCoord_reverb,PnWeight_reverb, k, MicCoord, Z_t)
        x, omp_error, c = OMP.localize(MicDereverb, PnCoord_direct,MicCoord, k)
        est_position = PnCoord_direct[c[0]]
        print('Measured Estimated Position:',est_position)
        X_t = est_position

    print(f"Time {t+1}: Position = {X_t[:3]}")#, ΔVelocity = {velocity_update.cpu().numpy()}")
    estimated_positions.append(X_t[:3])

# Save estimated positions 
np.savetxt(os.path.join(results_dir, 'estimated_positions (1).csv'), estimated_positions, delimiter=',')
