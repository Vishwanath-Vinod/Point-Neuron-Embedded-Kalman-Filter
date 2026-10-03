import torch
import numpy as np
import os
from scipy.io import loadmat
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
n_sources = 3  # <--- number of sources
pn_init_file = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/MICS=95/PN_INITIALIZATION_omp.mat'
data_dir = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/3sources/Circular/f=1200/'
results_dir = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/3sources/Circular/f=1200/KF+OMP/'

os.makedirs(results_dir, exist_ok=True)

# Load microphone coordinates and sound field measurements
mic_coordinates = loadmat(data_dir + 'MIC_DATA_ms.mat')
MicCoord = mic_coordinates['MicCoord']
MicData = mic_coordinates['MicData'].flatten()
mic_data = loadmat(data_dir + 'Measurements.mat')
Z = mic_data['MicData_time_noisy']  # (num_mics, num_timesteps)
num_steps = Z.shape[1]
MicN = MicCoord.shape[0]

# Load process noise
acc = loadmat(data_dir + 'Process_noise.mat')
acceleration_std = np.sqrt(acc['accel_std_estimated'][0])

state_dim = 6
R = torch.eye(3) * 1e-4
error_covariance = np.eye(state_dim) * 1e-6

# Localization init
pn_data = loadmat(pn_init_file)
InCoord = pn_data['biases']
PnN = InCoord.shape[0]
InWeight = np.random.rand(PnN, MicN)
InWeight = InWeight / InWeight.sum(axis=1, keepdims=True)
InWeight = InWeight @ MicData
PnCoord, PnWeight = OMP.initialize(k, MicData, MicCoord, InCoord, InWeight,
                                   IterN=10000, freeze_bias=False)

PnCoord_reverb = PnCoord[-num_reverb:]
PnWeight_reverb = PnWeight[-num_reverb:]
PnCoord_direct = PnCoord[:num_direct]
MicDereverb = OMP.dereverb(PnCoord_reverb, PnWeight_reverb, k, MicCoord, MicData)

# Initial localization with sparsity = n_sources
x, omp_error, c = OMP.localize(MicDereverb, PnCoord_direct, MicCoord, k)
init_positions = [[1.15,1.85,1.05],[1.65,1.35,1.15],[1.55,1.85,0.75]] #PnCoord_direct[c]   # shape: (n_sources, 3)
#Circular: [[1.15,1.85,1.05],[1.65,1.35,1.15],[1.55,1.85,0.75]]
#Mixed: [[1.15,1.85,1.05],[1.65,1.35,1.25],[1.15,1.55,0.75]]
print(f'Initial Localization Results: {init_positions}')

# Create 1 Kalman filter per source
kfs = []
for i in range(n_sources):
    X_t = np.concatenate([init_positions[i], velocity])
    kfs.append(KalmanFilter(X_t, error_covariance.copy(), acceleration_std, R))

# Storage for estimated trajectories
estimated_positions = [[] for _ in range(n_sources)]

period = 1
for t in tqdm(range(num_steps)):
    Z_t = Z[:, t]

    # Re-run OMP localization
    PnCoord, PnWeight = OMP.initialize(k, Z_t, MicCoord, PnCoord, PnWeight,
                                       IterN=500, freeze_bias=False)
    PnCoord_reverb = PnCoord[-num_reverb:]
    PnWeight_reverb = PnWeight[-num_reverb:]
    PnCoord_direct = PnCoord[:num_direct]
    MicDereverb = OMP.dereverb(PnCoord_reverb, PnWeight_reverb, k, MicCoord, Z_t)
    x, omp_error, c = OMP.localize(MicDereverb, PnCoord_direct, MicCoord, k)
    est_positions = PnCoord_direct[c]   # (n_sources, 3)
    # Update each Kalman filter with corresponding measurement
    for i in range(n_sources):
        Xhat_t, Fhat_t = kfs[i].predict()
        X_t, F_t = kfs[i].update(Xhat_t, Fhat_t, est_positions[i])
        kfs[i].X_t = X_t
        kfs[i].F_t = F_t
        estimated_positions[i].append(X_t[:3].cpu().numpy())

# Save trajectories
for i in range(n_sources):
    est_traj = np.vstack(estimated_positions[i])  # (num_steps, 3)
    np.savetxt(os.path.join(results_dir, f'estimated_positions_source{i+1}.csv'),
               est_traj, delimiter=',')
