import torch
import numpy as np
import os
from scipy.io import loadmat
from tqdm import tqdm
from mmkf import KalmanFilter, MultipleModelKalmanFilter
from omp import OMP

Frequency = 1200
num_reverb = 600
num_direct = 1000
num_sources = 3   # <---- MULTIPLE SOURCES
dereverb = True
c = 343
k = 2 * np.pi * Frequency / c
room_dims = np.array([3.2, 3.6, 2.2])
pn_init_file = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/MICS=95/PN_INITIALIZATION_omp.mat'
data_dir = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/3sources/Mixed/f=1200/'
results_dir = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/3sources/Mixed/f=1200/MMKF+OMP/'
os.makedirs(results_dir, exist_ok=True)

# Load microphone coordinates and sound field measurements
mic_coordinates = loadmat(data_dir + 'MIC_DATA_ms.mat')
MicCoord = mic_coordinates['MicCoord']   
MicData = mic_coordinates['MicData'].flatten()
mic_data = loadmat(data_dir + 'Measurements.mat')
Z_all = mic_data['MicData_time_noisy']               # shape: (num_mics, num_timesteps)

num_steps = Z_all.shape[1]
MicN = MicCoord.shape[0]

# OMP setup
pn_data = loadmat(pn_init_file)
InCoord = pn_data['biases']
PnN = InCoord.shape[0]
InWeight = np.random.rand(PnN, MicN)
InWeight = InWeight / InWeight.sum(axis=1, keepdims=True) 
InWeight = InWeight @ MicData  

# Initial localization
PnCoord,PnWeight = OMP.initialize(k, MicData, MicCoord, InCoord, InWeight,
                                  IterN=10000, freeze_bias=False)
PnCoord_reverb = PnCoord[-num_reverb:]
PnWeight_reverb = PnWeight[-num_reverb:]
PnCoord_direct = PnCoord[:num_direct]
MicDereverb = OMP.dereverb(PnCoord_reverb,PnWeight_reverb, k, MicCoord, MicData)
x, omp_error, c = OMP.localize(MicDereverb, PnCoord_direct,MicCoord, k)

# Take top `num_sources` estimates
init_positions = [[1.15,1.85,1.05],[1.65,1.35,1.25],[1.15,1.55,0.75]] #[PnCoord_direct[idx] for idx in c[:num_sources]]
#Circular: [[1.15,1.85,1.05],[1.65,1.35,1.15],[1.55,1.85,0.75]]
#Mixed: [[1.15,1.85,1.05],[1.65,1.35,1.25],[1.15,1.55,0.75]]
print("Initial localization results:", init_positions)

# Noise parameters
acc = loadmat(data_dir + 'Process_noise.mat')
process_std = np.sqrt(acc['accel_std_estimated'][0])   # shape (3,)
R = np.eye(3) * 1e-4
dt = 1.0

# Define covariance templates
error_cov_stationary = np.diag([1e-6, 1e-6, 1e-6, 1e3, 1e3, 1e3, 1e3, 1e3, 1e3])
error_cov_linear     = np.diag([1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5, 1e3, 1e3, 1e3])
error_cov_acc        = np.diag([1e-6, 1e-6, 1e-6, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-4])

def make_mmkf(est_position):
    x0_stationary  = np.hstack([est_position, np.zeros(3), np.zeros(3)])
    x0_linear      = np.hstack([est_position, np.ones(3)*1e-3, np.zeros(3)])
    x0_accelerated = np.hstack([est_position, np.ones(3)*1e-3, np.ones(3)*1e-5])
    kf_stationary = KalmanFilter(x0_stationary, error_cov_stationary, process_std, R, dt, 'stationary')
    kf_linear     = KalmanFilter(x0_linear, error_cov_linear, process_std, R, dt, 'linear')
    kf_acc        = KalmanFilter(x0_accelerated, error_cov_acc, process_std, R, dt, 'accelerated')
    return MultipleModelKalmanFilter(models=[kf_stationary, kf_linear, kf_acc])

# Initialize MMKF for each source
mmkfs = [make_mmkf(pos) for pos in init_positions]

# Store trajectories
estimated_positions = [[] for _ in range(num_sources)]

period = 1
for t in tqdm(range(num_steps)):
    if t % period == 0:
        Z_frame = Z_all[:, t]   # microphone frame
        PnCoord,PnWeight = OMP.initialize(k, Z_frame, MicCoord, PnCoord, PnWeight,
                                          IterN=500, freeze_bias=False)
        PnCoord_reverb = PnCoord[-num_reverb:]
        PnWeight_reverb = PnWeight[-num_reverb:]
        PnCoord_direct = PnCoord[:num_direct]
        MicDereverb = OMP.dereverb(PnCoord_reverb,PnWeight_reverb, k, MicCoord, Z_frame)
        x, omp_error, c = OMP.localize(MicDereverb, PnCoord_direct,MicCoord, k)

        # Get top N detections
        detections = [PnCoord_direct[idx] for idx in c[:num_sources]]

        # Update each MMKF with its detection
        for i, (mmkf, det) in enumerate(zip(mmkfs, detections)):
            fused_state, model_probs = mmkf.step(det)
            if isinstance(fused_state, torch.Tensor):
                fused_state = fused_state.cpu().numpy().reshape(-1)
            fused_pos = fused_state[:3]
            estimated_positions[i].append(fused_pos)

# Save trajectories
for i in range(num_sources):
    np.savetxt(results_dir + f'estimated_positions_source{i+1}.csv',
               estimated_positions[i], delimiter=',')
