import torch
import numpy as np
import os
from scipy.io import loadmat
from tqdm import tqdm
from mmkf import KalmanFilter, MultipleModelKalmanFilter
from omp import OMP

Frequency = 900
num_reverb = 600
num_direct = 1000
dereverb = True
c = 343
k = 2 * np.pi * Frequency / c
room_dims = np.array([3.2, 3.6, 2.2])
pn_init_file = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/MICS=95/PN_INITIALIZATION_omp.mat'
results_dir = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/PAPER_FINAL_PLOTS/MMKF+OMP/f=900/'
data_dir    = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/PAPER_FINAL_PLOTS/PNEKF_N=1/f=900/'
os.makedirs(results_dir, exist_ok=True)

# Load microphone coordinates and sound field measurements
mic_coordinates = loadmat(data_dir + 'MIC_DATA_ms (2).mat')
MicCoord = mic_coordinates['MicCoord']   
MicData = mic_coordinates['MicData'].flatten()
mic_data = loadmat(data_dir + 'Measurements (2).mat')
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
PnCoord,PnWeight = OMP.initialize(k, MicData, MicCoord, InCoord, InWeight,IterN=10000,freeze_bias=False)
PnCoord_reverb = PnCoord[-num_reverb:]
PnWeight_reverb = PnWeight[-num_reverb:]
PnCoord_direct = PnCoord[:num_direct]
MicDereverb = OMP.dereverb(PnCoord_reverb,PnWeight_reverb, k, MicCoord, MicData)
x, omp_error, c = OMP.localize(MicDereverb, PnCoord_direct,MicCoord, k)
est_position = PnCoord_direct[c[0]]  
print(f'Localization Result: {est_position} ')

acc = loadmat(data_dir + 'Process_noise.mat')
process_std = np.sqrt(acc['accel_std_estimated'][0])   # shape (3,)

R = np.eye(3) * 1e-4

dt = 1.0
state_dim = 9

error_cov_stationary = np.diag([
    1e-6, 1e-6, 1e-6,    # position very certain
    1e3 , 1e3 , 1e3,     # velocity very uncertain
    1e3 , 1e3 , 1e3      # acceleration very uncertain
])

error_cov_linear = np.diag([
    1e-6, 1e-6, 1e-6,    # position certain
    1e-5, 1e-5, 1e-5,    # velocity fairly certain
    1e3 , 1e3 , 1e3      # acceleration uncertain
])

error_cov_acc = np.diag([
    1e-6, 1e-6, 1e-6,    # position moderate
    1e-5, 1e-5, 1e-5,    # velocity moderate
    1e-4, 1e-4, 1e-4     # acceleration moderate
])

x0_stationary = np.hstack([est_position, np.zeros(3), np.zeros(3)])
x0_linear     = np.hstack([est_position, np.ones(3)*1e-3, np.zeros(3)])
x0_accelerated= np.hstack([est_position, np.ones(3)*1e-3, np.ones(3)*1e-5])

kf_stationary = KalmanFilter(
    X_t = x0_stationary,
    error_covariance = error_cov_stationary,
    process_std = process_std,
    R = R,
    dt = dt,
    model = 'stationary'
)

kf_linear = KalmanFilter(
    X_t = x0_linear,
    error_covariance = error_cov_linear,
    process_std = process_std,
    R = R,
    dt = dt,
    model = 'linear'
)

kf_accelerated = KalmanFilter(
    X_t = x0_accelerated,
    error_covariance = error_cov_acc,
    process_std = process_std,
    R = R,
    dt = dt,
    model = 'accelerated'
)

mmkf = MultipleModelKalmanFilter(models=[kf_stationary, kf_linear, kf_accelerated])   # adapt API if needed

period = 1
prev_fused_pos = est_position.copy()
estimated_positions = []
for t in tqdm(range(num_steps)):
    if t % period == 0:
        Z_frame = Z_all[:, t]   # microphone frame
        PnCoord,PnWeight = OMP.initialize(k, Z_frame, MicCoord, PnCoord, PnWeight,IterN=500,freeze_bias=False)
        PnCoord_reverb = PnCoord[-num_reverb:]
        PnWeight_reverb = PnWeight[-num_reverb:]
        PnCoord_direct = PnCoord[:num_direct]
        MicDereverb = OMP.dereverb(PnCoord_reverb,PnWeight_reverb, k, MicCoord, Z_frame)
        x, omp_error, c = OMP.localize(MicDereverb, PnCoord_direct,MicCoord, k)
        est_position = PnCoord_direct[c[0]]
        print('Measured Estimated Position:',est_position)
        # Run MMKF step: returns fused state (9-D) and possibly covariances/probs
        fused_state, model_probs = mmkf.step(est_position)
        #print("FUSED STATE:", fused_state)
        if isinstance(fused_state, torch.Tensor):
            fused_state = fused_state.cpu().numpy().reshape(-1)

        fused_pos = fused_state[:3]   # x,y,z
        fused_vel = fused_state[3:6]  # vx,vy,vz
        fused_acc = fused_state[6:9]  # ax,ay,az
        print(f"POSITION at time {t+1}", fused_pos)
        print(f"VELOCITY at time {t+1}", fused_vel)
        print(f"ACCELERATION at time {t+1}", fused_acc)
        estimated_positions.append(fused_pos)

np.savetxt(results_dir +'estimated_positions (2).csv', estimated_positions, delimiter=',')