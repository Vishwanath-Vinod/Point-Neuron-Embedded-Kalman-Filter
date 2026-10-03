import torch
import numpy as np
import os
from scipy.io import loadmat
from tqdm import tqdm
from mmkf import KalmanFilter, MultipleModelKalmanFilter
from xsrp.xsrp.conventional_srp import ConventionalSrp

Frequency = 900
room_dims = np.array([3.2, 3.6, 2.2])
results_dir = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/PAPER_FINAL_PLOTS/MMKF+SRP/f=900/'
data_dir    = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/PAPER_FINAL_PLOTS/PNEKF_N=1/f=900/'

os.makedirs(results_dir, exist_ok=True)

# Load microphone coordinates and sound field measurements
mic_coordinates = loadmat(data_dir + 'MIC_DATA_ms (2).mat')
MicCoord = mic_coordinates['MicCoord']                # (Q,3)
MicData  = mic_coordinates['MicData']

mic_data = loadmat(data_dir + 'Measurements (2).mat')
Z_all = mic_data['MicData_time_noisy']               # shape: (num_mics, num_timesteps)

num_steps = Z_all.shape[1]
MicN = MicCoord.shape[0]

# SRP setup
fs = 16000
grid_type = "3D"
n_grid_cells = 20
srp = ConventionalSrp(
    fs=fs,
    grid_type=grid_type,
    n_grid_cells=n_grid_cells,
    mic_positions=MicCoord,
    room_dims=room_dims,
    mode="gcc_phat_freq",
    Frequency=Frequency
)
MicData = np.squeeze(MicData)
candidate_grid = srp.create_initial_candidate_grid(room_dims)
candidate_grid1 = srp.create_initial_candidate_grid(room_dims, type='tracking')

# initial SRP-based guess (fallback)
signal_features = srp.compute_signal_features(MicData)
srp_map = srp.create_srp_map(MicCoord, candidate_grid, signal_features)
estimated_positions, _, _ = srp.grid_search(candidate_grid, srp_map, None, signal_features)
est_position = np.array(estimated_positions[0])
print(est_position)

acc = loadmat(data_dir + 'Process_noise.mat')
process_std = np.sqrt(acc['accel_std_estimated'][0])   # shape (3,)

R = np.eye(3) * 1e-2

dt = 1.0
state_dim = 9

error_cov_stationary = np.diag([
    1e-6, 1e-6, 1e-6,    # position very certain
    1e3 , 1e3 , 1e3,     # velocity very uncertain
    1e3 , 1e3 , 1e3      # acceleration very uncertain
])

error_cov_linear = np.diag([
    1e-6, 1e-6, 1e-6,    # position certain
    1e-4, 1e-4, 1e-4,    # velocity fairly certain
    1e2 , 1e2 , 1e2      # acceleration uncertain
])

error_cov_acc = np.diag([
    1e-3, 1e-3, 1e-3,    # position moderate
    1e-2, 1e-2, 1e-2,    # velocity moderate
    1e-1, 1e-1, 1e-1     # acceleration moderate
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

for t in tqdm(range(num_steps)):
    if t % period == 0:
        Z_frame = Z_all[:, t]   # microphone frame
        signal_features = srp.compute_signal_features(Z_frame)
        srp_map = srp.create_srp_map(MicCoord, candidate_grid1, signal_features)
        est_positions, _, _ = srp.grid_search(candidate_grid1, srp_map, None, signal_features)
        meas_pos = np.array(est_positions[0])
        print("SRP estimate:",meas_pos)
        # Run MMKF step: returns fused state (9-D) and possibly covariances/probs
        fused_state, model_probs = mmkf.step(meas_pos)
        #print("FUSED STATE:", fused_state)
        if isinstance(fused_state, torch.Tensor):
            fused_state = fused_state.cpu().numpy().reshape(-1)

        fused_pos = fused_state[:3]   # x,y,z
        fused_vel = fused_state[3:6]  # vx,vy,vz
        fused_acc = fused_state[6:9]  # ax,ay,az
        '''
        # If you want to estimate velocity from delta-position instead of fused vx:
        estimated_vel_from_delta = (fused_pos - prev_fused_pos) / dt
        # Example: overwrite fused velocity with delta estimate if you prefer
        fused_state[3:6] = estimated_vel_from_delta
        prev_fused_pos = fused_pos.copy()
        '''
        print(f"POSITION at time {t+1}", fused_pos)
        print(f"VELOCITY at time {t+1}", fused_vel)
        print(f"ACCELERATION at time {t+1}", fused_acc)
        estimated_positions.append(fused_pos)

estimated_positions = np.stack(estimated_positions)
np.savetxt(results_dir + 'estimated_positions (2).csv', estimated_positions, delimiter=',')