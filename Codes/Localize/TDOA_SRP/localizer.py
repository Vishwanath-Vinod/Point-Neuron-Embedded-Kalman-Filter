import os
import numpy as np
from scipy.io import loadmat
import csv
from point_neuron import PointNeuron
from xsrp.xsrp.conventional_srp import ConventionalSrp

# === Path Setup ===
base_data_dir = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/Data/Measurements/Localize_f=1200/'
base_results_dir = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/Codes/'
data_dir    = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/BR_m=120,f=1200/'
os.makedirs(base_results_dir, exist_ok=True)

# === Parameters ===
c = 343
fs = 16000
room_dims = np.array([3.2, 3.6, 2.2])
grid_type = "3D"
n_grid_cells = 15
IterN = 20000
StepW = 1e-3
StepC = [0.0005, 0.0005, 0.0005]
Lambda = 0.005
freeze_bias = False

# === Output CSV ===
csv_output_file = os.path.join(base_results_dir, 'Localize_f=1200_PseudoInverse.csv')
with open(csv_output_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Source_ID', 'GroundTruth_XYZ', 'Estimated_XYZ', 'MSE', 'SRP_Estimate', 'MSE_SRP'])

    # === Loop Over All 50 Source Files ===
    for i in range(1, 51):
        filename = os.path.join(base_data_dir, f'source_{i:02d}.mat')
        if not os.path.exists(filename):
            print(f"Missing: {filename}")
            continue

        data = loadmat(filename)
        MicCoord = data['MicCoord']
        MicData = data['MicData'].flatten()
        GroundTruth = data['src'].flatten()
        MicData = MicData / np.linalg.norm(MicData)

        # --- Estimate frequency from original MATLAB (1200 Hz here) ---
        Frequency = 1200
        k = 2 * np.pi * Frequency / c

        # --- SRP Estimation ---
        srp = ConventionalSrp(
            fs=fs,
            grid_type=grid_type,
            n_grid_cells=n_grid_cells,
            mic_positions=MicCoord,
            room_dims=room_dims,
            mode="gcc_phat_freq",
            Frequency=Frequency
        )
        candidate_grid = srp.create_initial_candidate_grid(room_dims)
        signal_features = srp.compute_signal_features(MicData)
        srp_map = srp.create_srp_map(MicCoord, candidate_grid, signal_features)
        estimated_pos, _, _ = srp.grid_search(candidate_grid, srp_map, None, signal_features)

        # --- PNL Initialization ---
        pn_init_path = os.path.join(data_dir, 'PN_INITIALIZATION.mat')
        if not os.path.exists(pn_init_path):
            raise FileNotFoundError("PN_INITIALIZATION.mat missing.")

        pn_data = loadmat(pn_init_path)
        InCoord_full = pn_data['biases']
        InCoord = InCoord_full.copy()
        InCoord[:10, :] = estimated_pos

        PnN = InCoord.shape[0]
        MicN = MicCoord.shape[0]
        InWeight = np.random.rand(PnN, MicN)
        InWeight /= InWeight.sum(axis=1, keepdims=True)
        InWeight = InWeight @ MicData

        model = PointNeuron(k, MicData, MicCoord, InCoord, InWeight, StepW, StepC, Lambda, IterN)

        if freeze_bias:
            PnCoord, PnWeight, PnWscale, _ = model.train_freeze_biases()
        else:
            PnCoord, PnWeight, PnWscale, _ = model.pseudo_inverse()

        PnCoord = PnCoord.cpu().numpy()
        PnWeight = PnWeight.cpu().numpy()
        top_coords = PnCoord[:10]
        top_weights = PnWeight[:10]
        indx = np.argmax(np.abs(top_weights))
        FinalEstimate = top_coords[indx]

        mse_dist = np.linalg.norm(FinalEstimate - GroundTruth)
        mse_srp = np.linalg.norm(estimated_pos[0] - GroundTruth)

        # === Write to CSV ===
        writer.writerow([
            i,
            f"[{GroundTruth[0]:.4f} {GroundTruth[1]:.4f} {GroundTruth[2]:.4f}]",
            f"[{FinalEstimate[0]:.4f} {FinalEstimate[1]:.4f} {FinalEstimate[2]:.4f}]",
            f"{mse_dist:.4f}",
            f"[{estimated_pos[0][0]:.4f} {estimated_pos[0][1]:.4f} {estimated_pos[0][2]:.4f}]",
            f"{mse_srp:.4f}"
        ])

        print(f"Source {i:02d}: GT [{GroundTruth[0]:.2f}, {GroundTruth[1]:.2f}, {GroundTruth[2]:.2f}] | "
              f"PNL Est [{FinalEstimate[0]:.2f}, {FinalEstimate[1]:.2f}, {FinalEstimate[2]:.2f}] | "
              f"SRP Est [{estimated_pos[0][0]:.2f}, {estimated_pos[0][1]:.2f}, {estimated_pos[0][2]:.2f}]")
