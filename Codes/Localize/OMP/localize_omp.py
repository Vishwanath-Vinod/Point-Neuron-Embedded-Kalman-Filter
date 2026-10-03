import numpy as np
import os
import pandas as pd
from scipy.io import loadmat
from helper import complex_converter
from omp import OMP

# -------------------
# Parameters
# -------------------
Frequency = 1200
c = 343
k = 2 * np.pi * Frequency / c

data_dir_base = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/3sources_MICS=95/f=1200/'
results_dir_base = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/3sources_MICS=95/f=1200/'

num_trials = 1
num_reverb = 583
num_direct = 1000

'''
Trial 1 Source: [1.67 1.5 0.852]
Trial 2 Source: [1.7 1.32 1.19]
Trial 3 Source: [1.52 1.92 0.834]
Trial 4 Source: [1.32 1.57 0.681]
Trial 5 Source: [2.02 1.73 0.847]
Trial 6 Source: [1.66 1.97 0.952]
Trial 7 Source: [1.63 2.04 0.75]
Trial 8 Source: [1.54 1.56 1.4]
Trial 9 Source: [1.75 1.47 0.749]
Trial 10 Source: [1.24 2.03 1.08]
Trial 11 Source: [1.14 1.62 0.637]
Trial 12 Source: [1.94 1.9 1.56]
Trial 13 Source: [1.3 1.33 1.33]
Trial 14 Source: [1.4 2.02 0.791]
Trial 15 Source: [1.71 2.24 1.2]
Trial 16 Source: [1.86 2.15 1.44]
Trial 17 Source: [1.92 1.53 0.658]
Trial 18 Source: [1.35 2.07 1.54]
Trial 19 Source: [1.21 1.47 0.726]
Trial 20 Source: [1.3 1.52 0.644]

Trial 1 Source: [1.67 1.5 0.852]
Trial 2 Source: [1.53 1.64 1.4]
Trial 3 Source: [1.81 1.77 1.55]
Trial 4 Source: [2.08 1.47 1.57]
Trial 5 Source: [2.09 1.48 1.17]
Trial 6 Source: [1.42 2.28 1.24]
Trial 7 Source: [2.08 2.17 1.05]
Trial 8 Source: [1.23 2.07 1.44]
Trial 9 Source: [1.53 1.4 0.715]
Trial 10 Source: [1.27 1.48 1.44]
Trial 11 Source: [2.06 2.26 0.638]
Trial 12 Source: [1.97 1.39 1.28]
Trial 13 Source: [2.04 1.91 0.814]
Trial 14 Source: [1.45 2.22 1.44]
Trial 15 Source: [1.46 1.38 1.03]
Trial 16 Source: [1.13 2.2 1.02]
Trial 17 Source: [1.2 1.5 1.05]
Trial 18 Source: [1.64 2.13 0.632]
Trial 19 Source: [1.29 2.06 0.75]
Trial 20 Source: [1.98 1.83 1.38]
'''
# -------------------
# Actual source positions (given)
# -------------------
'''
80 mics
actual_positions = [[1.67, 1.5 ,0.852],
[1.31, 1.55, 1.45],
[1.73, 1.59, 0.609],
[1.21, 2.09, 0.707],
[1.8,1.99, 0.853],
[1.38, 1.83, 0.939],
[1.28, 1.69,0.921],
[2.03, 2.16, 0.767],
[1.65, 1.49, 1.04],
[1.17, 2.01, 0.93],
[1.87, 1.55,1.26],
[1.38, 2.06, 0.84],
[1.71, 1.8, 0.921],
[1.72, 1.71, 0.65],
[1.48, 1.48, 0.668],
[1.28, 1.94, 1.34],
[1.26, 2.03, 1.49],
[1.41, 1.8, 1.09],
[1.71, 1.72, 0.986],
[1.5, 1.61, 0.838],
]
'''
#40 mics
actual_positions= [
    [1.67, 1.50, 0.852],
    [1.70, 1.32, 1.190],
    [1.52, 1.92, 0.834],
    [1.32, 1.57, 0.681],
    [2.02, 1.73, 0.847],
    [1.66, 1.97, 0.952],
    [1.63, 2.04, 0.750],
    [1.54, 1.56, 1.400],
    [1.75, 1.47, 0.749],
    [1.24, 2.03, 1.080],
    [1.14, 1.62, 0.637],
    [1.94, 1.90, 1.560],
    [1.30, 1.33, 1.330],
    [1.40, 2.02, 0.791],
    [1.71, 2.24, 1.200],
    [1.86, 2.15, 1.440],
    [1.92, 1.53, 0.658],
    [1.35, 2.07, 1.540],
    [1.21, 1.47, 0.726],
    [1.30, 1.52, 0.644]
]

# -------------------
# Store results
# -------------------
results = []

for trial in range(1, num_trials + 1):
    print(f"Processing trial {trial}...")

    # File paths
    mic_file = os.path.join(data_dir_base, f"MIC_DATA_s{trial}.mat")
    trial_results_dir = os.path.join(results_dir_base, f"trial_{trial}")
    bias_file_csv = os.path.join(trial_results_dir, 'pn_biases.csv')
    weights_file_csv = os.path.join(trial_results_dir, 'pn_weights.csv')
    scale_file_csv = os.path.join(trial_results_dir, 'pn_scale.csv')

    # Load microphone data
    mic_data = loadmat(mic_file)
    MicCoord = mic_data['MicCoord']
    MicPrimaryField = mic_data['MicData'].flatten()

    # Load PN data
    PnCoord = np.loadtxt(bias_file_csv, delimiter=',')
    PnWeight = np.genfromtxt(weights_file_csv, delimiter=',', dtype=complex, converters={0: complex_converter})
    PnScale = np.genfromtxt(scale_file_csv, delimiter=',', dtype=complex, converters={0: complex_converter})

    # -------------------
    # Dereverberation
    # -------------------
    PnCoord_reverb = PnCoord[-num_reverb:]
    PnWeight_reverb = PnWeight[-num_reverb:]

    diff = PnCoord_reverb[:, None, :] - MicCoord[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    hn = np.exp(1j * k * dist) / (dist * 4 * np.pi)
    dist_source = np.linalg.norm(PnCoord_reverb, axis=1, keepdims=True)
    scale = dist_source * np.exp(-1j * k * dist_source)
    mic_reverb = np.sum(PnWeight_reverb[:, None] * hn * scale, axis=0)
    MicDereverb = MicPrimaryField - mic_reverb

    # -------------------
    # Localization
    # -------------------
    PnCoord_direct = PnCoord[:num_direct]
    PnWeight_direct = PnWeight[:num_direct]

    diff = PnCoord_direct[:, None, :] - MicCoord[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    hn = np.exp(1j * k * dist) / (dist * 4 * np.pi)
    dist_source = np.linalg.norm(PnCoord_direct, axis=1, keepdims=True)
    scale = dist_source * np.exp(-1j * k * dist_source)
    Psi = (hn * scale).T

    x, omp_error, c = OMP.localize(MicDereverb, Psi, sparsity=10)
    est_pos = PnCoord_direct[c[0]]  # Take first localized PN as estimate

    print(f"Trial s{trial} Estimated Position: {est_pos}")
    results.append([f"s{trial}", actual_positions[trial-1], est_pos.tolist()])

# -------------------
# Save results to CSV
# -------------------
df = pd.DataFrame(results, columns=["Trial", "Actual", "Estimated"])
csv_output_path = os.path.join(results_dir_base, "localization_results.csv")
df.to_csv(csv_output_path, index=False)

print(f"\nAll results saved to {csv_output_path}")
