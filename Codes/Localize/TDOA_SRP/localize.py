import numpy as np
import os
from scipy.io import loadmat
from numpy.random import default_rng
from xsrp.xsrp.conventional_srp import ConventionalSrp
import matplotlib.pyplot as plt


# Parameters
fs = 16000  # Sampling rate
grid_type = "3D"
n_grid_cells = 25
c = 343  # Speed of sound

# Microphone setup: 4 mics in a square, 0.2 m apart
data_dir    = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/Data/Measurements/track_pns/localize/'
# Set random seed for reproducibility
rng = default_rng(1)
# Load microphone coordinates and measurements of sound field
mic_data = loadmat(data_dir +'MIC_DATA (2).mat')
MICCAR = mic_data['MicCoord']

mic_signals = mic_data['MicData'].flatten()
MicN = MICCAR.shape[0]

# Room dimensions
room_dims = np.array([1.3,1.6,1.05])  # meters

# --- Apply Conventional SRP ---
srp = ConventionalSrp(
    fs=fs,
    grid_type=grid_type,
    n_grid_cells=n_grid_cells,
    mic_positions=MICCAR,
    room_dims=room_dims,
    mode="gcc_phat_freq",
    Frequency = 900
)

candidate_grid = srp.create_initial_candidate_grid(room_dims)
signal_features = srp.compute_signal_features(mic_signals)
srp_map = srp.create_srp_map(MICCAR, candidate_grid, signal_features)
estimated_pos, _, _ = srp.grid_search(candidate_grid, srp_map, None, signal_features)
print("Estimated source position:", estimated_pos)
# Convert grid and SRP map to numpy arrays
grid_points = np.array(candidate_grid)       # shape: (N, 3)
srp_values = np.array(srp_map).flatten()     # shape: (N,)

# Choose unique z-slices[0.7923, 0.6738, 0.4884]
z_values = np.unique(grid_points[:, 2])
z_slice = 0.4884
# Filter points in that z-slice
mask = np.abs(grid_points[:, 2] - z_slice) < 0.02
x_slice = grid_points[mask, 0]
y_slice = grid_points[mask, 1]
srp_slice = srp_values[mask]

# Normalize SRP values for color scale
srp_slice = srp_slice - srp_slice.min()
srp_slice = srp_slice / (srp_slice.max() + 1e-8)

# Plot
marker_point = [0.7923, 0.6738, 0.4884]

# Plot
plt.figure(figsize=(8, 6))
sc = plt.scatter(x_slice, y_slice, c=srp_slice, cmap='viridis', s=80, edgecolors='k')
plt.colorbar(sc, label="Normalized SRP Power")

# Mark the point with an 'x'
plt.scatter(marker_point[0], marker_point[1], color='red', marker='x', s=200, linewidths=3, label='Ground Truth')

# Plot settings
plt.title(f"SRP-PHAT Map (Z={z_slice:.2f} m slice)")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('Heatmap.png')
#bias_file = os.path.join(data_dir, 'pn_biases.csv')
#np.savetxt(bias_file,estimated_pos, delimiter=',')
