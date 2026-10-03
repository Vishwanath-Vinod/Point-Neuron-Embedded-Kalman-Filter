import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.random import default_rng
from point_neuron import PointNeuron
import os
import time
from helper import plot_loss,plot_field
from xsrp.xsrp.conventional_srp import ConventionalSrp

# For 3D Point Neuron Learning stick to Automatic Gradient Descent using autograd functionality of Pytorch
freeze_bias = False

# Create Results Folder
mode = 'Auto'
pn_dir = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/MICS=80 (1)/'
results_dir = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/MICS=80 (1)/'
data_dir    = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/MICS=80 (1)/f=1200/'
os.makedirs(results_dir, exist_ok=True)
# Set random seed for reproducibility
rng = default_rng(1)

# Parameters
Frequency = 800
c = 343
k = 2 * np.pi * Frequency / c
TargetR = 0.50
IterN = 20000
StepW = 1e-3 #0.001
StepC = [0.0005, 0.0005, 0.0005] #0.02
Lambda = 0.005 # 0.005 #0.01
L = (3.2,3.6,2.2) # room_dimensions
room_dims = np.array([3.2,3.6,2.2])
eps =1e-8
fs = 16000  # Sampling rate
grid_type = "3D"
n_grid_cells = 20

# Load microphone coordinates and measurements of sound field
mic_data = loadmat(data_dir +'MIC_DATA.mat')
MICCAR = mic_data['MicCoord']
MicPrimaryField = mic_data['MicData'].flatten()
MicN = MICCAR.shape[0]

# --- Apply Conventional SRP ---
srp = ConventionalSrp(
    fs=fs,
    grid_type=grid_type,
    n_grid_cells=n_grid_cells,
    mic_positions=MICCAR,
    room_dims=room_dims,
    mode="gcc_phat_freq",
    Frequency = Frequency
)

candidate_grid = srp.create_initial_candidate_grid(room_dims)
signal_features = srp.compute_signal_features(MicPrimaryField)
srp_map = srp.create_srp_map(MICCAR, candidate_grid, signal_features)
estimated_pos, _, _ = srp.grid_search(candidate_grid, srp_map, None, signal_features)
print("Estimated source position:", estimated_pos)

# Load initial point neuron coordinates
pn_data = loadmat(pn_dir + 'PN_INITIALIZATION.mat')
InCoord_full = pn_data['biases']  # shape: (N, 3) or (N, D)
# Replace the first 10 rows of InCoord with estimated_pos
InCoord = InCoord_full.copy()
InCoord[:10, :] = estimated_pos
PnN = InCoord.shape[0]


InWeight = np.random.rand(PnN, MicN)
InWeight = InWeight / InWeight.sum(axis=1, keepdims=True)  # normalize rows
InWeight = InWeight @ MicPrimaryField  # Weighted avg mic response per PN

# Normalize
NormalFactor = np.linalg.norm(MicPrimaryField)
MicPrimaryField = MicPrimaryField/ NormalFactor

# Point Neuron Learning (Manual or Auto)
model = PointNeuron(k, MicPrimaryField,MICCAR,InCoord, InWeight,StepW, StepC, Lambda, IterN)

start_time = time.time()
if freeze_bias:
    PnCoord, PnWeight, PnWscale, Loss = model.train_freeze_biases()
else:
    PnCoord, PnWeight, PnWscale, Loss = model.pseudo_inverse()
end_time = time.time()

PnCoord = PnCoord.cpu().numpy()
PnWeight = PnWeight.cpu().numpy()
PnWscale = PnWscale.cpu().numpy()
# Collapse first 10 PNs into one
n = 10
top_coords = PnCoord[:n]                   # shape: (10, 3)
top_weights = PnWeight[:n]                # shape: (10,)
indx = np.argmax(np.abs(top_weights))
new_coord = top_coords[indx]
print(new_coord)
with open(os.path.join(results_dir, 'time.txt'), 'w') as f:
    f.write(f"{end_time - start_time:.2f}")

weights_file_csv = os.path.join(results_dir, 'pn_weights.csv')
np.savetxt(weights_file_csv, PnWeight, delimiter=',')

bias_file = os.path.join(results_dir, 'pn_biases.csv')
np.savetxt(bias_file, PnCoord, delimiter=',')

scale_file = os.path.join(results_dir, 'pn_scale.csv')
np.savetxt(scale_file, PnWscale, delimiter=',')





