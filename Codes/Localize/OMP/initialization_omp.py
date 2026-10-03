import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from numpy.random import default_rng
from point_neuron import PointNeuron
import os
import time

# For 3D Point Neuron Learning stick to Automatic Gradient Descent using autograd functionality of Pytorch
freeze_bias = False

# Directories
mode = 'Auto'
base_results_dir = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/3sources/Mixed/f=1200/moving_pns/'
data_dir    = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/3sources/Mixed/f=1200/'  # Folder with MIC_DATA_s*.mat

os.makedirs(base_results_dir, exist_ok=True)

# Set random seed for reproducibility
rng = default_rng(1)

# Parameters
Frequency = 1200
c = 343
k = 2 * np.pi * Frequency / c
TargetR = 0.50
IterN = 20000
StepW = 5e-3
StepC = [1e-4,1e-4,1e-4]
Lambda = 0.005
L = (3.2,3.6,2.2)
room_dims = np.array([3.2,3.6,2.2])
eps = 1e-8
fs = 16000
grid_type = "3D"
n_grid_cells = 20

# Load initial point neuron coordinates
pn_init_file = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/MICS=95_movingpns/f=1200/PN_INITIALIZATION (4).mat'
pn_data = loadmat(pn_init_file)
InCoord_full = pn_data['biases']
PnN = InCoord_full.shape[0]

# Loop through all 20 trials
for trial in range(1, 2):

    print(f"Processing trial {trial}...")

    # Create results subfolder for this trial
    trial_results_dir = base_results_dir
    os.makedirs(trial_results_dir, exist_ok=True)

    # Load microphone data for this trial
    mic_file = os.path.join(data_dir, f"MIC_DATA_ms.mat")
    mic_data = loadmat(mic_file)
    MICCAR = mic_data['MicCoord']
    MicPrimaryField = mic_data['MicData'].flatten()
    MicN = MICCAR.shape[0]

    # Initialize weights
    InCoord = InCoord_full.copy()
    InWeight = np.random.rand(PnN, MicN)
    InWeight = InWeight / InWeight.sum(axis=1, keepdims=True)  # normalize rows
    InWeight = InWeight @ MicPrimaryField  # Weighted avg mic response per PN
    
    # Normalize
    NormalFactor = np.linalg.norm(MicPrimaryField)
    MicPrimaryField = MicPrimaryField / NormalFactor
    # Create model and train
    model = PointNeuron(k, MicPrimaryField, MICCAR, InCoord, InWeight, StepW, StepC, Lambda, IterN)

    start_time = time.time()
    if freeze_bias:
        PnCoord, PnWeight, PnWscale, Loss = model.train_freeze_biases()
    else:
        PnCoord, PnWeight, PnWscale, Loss = model.pseudo_inverse()
    end_time = time.time()

    # Convert tensors to numpy
    PnCoord = PnCoord.cpu().numpy()
    PnWeight = PnWeight.cpu().numpy()
    PnWscale = PnWscale.cpu().numpy()

    # Save results
    with open(os.path.join(trial_results_dir, 'time.txt'), 'w') as f:
        f.write(f"{end_time - start_time:.2f}")

    np.savetxt(os.path.join(trial_results_dir, 'pn_weights.csv'), PnWeight, delimiter=',')
    np.savetxt(os.path.join(trial_results_dir, 'pn_biases.csv'), PnCoord, delimiter=',')
    np.savetxt(os.path.join(trial_results_dir, 'pn_scale.csv'), PnWscale, delimiter=',')


print("All trials processed successfully.")
