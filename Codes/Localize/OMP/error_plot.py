import os
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------
# CONFIG
# -------------------
csv_folder = "/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/MICS=120/LOCALIZATION_RESULTS/"
csv_folder1 = "/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/BASELINE/OMP_DATA/MICS=80/LOCALIZATION_RESULTS/"
output_plot = "Localization_error.png"

# Set seaborn dark theme without grid
sns.set_theme(style="dark")
sns.set_style("dark", {"axes.grid": False})

# Get all CSV files in folder

csv_files1 = [f for f in os.listdir(csv_folder1) if f.endswith(".csv")]
csv_files = [f for f in os.listdir(csv_folder) if f.endswith(".csv")]
freqs = []
mse_means = []
mse_stds = []

for file in sorted(csv_files):
    freq_label = ''.join([c for c in file if c.isdigit()])  # Extract frequency from filename
    csv_path = os.path.join(csv_folder, file)

    df = pd.read_csv(csv_path)

    # Parse string arrays into lists
    actual_positions = df["Actual"].apply(ast.literal_eval).to_list()
    estimated_positions = df["Estimated"].apply(ast.literal_eval).to_list()

    actual_positions = np.array(actual_positions)
    estimated_positions = np.array(estimated_positions)

    # MSE per trial
    mse_per_trial = np.sqrt(np.mean((actual_positions - estimated_positions) ** 2, axis=1))

    freqs.append(int(freq_label))
    mse_means.append(np.mean(mse_per_trial))
    mse_stds.append(np.std(mse_per_trial))

# Sort by frequency to ensure x-axis is ordered
sorted_indices = np.argsort(freqs)
freqs = np.array(freqs)[sorted_indices]
mse_means = np.array(mse_means)[sorted_indices]
mse_stds = np.array(mse_stds)[sorted_indices]

freqs1 = []
mse_means1= []
mse_stds1 = []

for file in sorted(csv_files1):
    freq_label = ''.join([c for c in file if c.isdigit()])  # Extract frequency from filename
    csv_path = os.path.join(csv_folder1, file)

    df = pd.read_csv(csv_path)

    # Parse string arrays into lists
    actual_positions = df["Actual"].apply(ast.literal_eval).to_list()
    estimated_positions = df["Estimated"].apply(ast.literal_eval).to_list()

    actual_positions = np.array(actual_positions)
    estimated_positions = np.array(estimated_positions)

    # MSE per trial
    mse_per_trial = np.sqrt(np.mean((actual_positions - estimated_positions) ** 2, axis=1))

    freqs1.append(int(freq_label))
    mse_means1.append(np.mean(mse_per_trial))
    mse_stds1.append(np.std(mse_per_trial))

# Sort by frequency to ensure x-axis is ordered
sorted_indices1 = np.argsort(freqs1)
freqs1 = np.array(freqs1)[sorted_indices1]
mse_means1 = np.array(mse_means1)[sorted_indices1]
mse_stds1 = np.array(mse_stds1)[sorted_indices1]

# -------------------
# Plot
# -------------------
plt.figure(figsize=(8, 5))
plt.errorbar(freqs1, mse_means, yerr=mse_stds, fmt='o', capsize=5,
             label="Num_mics = 120", linestyle='None', color='blue', markersize=6)
plt.errorbar(freqs1, mse_means1, yerr=mse_stds1, fmt='o', capsize=5,
             label="Num_mics = 80", linestyle='None', color='red', markersize=6)
plt.ylim(bottom=0)
plt.xlabel("Frequency (Hz)")
plt.ylabel(" Root Mean Squared Error")
plt.title("Localization Error Across Frequencies Averaged over 20 trials")
plt.xticks(freqs1)
plt.ylim((0,0.1))
plt.legend()
plt.tight_layout()
plt.savefig(output_plot, dpi=300)
plt.close()

print(f"Plot saved as {output_plot}")
