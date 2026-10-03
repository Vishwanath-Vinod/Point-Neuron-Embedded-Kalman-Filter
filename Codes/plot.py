import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob, os

sns.set(style="ticks",rc={"axes.grid": False})

# Directories
base_dir = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/PAPER_FINAL_PLOTS/'
data_dir = base_dir  # ground truth directory
results_dirs = {
    "PNEKF": base_dir + "PNEKF_N=1/f={}/",
    "OMP-MMKF": base_dir + "MMKF+OMP/f={}/",
    "OMP-KF": base_dir + "KF+OMP/f={}/",
    "TDOA-MMKF": base_dir + "MMKF+SRP/f={}/"
}

def load_and_compute_rmse(folder, gt_folder):
    """Load estimated trajectories + corresponding ground truths, compute RMSE."""
    est_files = sorted(glob.glob(os.path.join(folder, "estimated_positions*.csv")))
    gt_files  = sorted(glob.glob(os.path.join(gt_folder, "trajectory*.csv")))

    rmses = []
    for est_file, gt_file in zip(est_files, gt_files):
        est = np.loadtxt(est_file, delimiter=",")
        gt  = np.loadtxt(gt_file, delimiter=",")
        rmse = np.sqrt(np.mean((est - gt) ** 2, axis=1))
        rmses.append(rmse)

    rmses = np.array(rmses)  # shape = (num_runs, num_timesteps)
    return np.mean(rmses, axis=0), np.std(rmses, axis=0) / np.sqrt(rmses.shape[0])

# Frequencies to plot
freqs = [800, 900, 1000, 1100, 1200]

fig, axes = plt.subplots(1, len(freqs), figsize=(20, 5), sharey=True)

# Color + style mapping
styles = {
    "PNEKF": ("red", "-.", 3.5),
    "OMP-KF": ("green", "--", 3.5),
    "OMP-MMKF": ("blue", ":", 3.5),
    "TDOA-MMKF": ("purple", "-", 2.0)
}

# For legend handles
handles = []
labels = []

for ax, f in zip(axes, freqs):
    x = None
    for method, path_template in results_dirs.items():
        folder = path_template.format(f)
        mean, std = load_and_compute_rmse(folder, data_dir)
        x = range(len(mean))

        color, style, lw = styles[method]
        h, = ax.plot(mean, color=color, linestyle=style, linewidth=lw, label=method)
        # ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.0)

        if f == freqs[0]:  # only collect legend handles once
            handles.append(h)
            labels.append(method)

    ax.set_title(rf"$\mathbf{{f = {f}}}$ Hz", fontsize=20, fontweight="bold")
    ax.set_xlabel("Time Frames", fontsize=18, fontweight="bold")
    xticks = np.arange(0, len(mean)+1, 100)
    ax.set_xticks(xticks)

    ax.tick_params(axis="both", labelsize=18)

axes[0].set_ylabel("Tracking Error (m)", fontsize=18, fontweight="bold")
# Put legend in a box
fig.legend(handles, labels,
           loc="lower center", ncol=4,
           frameon=True,  # turn on the box
           edgecolor="black", facecolor="white",  # optional styling
           prop={"weight": "bold", "size": 20})

# Leave extra space below x-axis labels
plt.tight_layout(rect=[0, 0.15, 1, 1])  # increase bottom margin from 0.08 → 0.12

plt.savefig(base_dir + "Comparison_all_frequencies.png", dpi=600)