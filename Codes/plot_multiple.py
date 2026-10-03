import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob, os

sns.set(style="ticks", rc={"axes.grid": False})

# Directories
base_dir = '/mnt/e/Drive E/Intern@ANU/Point-Neuron-Learning/EKF_PNL/PAPER_FINAL_PLOTS/Multiple/'
data_dir = base_dir  # ground truth directory
results_dirs = {
    "PNEKF": base_dir + "PNEKF/f={}/",
    "OMP-MMKF": base_dir + "MMKF+OMP/f={}/",
    "OMP-KF": base_dir + "KF+OMP/f={}/"
}

# Number of sources and trajectories per source
num_sources = 3
num_trajs = 2

def load_and_compute_rmse(folder, gt_folder, num_sources, num_trajs):
    """
    Load estimated trajectories + ground truths, compute RMSE.
    Returns: total RMSE (summed across sources).
    """
    total_rmse_list = []

    for s in range(1, num_sources + 1):
        traj_rmse_list = []

        for t in range(1, num_trajs + 1):
            est_file = os.path.join(folder, f"estimated_positions_source{s}_{t}.csv")
            gt_file  = os.path.join(gt_folder, f"source{s}_{t}.csv")

            if not (os.path.exists(est_file) and os.path.exists(gt_file)):
                continue

            est = np.loadtxt(est_file, delimiter=",")
            gt  = np.loadtxt(gt_file, delimiter=",")
            rmse = np.sqrt(np.mean((est - gt) ** 2, axis=1))  # per timestep
            traj_rmse_list.append(rmse)

        if len(traj_rmse_list) > 0:
            traj_rmse_array = np.stack(traj_rmse_list, axis=0)  # (num_trajs, num_timesteps)
            source_avg_rmse = np.mean(traj_rmse_array, axis=0)  # avg across trajs
            total_rmse_list.append(source_avg_rmse)

    if len(total_rmse_list) == 0:
        return None, None

    total_rmse_array = np.stack(total_rmse_list, axis=0)  # (num_sources, num_timesteps)
    total_rmse = np.average(total_rmse_array, axis=0)  # sum across sources

    return total_rmse, None  # no std since deterministic here

# Frequencies to plot
freqs = [800, 1200]

fig, axes = plt.subplots(1, len(freqs), figsize=(10, 5), sharey=True)

# Color + style mapping
styles = {
    "PNEKF": ("red", "-.", 3.5),
    "OMP-KF": ("green", "--", 3.5),
    "OMP-MMKF": ("blue", "-", 3.5)
}

# For legend handles
handles = []
labels = []

for ax, f in zip(axes, freqs):
    x = None
    for method, path_template in results_dirs.items():
        folder = path_template.format(f)
        mean, _ = load_and_compute_rmse(folder, data_dir, num_sources, num_trajs)

        if mean is None:
            continue

        x = range(len(mean))
        color, style, lw = styles[method]
        h, = ax.plot(mean, color=color, linestyle=style, linewidth=lw, label=method)

        if f == freqs[0]:  # only collect legend handles once
            handles.append(h)
            labels.append(method)

    ax.set_title(rf"$\mathbf{{f = {f}}}$ Hz", fontsize=20, fontweight="bold")
    ax.set_xlabel("Time Frames", fontsize=18, fontweight="bold")
    if x is not None:
        xticks = np.arange(0, len(mean)+1, 100)
        ax.set_xticks(xticks)

    ax.tick_params(axis="both", labelsize=18)

axes[0].set_ylabel("Tracking Error (m)", fontsize=18, fontweight="bold")

# Put legend in a box
fig.legend(handles, labels,
           loc="lower center", ncol=4,
           frameon=True,
           edgecolor="black", facecolor="white",
           prop={"weight": "bold", "size": 20})

plt.tight_layout(rect=[0, 0.15, 1, 1])

plt.savefig(base_dir + "Multiple_comparison (2).png", dpi=600)
plt.show()