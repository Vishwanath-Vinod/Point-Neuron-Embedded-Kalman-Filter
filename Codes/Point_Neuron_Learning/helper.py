import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.ndimage import gaussian_filter

# Adding AWGN to Signal
def awgn(signal, snr_db):
    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.sqrt(noise_power) * np.random.randn(*signal.shape)
    return signal + noise

# Plotting
def plot_field(PLACAR, title, value, filename, clim=None, cmap=None, cbar_ticks=None,z=0.525):

    # Mesh Plot
    SizeX, SizeY = 1.3, 1.6
    Resolution = 0.025
    x = np.arange(0, SizeX+Resolution, Resolution)
    y = np.arange(0, SizeY+Resolution, Resolution)
    Px, Py = np.meshgrid(x, y)
    plt.figure()

    #Heatmap of given field
    PLACAR_2D = PLACAR[:, :2]
    field = griddata(PLACAR_2D, value, (Px, Py), method='linear')
    plt.pcolormesh(Px, Py, np.real(field), shading='auto', cmap=cmap)
    cbar = plt.colorbar()
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')

    # Target region boundary
    theta = np.linspace(0, 2 * np.pi, 1000)
    targetX = np.sqrt(0.5**2 - (z-0.525)**2)*np.cos(theta) + 0.65
    targetY = np.sqrt(0.5**2 - (z-0.525)**2)*np.sin(theta) + 0.80
    plt.plot(targetX, targetY, 'k-', linewidth=1)

    plt.title(title)
    if clim:
        plt.clim(clim)
    plt.savefig(filename, dpi=300, bbox_inches='tight')

def plot_loss(Loss, results_dir):
    plt.figure(figsize=(8, 5))
    ax_main = plt.gca()
    ax_main.plot(Loss, label='Loss')
    ax_main.set_title("Loss Curve with Zoom-In View")
    ax_main.set_xlabel("Iteration number")
    ax_main.set_ylabel("Loss")
    ax_main.set_yscale('log')  # Set y-axis to log scale
    ax_main.grid(False)
    plt.legend()
    plt.savefig(f'{results_dir}/loss.png')

def plot_compare():
    loss_manual = np.loadtxt('Results/Manual/loss.csv')
    loss_auto = np.loadtxt('Results/Auto/loss.csv')
    with open('Results/Manual/time.txt') as f:
        time_manual = float(f.read())
    with open('Results/Auto/time.txt') as f:
        time_auto = float(f.read())

    plt.figure(figsize=(8, 5))

    # Plot both losses
    plt.plot(loss_manual, label='Manual Gradient Computation', color='blue')
    plt.plot(loss_auto, label='Autograd', color='red')

    # Log scale on y-axis
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Convergence Comparison (Log Scale)")
    plt.grid(True)
    plt.tight_layout()

    # Add final value annotations
    manual_line = f"Manual        {loss_manual[-1]:>10.2e}     {time_manual:>6.1f}s"
    auto_line   = f"Auto          {loss_auto[-1]:>10.2e}     {time_auto:>6.1f}s"
    header      = "Method        Final Loss       Time"
    full_text   = f"{header}\n{manual_line}\n{auto_line}"

    # Draw white box
    plt.gca().text(0.02, 0.98, full_text,
                   transform=plt.gca().transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                   family='monospace')  # monospaced font for alignment

    plt.tight_layout()
    plt.legend()
    plt.savefig('Results/Compare/loss_comparison.png')

def complex_converter(val):
    # Convert from bytes (if needed) and strip the parentheses
    if isinstance(val, bytes):
        val = val.decode('utf-8')
    return complex(val.strip('()'))

if __name__ == "__main__":
    plot_compare()
