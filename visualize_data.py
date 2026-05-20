# May 2026

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os

# settings (must match the simulator that generated the file)
sampling_frequency = 2048   # Hz
duration = 2                # s
f_min = 20.0                # Hz

# ask for input file and number of plots
input_path = input("Input path: ").strip().strip("'\"")
n_plots = int(input("How many random datapoints to plot? "))

# output dir based on the input filename
base_name = os.path.splitext(os.path.basename(input_path))[0]
output_dir = f"plots_{base_name}"
os.makedirs(output_dir, exist_ok=True)

# load parameters and the dataset shape
with h5py.File(input_path, "r") as f:
    chirp_masses = f["chirp_mass"][:]
    distances = f["luminosity_distance"][:]
    n_datapoints = f["data"].shape[0]
    n_time = f["data"].shape[1]

# rebuild time and frequency axes
time_array = np.arange(n_time) / sampling_frequency
freq_array = np.fft.rfftfreq(n_time, d=1.0 / sampling_frequency)
freq_mask = freq_array >= f_min

# pick random indices
indices = np.random.choice(n_datapoints, size=min(n_plots, n_datapoints), replace=False)

# make the plots
with h5py.File(input_path, "r") as f:
    for idx in indices:
        signal = f["signal"][idx]
        noise = f["noise"][idx]
        data = f["data"][idx]

        # FFT magnitudes for the frequency domain panels
        signal_fd = np.abs(np.fft.rfft(signal) / sampling_frequency)
        noise_fd  = np.abs(np.fft.rfft(noise)  / sampling_frequency)
        data_fd   = np.abs(np.fft.rfft(data)   / sampling_frequency)

        # 3 rows (signal/noise/data) x 2 cols (time/frequency)
        fig, axes = plt.subplots(3, 2, figsize=(11, 8))

        # time domain, left column
        axes[0, 0].plot(time_array, signal, color="C0", linewidth=0.7)
        axes[0, 0].set_ylabel("signal")
        axes[1, 0].plot(time_array, noise, color="C1", linewidth=0.7)
        axes[1, 0].set_ylabel("noise")
        axes[2, 0].plot(time_array, data, color="C2", linewidth=0.7)
        axes[2, 0].set_ylabel("signal + noise")
        axes[2, 0].set_xlabel("time [s]")

        # frequency domain, right column
        axes[0, 1].loglog(freq_array[freq_mask], signal_fd[freq_mask], color="C0", linewidth=0.7)
        axes[1, 1].loglog(freq_array[freq_mask], noise_fd[freq_mask],  color="C1", linewidth=0.7)
        axes[2, 1].loglog(freq_array[freq_mask], data_fd[freq_mask],   color="C2", linewidth=0.7)
        axes[2, 1].set_xlabel("frequency [Hz]")

        # grids
        for ax in axes.flat:
            ax.grid(alpha=0.3)

        # title
        fig.suptitle(
            f"datapoint #{idx}   |   "
            rf"$\mathcal{{M}}_c = {chirp_masses[idx]:.2f}\, M_\odot$   |   "
            rf"$d_L = {distances[idx]:.0f}$ Mpc"
        )

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"datapoint_{idx:05d}.png")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"saved {out_path}")

print(f"done, {len(indices)} plots written to {output_dir}/")