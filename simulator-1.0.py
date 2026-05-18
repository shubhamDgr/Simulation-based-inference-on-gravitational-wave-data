# May 2026
# This python script is made by Angel Valer Melchor and is intended to generate the data that will be used in the SBI 
# project as part of the ML4PA course

import bilby
import numpy as np
import matplotlib.pyplot as plt
import h5py

# begin configurations

# script configurations
n_datapoints = 100
output_path = "data_1000_points.h5"
batch_size = 10000 # to flush to disk

np.random.seed(6)
bilby.core.utils.log.setup_logger(log_level="WARNING") # tells bilby to not be so chatty

# physics configurations, adopted from ExampleNotebook_DataAnalysis.ipynb
f_min = 20.0              # Hz — detector low-frequency cutoff
sampling_frequency = 2048 # Hz — time-domain sampling rate
duration = 4.0            # s  — analysis segment length
start_time = 0.0          # arbitrary reference

# prior ranges for the parameters we vary
chirp_mass_min = 25     # solar masses
chirp_mass_max = 35     # solar masses
distance_min = 1500      # Mpc
distance_max = 2500     # Mpc

# begin adapted from ExampleNotebook_DataAnalysis.ipynb, among others I added some comments, part of which where made by AI.

# Bilby's helper that builds aligned time/frequency arrays.
time_frequency_series = bilby.core.series.CoupledTimeAndFrequencySeries(duration=duration,
                                                                        sampling_frequency=sampling_frequency,
                                                                        start_time=start_time)

frequency_array = time_frequency_series.frequency_array
# there is a minimum frequency below which the detector is not sensitive and we want to ignore them
frequency_mask = (frequency_array>= f_min)
time_array = time_frequency_series.time_array

# end adapted from notebook

# Fixed parameters, values adopted from ExampleNotebook_DataAnalysis.ipynb, commented by AI
FIXED_PARAMETERS = dict(
    a_1=0.0,                # spin magnitude of primary
    a_2=0.0,                # spin magnitude of secondary
    tilt_1=0.0,             # tilt of primary spin
    tilt_2=0.0,             # tilt of secondary spin
    phi_12=0.0,             # azimuthal angle between spins
    phi_jl=0.0,             # azimuthal angle of total angular momentum
    theta_jn=0.0,           # inclination of total angular momentum vs line of sight
    psi=0.0,                # polarization angle
    phase=0.0,              # coalescence phase
    geocent_time=2.0,       # time of coalescence (s, within our 4s segment)
    ra=0.0,                 # right ascension
    dec=0.0,                # declination
)

# begin adapted from ExampleNotebook_DataAnalysis.ipynb. 
# The waveform_generator is a factory: parameters in -> frequency-domain strain out. The ifo is the H1 detector. Both are 
# build once and reused inside the functions.

approximant = "IMRPhenomXP"
waveform_arguments = dict(
    reference_frequency=50.0,
    minimum_frequency=f_min,
    waveform_approximant=approximant
)

# Create the waveform_generator. For our simple example it is quite unnecessary, but it is useful when we want to modify
# which approximant we use, what modes we want to produce and to control similar settings
waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole, # bilby built-in function to generate waveforms
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

# initializing an interferometer - for this exercise we only make 1. Here it is initialized empty
# H1 = Henford's Ligo interferometer
ifo = bilby.gw.detector.InterferometerList(["H1"])[0]
ifo.set_strain_data_from_zero_noise(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=start_time,
)

# end adapted from notebook

# functions
def black_hole_masses(chirp_mass, q = 1):
    """
    Takes the chirp mass and outputs the masses of the two black holes.

    Args:
        chirp_mass (float): Chirp mass.
        q (float): ratio of m2/m1, default  = 1 (m1 = m2)

    Returns:
        float: m1 and m2
    """
    m1 = chirp_mass * (1+q)**.2 / q**.6
    m2 = m1*q

    return m1, m2

# begin adapted from notebook

def generate_signal(params):
    """
    Generate the time-domain signal in H1 for one set of source parameters.

    Args:
        params (dict): parameter dict.

    Returns:
        numpy.ndarray: real-valued time-domain strain
    """
    # reset the interferometer to zero strain before injecting
    ifo.set_strain_data_from_zero_noise(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=start_time,
    )
    # project the waveform onto H1 and add it to the strain
    ifo.inject_signal(waveform_generator=waveform_generator, parameters=params)

    signal_fd = ifo.strain_data.frequency_domain_strain
    signal_td = bilby.utils.infft(signal_fd, sampling_frequency) # convert to dime domain
    return signal_td

def generate_noise():
    """
    Generate one independent noise realization from H1's PSD, in the time domain.

    Returns:
        numpy.ndarray: real-valued time-domain noise
    """
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=start_time,
    )
    noise_fd = ifo.strain_data.frequency_domain_strain
    noise_td = bilby.utils.infft(noise_fd, sampling_frequency) # convert to dime domain
    return noise_td

# end adapt notebook

def generate_data(signal, noise):
    """
    Adds signal and noise. Kept as a function for clarity in the main loop.

    Args:
        signal (numpy.ndarray): time-domain signal
        noise (numpy.ndarray): time-domain noise

    Returns:
        numpy.ndarray: signal + noise
    """
    return signal + noise

# generate injection parameters

# draw chirp masses and distances uniformly from the prior ranges
chirp_masses = np.random.uniform(chirp_mass_min, chirp_mass_max, n_datapoints)
distances = np.random.uniform(distance_min, distance_max, n_datapoints)

# synthesizes and saves the data
# written in batches to not have managable ram usage

n_time = len(time_array)

# pre-allocate buffers, one batch worth, in RAM
buf_signal = np.empty((batch_size, n_time), dtype=np.float32)
buf_noise = np.empty((batch_size, n_time), dtype=np.float32)
buf_data = np.empty((batch_size, n_time), dtype=np.float32)

# begin adapted code from AI

with h5py.File(output_path, "w") as f:
    # pre-allocate the full datasets on disk
    f.create_dataset("chirp_mass", data=chirp_masses.astype(np.float32))
    f.create_dataset("luminosity_distance", data=distances.astype(np.float32))

    # for the big arrays we make them empty first, then fill batch by batch
    d_signal = f.create_dataset("signal", (n_datapoints, n_time), dtype=np.float32)
    d_noise  = f.create_dataset("noise",  (n_datapoints, n_time), dtype=np.float32)
    d_data   = f.create_dataset("data",   (n_datapoints, n_time), dtype=np.float32)

    # this adds metadata to the hfd5 file
    f.attrs["n_datapoints"] = n_datapoints
    f.attrs["sampling_frequency"] = sampling_frequency
    f.attrs["duration"] = duration
    f.attrs["f_min"] = f_min
    f.attrs["mass_ratio"] = 1.0
    f.attrs["chirp_mass_min"] = chirp_mass_min
    f.attrs["chirp_mass_max"] = chirp_mass_max
    f.attrs["distance_min"] = distance_min
    f.attrs["distance_max"] = distance_max

    # main loop, in batchs
    written = 0
    while written < n_datapoints:
        # how many datapoints in this batch (might be smaller for the last one)
        bs = min(batch_size, n_datapoints - written)

        # fill the buffer
        for k in range(bs):
            i = written + k # global index
            m1, m2 = black_hole_masses(chirp_masses[i])
            params = dict(FIXED_PARAMETERS,
                          mass_1=m1,
                          mass_2=m2,
                          luminosity_distance=distances[i])

            buf_signal[k] = generate_signal(params).astype(np.float32)
            buf_noise[k] = generate_noise().astype(np.float32)
            buf_data[k] = generate_data(buf_signal[k], buf_noise[k])

        # flush the buffer to disk
        d_signal[written:written+bs] = buf_signal[:bs]
        d_noise[written:written+bs] = buf_noise[:bs]
        d_data[written:written+bs] = buf_data[:bs]

        written += bs
        print(f"generated and saved {written}/{n_datapoints} datapoints")

print(f"wrote {n_datapoints} datapoints to {output_path}")

# de plots maken, coppied from AI

# sanity check: 10 willekeurige datapunten plotten in 1 figuur
n_plots = 5

# globale y-grenzen bepalen door alle datapunten te scannen
# h5py leest rij voor rij van schijf, dus dit kost nauwelijks RAM
print("bepalen van globale y-grenzen...")
with h5py.File(output_path, "r") as f:
    signal_max = max(np.max(np.abs(f["signal"][i])) for i in range(n_datapoints))
    noise_max  = max(np.max(np.abs(f["noise"][i]))  for i in range(n_datapoints))
    data_max   = max(np.max(np.abs(f["data"][i]))   for i in range(n_datapoints))

signal_ylim = (-1.05 * signal_max, 1.05 * signal_max)
noise_ylim  = (-1.05 * noise_max,  1.05 * noise_max)
data_ylim   = (-1.05 * data_max,   1.05 * data_max)

# willekeurige indices trekken
indices = np.random.choice(n_datapoints, size=min(n_plots, n_datapoints), replace=False)

# 1 figuur met n_plots kolommen en 3 rijen (signal, noise, data)
fig, axes = plt.subplots(3, n_plots, figsize=(3 * n_plots, 7), sharex=True, sharey="row")

with h5py.File(output_path, "r") as f:
    for plot_nr, idx in enumerate(indices):
        col = axes[:, plot_nr]

        for ax, key, ylim, color in zip(
            col,
            ["signal", "noise", "data"],
            [signal_ylim, noise_ylim, data_ylim],
            ["C0", "C1", "C2"],
        ):
            ax.plot(time_array, f[key][idx], color=color, linewidth=0.6)
            ax.set_ylim(ylim)
            ax.grid(alpha=0.3)

        col[0].set_title(
            f"#{idx}\n"
            rf"$\mathcal{{M}}_c = {chirp_masses[idx]:.2f}\, M_\odot$"
            "\n"
            rf"$d_L = {distances[idx]:.0f}$ Mpc",
            fontsize=9,
        )
        col[2].set_xlabel("time [s]")
        col[2].set_xlim(0, duration)

# y-labels alleen op de linker kolom, anders wordt het rommelig
axes[0, 0].set_ylabel("signal")
axes[1, 0].set_ylabel("noise")
axes[2, 0].set_ylabel("data")

plt.tight_layout()
plt.savefig("sanity_check.png", dpi=120)
plt.show()
print("opgeslagen als sanity_check.png")