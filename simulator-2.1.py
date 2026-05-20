# May 2026
# This python script is made by Angel Valer Melchor and is intended to generate the data that will be used in the SBI 
# project as part of the ML4PA course

import bilby
import numpy as np
import matplotlib.pyplot as plt
import h5py

# begin configurations

# script configurations
n_datapoints = 200
output_path = input("What do you want to name the datafile (don't add extention)") + ".h5"

np.random.seed(6)
bilby.core.utils.log.setup_logger(log_level="WARNING") # tells bilby to not be so chatty

# physics configurations, adopted from ExampleNotebook_DataAnalysis.ipynb
f_min = 20.0              # Hz — detector low-frequency cutoff
sampling_frequency = 2048 # Hz — time-domain sampling rate
duration = 2            # s  — analysis segment length
start_time = 0          # arbitrary reference

# prior ranges for the parameters we vary
chirp_mass_min = 30     # solar masses
chirp_mass_max = 60     # solar masses
distance_min = 400      # Mpc
distance_max = 800      # Mpc

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
    geocent_time=1.3,       # time of coalescence (s, within our 4s segment)
    ra=0.0,                 # right ascension
    dec=0.0,                # declination
)

# begin adapted from ExampleNotebook_DataAnalysis.ipynb. 
# The waveform_generator is a factory: parameters in -> frequency-domain strain out. The ifos is the H1 detector list. Both are 
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
# we keep it as a list (no [0]) because set_strain_data_from_power_spectral_densities lives on the InterferometerList
ifos = bilby.gw.detector.InterferometerList(["H1"])
ifos.set_strain_data_from_zero_noise(
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
    ifos.set_strain_data_from_zero_noise(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=start_time,
    )
    # project the waveform onto H1 and add it to the strain
    ifos.inject_signal(waveform_generator=waveform_generator, parameters=params)

    signal_fd = ifos[0].strain_data.frequency_domain_strain
    signal_td = bilby.utils.infft(signal_fd, sampling_frequency) # convert to time domain
    return signal_td

def generate_noise():
    """
    Generate one independent noise realization from H1's PSD, in the time domain.

    Returns:
        numpy.ndarray: real-valued time-domain noise
    """
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=start_time,
    )
    noise_fd = ifos[0].strain_data.frequency_domain_strain
    noise_td = bilby.utils.infft(noise_fd, sampling_frequency) # convert to time domain
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

n_time = len(time_array)

with h5py.File(output_path, "w") as f:
    # pre-allocate the full datasets on disk
    f.create_dataset("chirp_mass", data=chirp_masses.astype(np.float32))
    f.create_dataset("luminosity_distance", data=distances.astype(np.float32))

    # for the big arrays we make them empty first, then fill row by row
    d_signal = f.create_dataset("signal", (n_datapoints, n_time), dtype=np.float32)
    d_noise  = f.create_dataset("noise",  (n_datapoints, n_time), dtype=np.float32)
    d_data   = f.create_dataset("data",   (n_datapoints, n_time), dtype=np.float32)

    # main loop
    for i in range(n_datapoints):
        m1, m2 = black_hole_masses(chirp_masses[i])
        params = dict(FIXED_PARAMETERS,
                      mass_1=m1,
                      mass_2=m2,
                      luminosity_distance=distances[i])

        signal = generate_signal(params).astype(np.float32)
        noise = generate_noise().astype(np.float32)
        data = generate_data(signal, noise)

        d_signal[i] = signal
        d_noise[i] = noise
        d_data[i] = data

        if (i + 1) % 100 == 0:
            print(f"generated and saved {i+1}/{n_datapoints} datapoints")

print(f"wrote {n_datapoints} datapoints to {output_path}")