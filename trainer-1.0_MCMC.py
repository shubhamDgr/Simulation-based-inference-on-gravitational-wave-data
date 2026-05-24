"""
Wave Parameter Recovery with SBI — Class Implementation
===================================================================

ADAPTED: uses GW training data from an HDF5 file (tabular SBI).

Usage:
    study = WaveSBI(seed=42, n_posterior=10000)
    study.run_all()
"""

import bilby
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import emcee, corner
from sbi import utils as sbi_utils
from sbi.inference import NPE, simulate_for_sbi
from sbi.neural_nets import posterior_nn
from matplotlib.patches import Patch

# Importing necessary parts from simulator.py

f_min = 20.0
sampling_frequency = 2048
duration = 1.0
start_time = 0.0

FIXED_PARAMETERS = dict(
    a_1=0.0, a_2=0.0,
    tilt_1=0.0, tilt_2=0.0,
    phi_12=0.0, phi_jl=0.0,
    theta_jn=0.0,
    psi=0.0,
    phase=0.0,
    geocent_time=.95,
    ra=0.0,
    dec=0.0,
)

approximant = "IMRPhenomXP"
waveform_arguments = dict(
    reference_frequency=50.0,
    minimum_frequency=f_min,
    waveform_approximant=approximant,
)

waveform_generator = bilby.gw.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
    parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
    waveform_arguments=waveform_arguments,
)

ifos = bilby.gw.detector.InterferometerList(["H1"])
ifos.set_strain_data_from_zero_noise(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=start_time,
)

def black_hole_masses(chirp_mass, q=1.0):
    m1 = chirp_mass * (1 + q)**0.2 / q**0.6
    m2 = m1 * q
    return m1, m2

def generate_signal(params):
    ifos.set_strain_data_from_zero_noise(
        sampling_frequency=sampling_frequency,
        duration=duration,
        start_time=start_time,
    )
    ifos.inject_signal(waveform_generator=waveform_generator, parameters=params)
    signal_fd = ifos[0].strain_data.frequency_domain_strain
    signal_td = bilby.utils.infft(signal_fd, sampling_frequency)
    return np.real(signal_td)

class LinearMLPEmbedding(nn.Module):
    """Embedding network: linear compression + MLP, used as summary extractor
    before the SBI density estimator.
    """

    def __init__(self, n_points: int, ncomponents: int, hidden_dims: list = [64, 64], mlp_out_dim: int = 16):
        super().__init__()
        self.linear = nn.Linear(n_points, ncomponents)

        layers = []
        in_dim = ncomponents
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, mlp_out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = self.mlp(x)
        return x


class WaveSBI:
    """
    SBI/NPE for GW parameter recovery (chirp mass, luminosity distance).

    Training data is loaded once from data.h5 in __init__.
    Each call to _generate_observed_data() returns the next datapoint
    from the file (cycles around at the end).
    """

    # path to the training data file
    h5_path = "C:/Users/myswl/OneDrive - UvA/Uni/S2/P5/5354MLFP6/project/data-200k.h5"

    def __init__(self,
                 # n_simulations not used: trainingset size = number of rows in the .h5 file
                 seed=42, n_posterior=10000,
                 ncomponents=10, hidden_dims=[64, 64], mlp_out_dim=16):
        """
        Initialize the GW Parameter Recovery study.

        Parameters
        ----------
        seed : int
            Random seed for reproducibility
        n_posterior : int
            Number of posterior samples to draw
        ncomponents : int
            Output components of the linear compression layer
        hidden_dims : list
            Hidden layer dimensions of the MLP
        mlp_out_dim : int
            Final output dim of the embedding net
        """
        self.seed = seed
        self.n_posterior = n_posterior
        self.ncomponents = ncomponents
        self.hidden_dims = hidden_dims
        self.mlp_out_dim = mlp_out_dim

        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # --- NEW: store sampling settings from the simulator ---
        self.sampling_frequency = 2048.0  # Hz
        self.duration = 1.0              # s
        self.dt = 1.0 / self.sampling_frequency

        # Load GW training data once
        with h5py.File(self.h5_path, "r") as f:
            self.h5_data         = f["data"][:].astype(np.float32)
            self.h5_chirp_masses = f["chirp_mass"][:].astype(np.float32)
            self.h5_distances    = f["luminosity_distance"][:].astype(np.float32)
            # --- NEW: load noise-only data for PSD estimation ---
            self.h5_noise        = f["noise"][:].astype(np.float32)

        self.data_scale = float(np.std(self.h5_data))
        self.h5_data = self.h5_data / self.data_scale
        self.h5_noise = self.h5_noise / self.data_scale

        self.h5_n_points = self.h5_data.shape[0]
        self.n_points    = self.h5_data.shape[1]
        self.next_index  = 0

        # --- NEW: build frequency grid and PSD from noise ---
        # Real FFT frequency axis
        self.freqs = np.fft.rfftfreq(self.n_points, d=self.dt)

        # FFT of noise along time axis
        noise_fft = np.fft.rfft(self.h5_noise, axis=1)  # shape (N, n_freqs)

        # Empirical PSD: average |noise_fft|^2 over many noise realizations
        # Normalization constants don't matter for MCMC (they add constants to logL)
        self.psd = np.mean(np.abs(noise_fft)**2, axis=0)

        # Avoid zeros in PSD to prevent division issues
        positive = self.psd > 0
        self.psd[~positive] = np.min(self.psd[positive])

        # Optional: low-frequency cut (same as simulator f_min)
        self.f_min = 20.0
        self.freq_mask = self.freqs >= self.f_min

        # set by _generate_observed_data
        self.true_chirp_mass = None
        self.true_distance = None

        # Get the first observed data (also sets the true_* attributes)
        self.wave_observed = self._generate_observed_data()

        # Summary network
        self.embedding_net = LinearMLPEmbedding(
            n_points    = self.n_points,
            ncomponents = ncomponents,
            hidden_dims = hidden_dims,
            mlp_out_dim = mlp_out_dim,
        )

        # Results storage
        self.sbi_samples_np = None
        self.chirp_mass_sbi = None
        self.distance_sbi = None
        self.posterior = None

        # --- NEW: placeholders for MCMC ---
        self.mcmc_samples = None
        self.chirp_mass_mcmc = None
        self.distance_mcmc = None

    def _generate_observed_data(self):
        """
        Returns the next GW datapoint from the loaded HDF5 file.

        Cycles through the file: after returning the last datapoint,
        the next call returns the first one again. Also updates
        self.true_chirp_mass and self.true_distance so the plotting
        functions reflect the current datapoint.
        """
        idx = self.next_index
        data = self.h5_data[idx]
        self.true_chirp_mass = float(self.h5_chirp_masses[idx])
        self.true_distance   = float(self.h5_distances[idx])
        self.next_index = (self.next_index + 1) % self.h5_n_points
        return data

    # SBI / NPE

    def run_sbi(self, verbose=True):
        """
        Train NPE on the GW data loaded from the .h5 file (tabular SBI).
        No on-the-fly simulator is called: the file IS the training set.
        """
        if verbose:
            print("=" * 60)
            print("SBI — Neural Posterior Estimation (NPE)")
            print(f"  Trainingset: {self.h5_n_points} GW datapoints from {self.h5_path}")
            print(f"  Each datapoint: {self.n_points} time samples")
            print(f"  Summary net: Linear({self.n_points}→{self.ncomponents})"
                  f" + ReLU + MLP{self.hidden_dims}→{self.mlp_out_dim}")
            print("=" * 60)

        # Step 1: prior — must match the prior ranges used when generating the .h5 file
        prior = sbi_utils.BoxUniform(
            low  = torch.tensor([float(self.h5_chirp_masses.min()),
                                 float(self.h5_distances.min())]),
            high = torch.tensor([float(self.h5_chirp_masses.max()),
                                 float(self.h5_distances.max())])
        )

        # Step 2: convert the loaded GW data to torch tensors
        # theta_sim: (N, 2)  — the parameters (chirp_mass, distance) for each row
        # x_sim:     (N, n_points)  — the corresponding time-domain data
        theta_sim = torch.tensor(
            np.stack([self.h5_chirp_masses, self.h5_distances], axis=1),
            dtype=torch.float32
        )
        x_sim = torch.tensor(self.h5_data, dtype=torch.float32)

        if verbose:
            print(f"\nLoaded training tensors:")
            print(f"  theta_sim : {theta_sim.shape}")
            print(f"  x_sim     : {x_sim.shape}")

        # debug: weights before training
        before = self.embedding_net.linear.weight.detach().clone()
        print("Linear layer weights before training (norm):", torch.norm(before).item())

        # Step 3: train NPE
        if verbose:
            print("\nTraining NPE …")

        neural_posterior = posterior_nn(
            model="nsf",
            embedding_net=self.embedding_net,
        )

        inference = NPE(prior=prior, density_estimator=neural_posterior)
        density_estimator = (inference
                             .append_simulations(theta_sim, x_sim)
                             .train(show_train_summary=verbose))

        # debug: weights after training
        after = self.embedding_net.linear.weight.detach().clone()
        print("Linear layer weights after training (norm):", torch.norm(after).item())
        print("Weight change (norm of difference):", torch.norm(after - before).item())

        # Step 4: build posterior and sample on the currently selected observed datapoint
        self.posterior = inference.build_posterior(density_estimator)

        x_obs_torch = torch.tensor(self.wave_observed, dtype=torch.float32)
        sbi_samples = self.posterior.sample(
            (self.n_posterior,),
            x = x_obs_torch,
        )
        self.sbi_samples_np = sbi_samples.numpy()
        self.chirp_mass_sbi = np.median(self.sbi_samples_np[:, 0])
        self.distance_sbi   = np.median(self.sbi_samples_np[:, 1])

        if verbose:
            print(f"\nSBI/NPE: chirp_mass = {self.chirp_mass_sbi:.3f},  distance = {self.distance_sbi:.3f}")
            print(f"  (true: {self.true_chirp_mass:.3f}, {self.true_distance:.3f})\n")

    #Likelihood-based inference / MCMC

    # --- NEW: prior for MCMC ---------------------------------------------
    def log_prior(self, theta):
        """
        Uniform prior matching the SBI BoxUniform:
        theta = [chirp_mass, distance].
        """
        mc, dL = theta

        mc_min = float(self.h5_chirp_masses.min())
        mc_max = float(self.h5_chirp_masses.max())
        dL_min = float(self.h5_distances.min())
        dL_max = float(self.h5_distances.max())

        if (mc_min <= mc <= mc_max) and (dL_min <= dL <= dL_max):
            # uniform in the box (log-constant)
            return 0.0
        else:
            return -np.inf

    # --- NEW: waveform simulator using bilby ------------------------------
    def simulate_waveform(self, theta):
        """
        Generate a *noise-free* model waveform for given parameters.
        Uses the bilby-based generator from your simulation script.

        theta: array-like, [chirp_mass, distance]

        Returns:
            model (np.ndarray): 1D, length self.n_points, normalized like h5_data.
        """
        mc, dL = theta
        m1, m2 = black_hole_masses(mc)

        params = dict(FIXED_PARAMETERS,
                      mass_1=m1,
                      mass_2=m2,
                      luminosity_distance=dL)

        signal = generate_signal(params)  # unnormalized, length ≈ n_points

        # Ensure correct length (trim or pad if needed)
        signal = np.asarray(signal, dtype=np.float32)
        if signal.shape[0] != self.n_points:
            # Simple handling: trim or pad with zeros
            if signal.shape[0] > self.n_points:
                signal = signal[:self.n_points]
            else:
                pad = self.n_points - signal.shape[0]
                signal = np.pad(signal, (0, pad), mode="constant")

        # Apply the same scaling as the training data
        signal_norm = signal / self.data_scale
        return signal_norm

    # --- NEW: frequency-domain PSD-based log-likelihood -----------------
    def log_likelihood_psd(self, theta):
        """
        Frequency-domain likelihood using empirical PSD:

        log L ∝ -0.5 * sum_{f >= f_min} |d~(f) - h~(theta, f)|^2 / PSD(f)

        Both d and h are normalized the same way as the training data.
        """
        x_obs = self.wave_observed  # time-domain, normalized
        x_model = self.simulate_waveform(theta)

        if x_model.shape != x_obs.shape:
            return -np.inf

        # FFT both observed data and model
        dtilde = np.fft.rfft(x_obs)
        htilde = np.fft.rfft(x_model)

        # Residual in frequency domain
        resid = dtilde - htilde

        # Apply frequency mask (f >= f_min)
        mask = self.freq_mask
        resid = resid[mask]
        psd   = self.psd[mask]

        # Compute weighted sum of squared residuals
        # Normalization constants (dt, etc.) only add constants to logL,
        # so we omit them for MCMC.
        chi2 = np.sum(np.abs(resid)**2 / psd)

        return -0.5 * float(chi2)

    # --- NEW: log-posterior for emcee ------------------------------------
    def log_posterior(self, theta):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        ll = self.log_likelihood_psd(theta)
        return lp + ll

    # --- NEW: emcee driver -----------------------------------------------
    def run_mcmc(self, n_walkers=32, n_steps=5000, burn_in=1000, verbose=True):
        """
        Run MCMC with emcee to sample p(theta | wave_observed).

        Stores:
            self.mcmc_samples
            self.chirp_mass_mcmc
            self.distance_mcmc
        """
        print('test')
        ndim = 2  # chirp_mass, distance

        # Initial center: true params if available, else prior mean
        mc0 = self.true_chirp_mass if self.true_chirp_mass is not None else float(np.mean(self.h5_chirp_masses))
        dL0 = self.true_distance   if self.true_distance   is not None else float(np.mean(self.h5_distances))
        theta0 = np.array([mc0, dL0])

        # Initialize walkers in a small Gaussian ball around theta0
        pos0 = theta0 + 1e-2 * np.random.randn(n_walkers, ndim)

        sampler = emcee.EnsembleSampler(n_walkers, ndim, self.log_posterior)

        if verbose:
            print("\nRunning burn-in...")
        pos, prob, state = sampler.run_mcmc(pos0, burn_in, progress=verbose)
        sampler.reset()

        if verbose:
            print("Running main MCMC...")
        sampler.run_mcmc(pos, n_steps, progress=verbose)

        samples = sampler.get_chain(discard=0, flat=True)
        self.mcmc_samples = samples

        self.chirp_mass_mcmc = float(np.median(samples[:, 0]))
        self.distance_mcmc   = float(np.median(samples[:, 1]))

        if verbose:
            print("\nMCMC results (median):")
            print(f"  chirp_mass = {self.chirp_mass_mcmc:.3f}")
            print(f"  distance   = {self.distance_mcmc:.3f}")
            print(f"  true       = {self.true_chirp_mass:.3f}, {self.true_distance:.3f}")

        return samples

    # ---- Plotting -----------------------------------------------

    def plot_comparison(self, figsize=(15, 4)):
        print("\nPlotting …")

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle("SBI (NPE) — GW Parameter Recovery", fontsize=14)

        # (a) Observed data
        ax = axes[0]
        ax.plot(self.wave_observed, color="tomato", lw=0.7)
        ax.set(xlabel="time sample", ylabel="strain",
               title=f"Observed: true Mc={self.true_chirp_mass:.2f}, dL={self.true_distance:.0f}")

        # (b) Marginal posterior for chirp_mass
        ax = axes[1]
        ax.hist(self.sbi_samples_np[:, 0], bins=40, density=True,
                color="darkorange", alpha=0.7)
        ax.axvline(self.true_chirp_mass, color="black", lw=2, ls="--",
                   label=f"True Mc={self.true_chirp_mass:.2f}")
        ax.axvline(self.chirp_mass_sbi,  color="tomato", lw=2,
                   label=f"Median={self.chirp_mass_sbi:.2f}")
        lo, hi = np.percentile(self.sbi_samples_np[:, 0], [16, 84])
        ax.axvspan(lo, hi, alpha=0.2, color="tomato", label="68% CI")
        ax.set(xlabel="chirp mass [M_sun]", ylabel="Density",
               title="P(chirp_mass | data)")
        ax.legend(fontsize=8)

        # (c) Marginal posterior for distance
        ax = axes[2]
        ax.hist(self.sbi_samples_np[:, 1], bins=40, density=True,
                color="darkorange", alpha=0.7)
        ax.axvline(self.true_distance, color="black", lw=2, ls="--",
                   label=f"True dL={self.true_distance:.0f}")
        ax.axvline(self.distance_sbi, color="tomato", lw=2,
                   label=f"Median={self.distance_sbi:.0f}")
        lo, hi = np.percentile(self.sbi_samples_np[:, 1], [16, 84])
        ax.axvspan(lo, hi, alpha=0.2, color="tomato", label="68% CI")
        ax.set(xlabel="distance [Mpc]", ylabel="Density",
               title="P(distance | data)")
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()

        # Corner plot
        fig2 = corner.corner(
            self.sbi_samples_np,
            labels=["chirp_mass [M_sun]", "distance [Mpc]"],
            truths=[self.true_chirp_mass, self.true_distance],
            truth_color="black",
            color="darkorange",
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
        )
        fig2.suptitle("Joint posterior  P(Mc, dL | data)", y=1.02, fontsize=13)
        plt.show()

        print("\n── Summary ─────────────────────────────────────────────")
        print(f"{'':10} {'Mc':>8} {'dL':>8}")
        print(f"{'True':10} {self.true_chirp_mass:>8.3f} {self.true_distance:>8.3f}")
        print(f"{'SBI/NPE':10} {self.chirp_mass_sbi:>8.3f} {self.distance_sbi:>8.3f}")

    def plot_mcmc_results(self, figsize=(15, 4)):
        samples = self.mcmc_samples

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle("MCMC (emcee) — GW Parameter Recovery", fontsize=14)

        # (a) Observed data
        ax = axes[0]
        ax.plot(self.wave_observed, color="steelblue", lw=0.7)
        ax.set(xlabel="time sample", ylabel="strain",
               title=f"Observed: true Mc={self.true_chirp_mass:.2f}, dL={self.true_distance:.0f}")

        # (b) chirp mass
        ax = axes[1]
        ax.hist(samples[:, 0], bins=40, density=True, color="royalblue", alpha=0.7)
        ax.axvline(self.true_chirp_mass, color="black", lw=2, ls="--",
                   label=f"True Mc={self.true_chirp_mass:.2f}")
        ax.axvline(self.chirp_mass_mcmc, color="navy", lw=2,
                   label=f"Median={self.chirp_mass_mcmc:.2f}")
        lo, hi = np.percentile(samples[:, 0], [16, 84])
        ax.axvspan(lo, hi, alpha=0.2, color="navy", label="68% CI")
        ax.set(xlabel="chirp mass [M_sun]", ylabel="Density",
               title="P(chirp_mass | data)")
        ax.legend(fontsize=8)

        # (c) distance
        ax = axes[2]
        ax.hist(samples[:, 1], bins=40, density=True, color="royalblue", alpha=0.7)
        ax.axvline(self.true_distance, color="black", lw=2, ls="--",
                   label=f"True dL={self.true_distance:.0f}")
        ax.axvline(self.distance_mcmc, color="navy", lw=2,
                   label=f"Median={self.distance_mcmc:.0f}")
        lo, hi = np.percentile(samples[:, 1], [16, 84])
        ax.axvspan(lo, hi, alpha=0.2, color="navy", label="68% CI")
        ax.set(xlabel="distance [Mpc]", ylabel="Density",
               title="P(distance | data)")
        ax.legend(fontsize=8)

        plt.tight_layout()
        plt.show()

        # Corner plot
        fig2 = corner.corner(
            samples,
            labels=["chirp_mass [M_sun]", "distance [Mpc]"],
            truths=[self.true_chirp_mass, self.true_distance],
            truth_color="black",
            color="royalblue",
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
        )
        fig2.suptitle("Joint posterior  P(Mc, dL | data)", y=1.02, fontsize=13)
        plt.show()

    def run_all(self, verbose=True):
        self.run_sbi(verbose=verbose)
        self.plot_comparison()
    
    def run_all_mcmc(self, verbose=True):
        self.run_mcmc(verbose=verbose)
        self.plot_mcmc_results()


# begin AI
# Run the study with GW data from data.h5

#print('test')

study = WaveSBI(seed=42, n_posterior=10000, ncomponents=128, hidden_dims=[256, 256], mlp_out_dim=64)
#study.run_all()
# eind AI

# For the MCMC results
study.run_all_mcmc()