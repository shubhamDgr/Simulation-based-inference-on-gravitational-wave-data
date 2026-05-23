"""
Wave Parameter Recovery with SBI — Class Implementation
===================================================================

ADAPTED: uses GW training data from an HDF5 file (tabular SBI).

Usage:
    study = WaveSBI(seed=42, n_posterior=10000)
    study.run_all()
"""

import corner
import emcee
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.patches import Patch
from sbi import utils as sbi_utils
from sbi.inference import NPE, simulate_for_sbi
from sbi.neural_nets import posterior_nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class LinearMLPEmbedding(nn.Module):
    """Embedding network: linear compression + MLP, used as summary extractor
    before the SBI density estimator.
    """

    def __init__(
        self,
        n_points: int,
        ncomponents: int,
        hidden_dims: list = [64, 64],
        mlp_out_dim: int = 16,
    ):
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
    h5_path = "data.h5"

    def __init__(
        self,
        # n_simulations not used: trainingset size = number of rows in the .h5 file
        seed=42,
        n_posterior=10000,
        ncomponents=10,
        hidden_dims=[64, 64],
        mlp_out_dim=16,
    ):
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

        # Load GW training data once
        with h5py.File(self.h5_path, "r") as f:
            self.h5_data = f["data"][:].astype(np.float32)
            self.h5_chirp_masses = f["chirp_mass"][:].astype(np.float32)
            self.h5_distances = f["luminosity_distance"][:].astype(np.float32)

        self.data_scale = float(np.std(self.h5_data))
        self.h5_data = self.h5_data / self.data_scale
        self.h5_n_points = self.h5_data.shape[0]
        self.n_points = self.h5_data.shape[1]
        self.next_index = 0

        # set by _generate_observed_data
        self.true_chirp_mass = None
        self.true_distance = None

        # Get the first observed data (also sets the true_* attributes)
        self.wave_observed = self._generate_observed_data()

        # Summary network
        self.embedding_net = LinearMLPEmbedding(
            n_points=self.n_points,
            ncomponents=ncomponents,
            hidden_dims=hidden_dims,
            mlp_out_dim=mlp_out_dim,
        ).to(device)

        # Results storage
        self.sbi_samples_np = None
        self.chirp_mass_sbi = None
        self.distance_sbi = None
        self.posterior = None

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
        self.true_distance = float(self.h5_distances[idx])
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
            print(
                f"  Trainingset: {self.h5_n_points} GW datapoints from {self.h5_path}"
            )
            print(f"  Each datapoint: {self.n_points} time samples")
            print(
                f"  Summary net: Linear({self.n_points}→{self.ncomponents})"
                f" + ReLU + MLP{self.hidden_dims}→{self.mlp_out_dim}"
            )
            print("=" * 60)

        # Step 1: prior — must match the prior ranges used when generating the .h5 file
        prior = sbi_utils.BoxUniform(
            low=torch.tensor(
                [
                    float(self.h5_chirp_masses.min()),
                    float(self.h5_distances.min()),
                ],
                device=device,
            ),
            high=torch.tensor(
                [
                    float(self.h5_chirp_masses.max()),
                    float(self.h5_distances.max()),
                ],
                device=device,
            ),
        )

        # Step 2: convert the loaded GW data to torch tensors
        # theta_sim: (N, 2)  — the parameters (chirp_mass, distance) for each row
        # x_sim:     (N, n_points)  — the corresponding time-domain data
        theta_sim = torch.tensor(
            np.stack([self.h5_chirp_masses, self.h5_distances], axis=1),
            dtype=torch.float32,
        )
        x_sim = torch.tensor(self.h5_data, dtype=torch.float32)

        if verbose:
            print(f"\nLoaded training tensors:")
            print(f"  theta_sim : {theta_sim.shape}")
            print(f"  x_sim     : {x_sim.shape}")

        # debug: weights before training
        before = self.embedding_net.linear.weight.detach().clone()
        print(
            "Linear layer weights before training (norm):",
            torch.norm(before).item(),
        )

        # Step 3: train NPE
        if verbose:
            print("\nTraining NPE …")

        neural_posterior = posterior_nn(
            model="nsf",
            embedding_net=self.embedding_net,
        )

        inference = NPE(prior=prior, density_estimator=neural_posterior, device=device)
        density_estimator = inference.append_simulations(
            theta_sim, x_sim,
        data_device=device
        ).train(
            training_batch_size=512,
            learning_rate=1e-4,
            show_train_summary=verbose,
        )
        # debug: weights after training
        after = self.embedding_net.linear.weight.detach().clone()
        print(
            "Linear layer weights after training (norm):",
            torch.norm(after).item(),
        )
        print(
            "Weight change (norm of difference):",
            torch.norm(after - before).item(),
        )

        # Step 4: build posterior and sample on the currently selected observed datapoint
        self.posterior = inference.build_posterior(density_estimator)

        x_obs_torch = torch.tensor(
            self.wave_observed, dtype=torch.float32, device=device
        )
        sbi_samples = self.posterior.sample(
            (self.n_posterior,),
            x=x_obs_torch,
        )
        self.sbi_samples_np = sbi_samples.cpu().numpy()
        self.chirp_mass_sbi = np.median(self.sbi_samples_np[:, 0])
        self.distance_sbi = np.median(self.sbi_samples_np[:, 1])

        if verbose:
            print(
                f"\nSBI/NPE: chirp_mass = {self.chirp_mass_sbi:.3f},  distance = {self.distance_sbi:.3f}"
            )
            print(
                f"  (true: {self.true_chirp_mass:.3f}, {self.true_distance:.3f})\n"
            )

    # ---- Plotting -----------------------------------------------

    def plot_comparison(self, figsize=(15, 4)):
        print("\nPlotting …")

        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle("SBI (NPE) — GW Parameter Recovery", fontsize=14)

        # (a) Observed data
        ax = axes[0]
        ax.plot(self.wave_observed, color="tomato", lw=0.7)
        ax.set(
            xlabel="time sample",
            ylabel="strain",
            title=f"Observed: true Mc={self.true_chirp_mass:.2f}, dL={self.true_distance:.0f}",
        )

        # (b) Marginal posterior for chirp_mass
        ax = axes[1]
        ax.hist(
            self.sbi_samples_np[:, 0],
            bins=40,
            density=True,
            color="darkorange",
            alpha=0.7,
        )
        ax.axvline(
            self.true_chirp_mass,
            color="black",
            lw=2,
            ls="--",
            label=f"True Mc={self.true_chirp_mass:.2f}",
        )
        ax.axvline(
            self.chirp_mass_sbi,
            color="tomato",
            lw=2,
            label=f"Median={self.chirp_mass_sbi:.2f}",
        )
        lo, hi = np.percentile(self.sbi_samples_np[:, 0], [16, 84])
        ax.axvspan(lo, hi, alpha=0.2, color="tomato", label="68% CI")
        ax.set(
            xlabel="chirp mass [M_sun]",
            ylabel="Density",
            title="P(chirp_mass | data)",
        )
        ax.legend(fontsize=8)

        # (c) Marginal posterior for distance
        ax = axes[2]
        ax.hist(
            self.sbi_samples_np[:, 1],
            bins=40,
            density=True,
            color="darkorange",
            alpha=0.7,
        )
        ax.axvline(
            self.true_distance,
            color="black",
            lw=2,
            ls="--",
            label=f"True dL={self.true_distance:.0f}",
        )
        ax.axvline(
            self.distance_sbi,
            color="tomato",
            lw=2,
            label=f"Median={self.distance_sbi:.0f}",
        )
        lo, hi = np.percentile(self.sbi_samples_np[:, 1], [16, 84])
        ax.axvspan(lo, hi, alpha=0.2, color="tomato", label="68% CI")
        ax.set(
            xlabel="distance [Mpc]",
            ylabel="Density",
            title="P(distance | data)",
        )
        ax.legend(fontsize=8)

        plt.tight_layout()
        # plt.show()
        plt.savefig("posterior.png")

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
        #plt.show()
        plt.savefig("joint_posterior.png")

        print("\n── Summary ─────────────────────────────────────────────")
        print(f"{'':10} {'Mc':>8} {'dL':>8}")
        print(
            f"{'True':10} {self.true_chirp_mass:>8.3f} {self.true_distance:>8.3f}"
        )
        print(
            f"{'SBI/NPE':10} {self.chirp_mass_sbi:>8.3f} {self.distance_sbi:>8.3f}"
        )

    def run_all(self, verbose=True):
        self.run_sbi(verbose=verbose)
        self.plot_comparison()


# begin AI
# Run the study with GW data from data.h5

study = WaveSBI(
    seed=42,
    n_posterior=10000,
    ncomponents=128,
    hidden_dims=[256, 256],
    mlp_out_dim=64,
)
study.run_all()
# eind AI
