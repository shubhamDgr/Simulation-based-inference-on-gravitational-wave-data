# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: stats
#     language: python
#     name: python3
# ---

import corner
import emcee
import matplotlib.pyplot as plt
# %%
import numpy as np
import scipy

# %% [markdown]
# Going to keep the seed of 6 - our group number (for debugging).

# %%
np.random.seed(seed=6)  # DEBUG

from polynomial_model import poly_generate_parmas, poly_model

# %% [markdown]
# Define variables and simulate the data:

# %%
sigma = 1  # the scale of Gaussian error
N_dim = 5
x = np.linspace(-2, 1, 100)
true_params = poly_generate_parmas(N_dim)
print("True parameters: ", true_params)
y_true = poly_model(true_params, x)
y_obs = y_true + scipy.stats.norm.rvs(0, sigma, len(x))


import polynomial_model
from polynomial_model import sample_mcmc

samples_mcmc = sample_mcmc(
true_params=true_params,
x=x,
y_obs=y_obs,
sigma=sigma,
nwalkers=50,
nsteps=200,
burn_in=100,
N_dim=N_dim,)

print(type(samples_mcmc))
print(samples_mcmc.shape)
print(samples_mcmc)

# %%
import sbi
import torch
from sbi.analysis import pairplot
from sbi.inference import NPE
from sbi.utils import BoxUniform

print(sbi.__version__)

# %%
# Probably a good practice - redefine variables using torch objects
theta = torch.tensor(true_params)
x = torch.linspace(-2, 1, 100)
print(len(x))


# %%
def poly_model(theta, x):
    power = 0
    y = torch.zeros((len(x)))
    for t in theta:
        y += x**power * t
        power += 1
    return y


poly_model(theta, x)

_ = torch.manual_seed(6)


def simulator(theta):
    # Linear Gaussian.
    return poly_model(theta, x) + torch.randn_like(x)


prior = BoxUniform(
    low=-5.0 * torch.ones(N_dim),
    high=5.0 * torch.ones(N_dim),
)

num_simulations = 5000

theta_sim_list = []
x_sim_list = []

for _ in range(num_simulations):
    # sample one parameter vector from prior
    theta_i = prior.sample((1,))[0]  # (3,)
    x_i = simulator(theta_i)  # (N,)

    theta_sim_list.append(theta_i)
    x_sim_list.append(x_i)

theta_sim = torch.stack(theta_sim_list, dim=0)  # (num_simulations, 3)
x_sim = torch.stack(x_sim_list, dim=0)  # (num_simulations, N)

from functools import partial

import matplotlib.pyplot as plt
import torch
from sbi.analysis import pairplot
from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

# %%
from realnvp import build_realnvp

# %% [markdown]
# Now we define some different normalizing flows. RealNVP is implemented in 
# realnvp.py, to save this file from cluttering.

models = {
    "RealNVP": partial(build_realnvp, hidden_features=128, num_transforms=10),
    "MAF": posterior_nn(
        model="maf",
        hidden_features=128,
        num_transforms=10,
        activation=torch.nn.ReLU,
        dropout_probability=0.0,
    ),
    "NSF": posterior_nn(
        model="nsf",
        hidden_features=128,
        num_transforms=10,
        num_bins=16,
        tail_bound=3.0,
    ),
}

# %% 
samples_dict = {}
loss_dict = {}

for name, model in models.items():

    print(f"\nTraining {name}...\n")

    density_estimator_builder = model

    inference = NPE(
        prior=prior,
        density_estimator=density_estimator_builder,
    )

    density_estimator = inference.append_simulations(theta_sim, x_sim).train()

    posterior = inference.build_posterior(density_estimator)

    # posterior samples
    samples = posterior.sample(
        (5000,),
        x=y_obs,
    )

    samples_dict[name] = samples.detach()

    # training losses
    loss_dict[name] = inference

# %% [markdown]
# After training we can obtain the loss plot.

#%%
fig_loss, ax_loss = plt.subplots()
for key, val in loss_dict.items():
    ax_loss.plot(
        np.arange(len(val.summary["validation_loss"])),
        val.summary["training_loss"],
        label=key,
    )
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Training loss")
ax_loss.legend()
plt.show()
plt.close(fig_loss)

# %% [markdown]
# We now generate a corner plot comparing MCMC with the different normalizing flows.

# %%
import corner

samples_dict["MCMC"] = torch.tensor(samples_mcmc)
colors = ["C0", "C1", "C2", "C3"]
labels = [rf"$\theta_{i}$" for i in range(theta_sim.shape[1])]

fig = None
for (name, samples), color in zip(samples_dict.items(), colors):
    fig = corner.corner(
        samples.numpy(),
        labels=labels,
        color=color,
        fig=fig,  # overlay on the same figure each iteration
        bins=30,
        smooth=1.0,
        plot_datapoints=False,
        plot_density=False,
        fill_contours=True,
        levels=(0.68, 0.95),
    )

corner.overplot_points(
    fig,
    np.array(true_params).reshape(1, -1),
    marker="*",
    color="black",
    markersize=10,
)

handles = [
    plt.Line2D([0], [0], color=color, label=name)
    for color, name in zip(colors, samples_dict.keys())
]

fig.legend(handles=handles, loc="upper right", fontsize=12)
plt.suptitle("Posterior comparison")
plt.show()
