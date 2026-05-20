# basically the code from Sung Woo for the linear regression SBI.
import numpy as np
import scipy


def poly_generate_parmas(N_dim, param_min=-5.0, param_max=5.0):
    """
    FUNC:
    Generates paramters of a polynomial function up to N_dim power
    ---
    INPUT:
    N_dim (int)             - number of polynomial powers
    param_min (float/array) - the minimum value of each power
    param_max (float/array) - the maximum value of each power
    """
    if type(N_dim) is not int:
        print("N_dim is not an integer.")
        return

    params = scipy.stats.uniform.rvs(
        loc=param_min, scale=param_max - param_min, size=(N_dim)
    )

    return params


def poly_model(params, x):
    """
    FUNC:
    Returns y of polynomial model with given parameters
    """
    # polyval expects highest power first, so reverse theta
    # If theta = [a0, a1, a2], polynomial is a0 + a1 x + a2 x^2
    # polyval coefficients: [a2, a1, a0]
    return np.polyval(params[::-1], x)


def log_likelihood(theta, x, y, sigma):
    """
    Assume:
    y_i = model(theta, x_i) + N(0, sigma^2), with fixed known sigma.
    This is equivalent to using the MSE in the exponent of the likelihood.
    """
    y_model = poly_model(theta, x)
    residuals = y - y_model
    mse = np.mean(residuals**2)  # mean squared error
    N = len(y)
    return -0.5 * N * (np.log(2 * np.pi * sigma**2) + mse / sigma**2)


def log_prior(params, param_max=5.0, param_min=-5.0):
    """
    Simple broad uniform priors for each parameter.
    Adjust ranges as needed.
    """
    if np.any(params < param_min) or np.any(params > param_max):
        return -np.inf  # log(0), impossible
    return 0.0  # log(1)


def log_posterior(theta, x, y, sigma):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, sigma)


import emcee


def sample_mcmc(true_params, x, y_obs, sigma, nwalkers, nsteps, burn_in, N_dim):
    # # Initialize walkers around a guess near the true parameters
    initial_guess = true_params + 0.1 * np.random.randn(nwalkers, N_dim)
    sampler = emcee.EnsembleSampler(
        nwalkers, N_dim, log_posterior, args=(x, y_obs, sigma)
    )

    sampler.run_mcmc(initial_state=initial_guess, nsteps=nsteps, progress=True)

    return sampler.get_chain(discard=burn_in, flat=True)
