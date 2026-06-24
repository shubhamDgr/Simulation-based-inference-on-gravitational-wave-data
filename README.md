# Simulation-Based Inference with Normalizing Flows to predict GW parameters

> **ML Group Project** — University of Amsterdam, 2026  
>  Sung Woo Lee · Beer Meester · Shubham Apte · Angel Valer Melchor

---

## Overview

This project explores **Simulation-Based Inference (SBI)** as an alternative to traditional likelihood-based methods (e.g. MCMC) for Bayesian parameter estimation, with a focus on gravitational-wave (GW) data.

In classical MCMC, you need an analytic likelihood — a formula for how probable your data is given the parameters. For GW detectors, the noise is non-Gaussian and glitchy, making this formula hard to write down. SBI sidesteps this entirely: it only needs a **simulator** that can generate realistic data. A normalizing flow is then trained to approximate the posterior directly from simulated examples.

---

## What We Did

### 1. Summary Extractors — when and why they help
Raw high-dimensional data fed directly into a normalizing flow often leads to poor posteriors. We investigated **summary extractors** `T(x)` that compress the data before inference:

- **Binning**: averaging every *n* neighbouring data points. On a simple linear model `y = mx + c`, binning 30 raw points into 10 bins moved the SBI posterior mean from `(0.236, 1.638)` to `(0.300, 1.529)` — almost exactly the true values `(0.3, 1.5)` — while MCMC stayed stable throughout.

- **PCA compression**: on a sinusoidal toy model with 200-dimensional waveforms, PCA automatically discovered sine- and cosine-like basis functions. Keeping only the first 2 PCA components (which explain ~85% of variance) was enough for SBI to match MCMC closely. Raw 200-dim input produced a broader, biased posterior.

### 2. Neural Embedding (MLP)
A Multilayer Perceptron was used as a trainable summary extractor, reducing 2048-dimensional time-domain GW spectra to a compact embedding before passing to the normalizing flow.

### 3. Gravitational Wave Simulation
We simulated binary black hole mergers using fixed spin and sky-location parameters, varying:
- Chirp mass: 30–60 solar masses
- Luminosity distance: 400–800 Mpc

Simulated waveforms were injected into realistic detector noise for all three LIGO/Virgo detectors.

### 4. Normalizing Flows
We used **Neural Spline Flows** (Durkan et al., 2019) as the posterior estimator. The flow learns a bijective mapping from a simple base distribution to the target posterior, conditioned on the data summary.

### 5. Comparison with MCMC
Results from SBI (NPE) were compared against MCMC (emcee) for both toy models and GW parameter recovery. SBI successfully recovered chirp mass and luminosity distance posteriors consistent with MCMC, at a fraction of the computational cost once trained.

---

## Key Findings

| Setting | MCMC | SBI |
|---|---|---|
| Linear model, raw data | Stable | Slightly biased |
| Linear model, binned data | Stable | Matches MCMC |
| Sinusoidal, raw 200-dim | Stable | Broad, biased |
| Sinusoidal, 2 PCA dims | Stable | Matches MCMC |
| GW parameter recovery | Reference | Consistent recovery |

> **Main lesson**: data representation matters as much as the inference method. A well-chosen summary extractor `T(x)` — whether binning, PCA, or a neural network — is essential for SBI to work well on high-dimensional data.

---



---

## References

- Cranmer, K., Brehmer, J., & Louppe, G. (2020). *The frontier of simulation-based inference.* PNAS.
- Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019). *Neural spline flows.* NeurIPS.
- Foreman-Mackey, D. et al. (2013). *emcee: The MCMC Hammer.* PASP.
