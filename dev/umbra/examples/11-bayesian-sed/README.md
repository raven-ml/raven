# `11-bayesian-sed`

Fisher information matrix analysis and Hamiltonian Monte Carlo sampling for
Bayesian SED parameter estimation. Computes Cramer-Rao bounds (theoretical
minimum uncertainties) from the Fisher matrix, then samples the full posterior
via HMC through the differentiable spectrum -> extinction -> photometry pipeline.

```bash
cd dev/umbra
dune exec --root . examples/11-bayesian-sed/main.exe
```

## What You'll Learn

- Computing the Fisher information matrix via reverse-mode Jacobians
- Deriving Cramer-Rao bounds on SED parameters (temperature, extinction)
- Sampling full Bayesian posteriors with Hamiltonian Monte Carlo
- Comparing Fisher-predicted vs HMC-sampled uncertainties
- Building differentiable forward models through tophat bandpasses

## Key Functions

| Function                   | Purpose                                             |
| -------------------------- | --------------------------------------------------- |
| `Rune.jacrev`              | Reverse-mode Jacobian for Fisher matrix computation |
| `Nx.inv`                   | Matrix inverse for Fisher -> covariance             |
| `Nx.diagonal`              | Extract diagonal (marginal variances)               |
| `Spectrum.blackbody`       | Generate Planck SED at given temperature            |
| `Extinction.apply`         | Apply CCM89 dust reddening                          |
| `Photometry.tophat`        | Create rectangular bandpass filters                 |
| `Photometry.ab_mag`        | Compute AB magnitude through a bandpass             |
| `Norn.hmc`                 | Hamiltonian Monte Carlo posterior sampling           |

## How It Works

The forward model maps two parameters -- log(T) and A_V -- to five broadband
magnitudes through the pipeline: `blackbody -> extinction -> ab_mag`. Synthetic
observations are generated at T=6500 K, A_V=0.5 with realistic photometric
errors (0.03-0.05 mag).

The Fisher information matrix F = J^T C^-1 J is computed from the Jacobian of
the model (via `Rune.jacrev`) and the observational covariance C. Inverting F
gives the Cramer-Rao lower bound -- the best achievable 1-sigma uncertainty on
each parameter for a given dataset, regardless of estimation method.

The example then samples the actual Bayesian posterior using `Norn.hmc`. The
log-posterior is a Gaussian likelihood with flat priors, and HMC uses Rune's
gradients to efficiently explore the parameter space with 500 post-warmup
samples. Comparing the HMC posterior width to the Fisher prediction validates
that the model is well-behaved: when they agree, the posterior is approximately
Gaussian and the Fisher bound is tight.

## Try It

1. Reduce the photometric errors to 0.01 mag and observe how both Fisher bounds
   and HMC posteriors tighten.
2. Add a third parameter (redshift) and examine the resulting parameter
   degeneracies in the Fisher matrix.
3. Replace the flat prior with an informative Gaussian prior on A_V and see
   how the posterior shifts.

## Next Steps

Continue to [12-survey-optimization](../12-survey-optimization/) to see how
differentiable Fisher forecasting enables gradient-based optimization of survey
design parameters for weak gravitational lensing.
