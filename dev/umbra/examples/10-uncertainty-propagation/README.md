# `10-uncertainty-propagation`

Automatic uncertainty propagation through cosmological distance calculations.
Propagates H0 and Omega_m uncertainties through distance modulus using exact
AD Jacobians via forward-mode differentiation. The linear error propagation
formula (Sigma_out = J Sigma_in J^T) is validated against Monte Carlo sampling
with 50,000 draws.

```bash
cd dev/umbra
dune exec --root . examples/10-uncertainty-propagation/main.exe
```

## What You'll Learn

- Computing exact Jacobians automatically with forward-mode AD (`Rune.jacfwd`)
- Applying linear error propagation via the Jacobian covariance formula
- Validating analytical uncertainty estimates with Monte Carlo sampling
- Propagating scalar uncertainties through cosmological models with JVP

## Key Functions

| Function                     | Purpose                                          |
| ---------------------------- | ------------------------------------------------ |
| `Cosmo.create_flat_lcdm`     | Create a flat Lambda-CDM cosmology               |
| `Cosmo.distance_modulus`     | Compute distance modulus at a given redshift      |
| `Rune.jacfwd`                | Forward-mode Jacobian of a function               |
| `Rune.jvp`                   | Jacobian-vector product for scalar propagation    |
| `Nx.cholesky`                | Cholesky decomposition for MC sampling            |
| `Nx.matmul`                  | Matrix multiply for J Sigma J^T                  |
| `Nx.diag`                    | Build diagonal covariance from variances          |

## How It Works

Given input parameters with uncertainties (H0 = 70 +/- 1 km/s/Mpc, Omega_m =
0.30 +/- 0.01), the example propagates these through `Cosmo.distance_modulus`
at five redshifts (z = 0.1 to 1.0). The propagation uses the standard linear
formula: Sigma_out = J Sigma_in J^T, where J is the Jacobian of the distance
modulus with respect to [H0, Omega_m]. Rather than deriving J analytically,
`Rune.jacfwd` computes it automatically with just two JVP evaluations (one per
input parameter).

For validation, the example draws 50,000 Monte Carlo samples from the input
covariance via Cholesky decomposition, evaluates the model at each sample, and
computes empirical output statistics. Agreement below 1% between AD and MC
confirms that linear propagation is accurate for these parameter ranges.

A scalar API demo shows the simpler case: propagating redshift uncertainty
(z = 0.5 +/- 0.01) through a single `jvp` call, which returns both the output
value and its sensitivity to the input perturbation.

## Try It

1. Add correlation between H0 and Omega_m by putting off-diagonal terms in the
   input covariance matrix.
2. Increase the uncertainties to see where linear propagation breaks down and
   MC diverges from AD.
3. Propagate uncertainties through `Cosmo.luminosity_distance` instead of
   distance modulus and compare the relative errors.

## Next Steps

Continue to [11-bayesian-sed](../11-bayesian-sed/) to see how Fisher information
and Hamiltonian Monte Carlo provide both theoretical bounds and full Bayesian
posteriors for SED parameter estimation.
