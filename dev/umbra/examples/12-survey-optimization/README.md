# `12-survey-optimization`

Differentiable survey optimization for a Stage IV weak lensing survey. Uses
exact autodiff gradients to optimize survey parameters that minimize the
uncertainty on S8 = sigma8 * sqrt(Omega_m / 0.3), replacing traditional grid
search with gradient-based Fisher forecasting. Demonstrates both a single-bin
area/depth tradeoff and joint optimization of sky fraction with tomographic bin
edges.

```bash
cd dev/umbra
dune exec --root . examples/12-survey-optimization/main.exe
```

## What You'll Learn

- Computing differentiable Fisher information matrices for survey forecasting
- Optimizing the area/depth tradeoff for sky coverage vs galaxy density
- Jointly optimizing sky fraction and tomographic bin edges with gradient descent
- Using sigmoid-windowed bins for smooth gradient flow through discrete boundaries
- Comparing gradient-based optimization against brute-force grid search

## Key Functions

| Function                    | Purpose                                              |
| --------------------------- | ---------------------------------------------------- |
| `Survey.angular_cl`         | Compute angular power spectra for tracer pairs       |
| `Survey.weak_lensing`       | Create a weak lensing tracer from n(z)               |
| `Survey.smail`              | Smail redshift distribution for source galaxies      |
| `Cosmo.planck18`            | Planck 2018 fiducial cosmology                       |
| `Cosmo.linear_power`        | Linear matter power spectrum P(k, z)                 |
| `Cosmo.comoving_distance`   | Comoving distance for lensing kernel computation     |
| `Rune.value_and_grad`       | Loss and gradient for survey parameter optimization  |
| `Vega.adam`                 | Adam optimizer for continuous parameter search       |

## How It Works

Part 1 tackles the area/depth tradeoff for a single tomographic bin. A fixed
galaxy budget (n_gal * f_sky = constant) means wider surveys are shallower. The
Fisher matrix for [Omega_m, sigma8] is computed from Limber-integrated angular
power spectra, with shape noise that depends on galaxy density. The objective
function -- sigma(S8) derived from the 2x2 Fisher inverse -- is fully
differentiable through f_sky via sigmoid parameterization. Adam finds the
optimal sky fraction in 300 steps with exact gradients, verified by a
finite-difference check.

Part 2 extends to joint optimization of sky fraction and two tomographic bin
edges that divide galaxies into three redshift bins. The bin boundaries use
sigmoid window functions (with width delta=0.03) so that gradients flow smoothly
through the discrete bin assignment. Narrower bins concentrate signal but
increase shot noise; the optimizer balances this tradeoff automatically. The
Limber integral uses precomputed cosmological grids (comoving distances, Hubble
rates, power spectra) evaluated at five cosmology perturbations for numerical
derivatives of C_l with respect to Omega_m and sigma8, while gradients with
respect to survey parameters (f_sky, z1, z2) flow through Rune's autodiff.

A brute-force grid search over 12 x 15 x 15 = 2700 parameter combinations
validates the gradient result, demonstrating that 500 Adam steps achieve equal
or better precision with orders of magnitude fewer function evaluations.

## Try It

1. Increase the galaxy budget from 10 to 50 gal/arcmin2 and observe how the
   optimal sky fraction shifts toward wider coverage.
2. Add a fourth tomographic bin and compare the improvement in sigma(S8).
3. Replace the Smail n(z) with a sharper distribution and see how the optimal
   bin edges respond.

## Next Steps

This is the final example in the Umbra series. For earlier topics, revisit
[01-constants-and-units](../01-constants-and-units/) for physical constants and
unit handling, or [05-sed-fitting](../05-sed-fitting/) for the foundations of
differentiable spectral energy distribution fitting that this example builds on.
