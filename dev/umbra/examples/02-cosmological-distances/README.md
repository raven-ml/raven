# `02-cosmological-distances`

Cosmological distance calculations and parameter fitting. First prints a
distance table for the Planck 2018 cosmology, then fits H0 and Omega_m from
synthetic Type Ia supernova distance moduli using gradient descent.

```bash
dune exec dev/umbra/examples/02-cosmological-distances/main.exe
```

## What You'll Learn

- Using preset cosmologies (`Cosmo.planck18`)
- Computing distances (`comoving_distance`, `luminosity_distance`, `angular_diameter_distance`)
- Computing distance modulus and lookback time
- Building differentiable cosmological models with `create_flat_lcdm`
- Fitting cosmological parameters with Rune autodiff and Vega optimizers

## Key Functions

| Function                      | Purpose                                       |
| ----------------------------- | --------------------------------------------- |
| `Cosmo.planck18`              | Planck 2018 flat LCDM preset                  |
| `Cosmo.comoving_distance`     | Line-of-sight comoving distance               |
| `Cosmo.luminosity_distance`   | Luminosity distance at redshift z              |
| `Cosmo.distance_modulus`      | Distance modulus mu = 5 log10(d_L/Mpc) + 25   |
| `Cosmo.lookback_time`         | Time since light was emitted                   |
| `Cosmo.age`                   | Age of the universe at redshift z              |
| `Cosmo.create_flat_lcdm`     | Tensor-parameterized cosmology for autodiff    |
| `Rune.value_and_grads`        | Forward pass + gradient computation            |

## How It Works

The distance modulus forward model uses `Cosmo.distance_modulus`, which
internally integrates E(z) via 16-point Gauss-Legendre quadrature. Since all
operations are Nx tensor ops, gradients flow through the entire pipeline
automatically via Rune.

The optimizer starts from H0=65, Omega_m=0.25 and converges toward the true
values (H0~73, Omega_m~0.3) that generated the synthetic data.

## Try It

1. Change the preset to `Cosmo.wmap9` and compare the distance table.
2. Add `Omega_L` as a free parameter using `create_lcdm` for a non-flat model.
3. Use `Cosmo.z_at_value` to find the redshift where the lookback time is 10 Gyr.

## Next Steps

Continue to [03-blackbody-fitting](../03-blackbody-fitting/) to fit stellar
temperatures from photometry.
