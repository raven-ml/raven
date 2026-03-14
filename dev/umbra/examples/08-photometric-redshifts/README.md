# `08-photometric-redshifts`

Two-stage photometric redshift estimation: coarse grid search followed by
gradient-based refinement using Adam. The full pipeline (blackbody -> redshift
-> extinction -> photometry) is differentiable through Rune, enabling gradient
descent on redshift and normalization parameters against synthetic SDSS ugriz
observations.

```bash
cd dev/umbra
dune exec --root . examples/08-photometric-redshifts/main.exe
```

## What You'll Learn

- Building an end-to-end differentiable photometric pipeline through SDSS ugriz filters
- Composing spectrum redshifting, dust extinction, and synthetic photometry
- Combining grid search initialization with autodiff gradient refinement
- Using multi-parameter gradients to jointly fit redshift and normalization

## Key Functions

| Function                   | Purpose                                              |
| -------------------------- | ---------------------------------------------------- |
| `Spectrum.blackbody`       | Generate a template SED at given temperature         |
| `Spectrum.redshift`        | Apply cosmological redshift to a spectrum             |
| `Spectrum.scale`           | Scale spectrum by a normalization factor              |
| `Extinction.apply`         | Apply dust reddening with CCM89 law                  |
| `Photometry.ab_mag`        | Compute AB magnitude through a bandpass              |
| `Photometry.wavelength`    | Extract the wavelength grid of a bandpass filter      |
| `Rune.value_and_grads`     | Compute loss and parameter gradients in one pass     |
| `Vega.adam`                | Adam optimizer for gradient refinement               |

## How It Works

The example generates synthetic observed magnitudes for a galaxy at z=0.3 with
T=5500 K, A_V=0.2, by pushing a blackbody through the full pipeline:
`blackbody -> scale -> extinction -> redshift -> ab_mag` in each of the five
SDSS bands. These serve as the "data" to fit against.

Stage 1 performs a coarse grid search over 30 redshift values from 0.01 to
0.90, computing chi-squared at each point with a fixed template. This
identifies a rough minimum without requiring gradients.

Stage 2 takes the best grid redshift and refines it with 500 Adam optimizer
steps. The loss function (sum of squared magnitude residuals) flows through
`Spectrum.redshift` and `Photometry.ab_mag`, so Rune provides exact gradients
with respect to log(1+z) and log(normalization). The parameterization in
log-space ensures positivity and improves conditioning.

## Try It

1. Change the true redshift to z=0.7 and observe how the grid search coarseness
   affects the initial estimate.
2. Add temperature as a third free parameter in the refinement stage.
3. Replace the single blackbody template with a composite SED that includes an
   emission line.

## Next Steps

Continue to [09-gravitational-lensing](../09-gravitational-lensing/) to see how
Rune's autodiff can fit physical parameters of a gravitational lens model from
observed image positions.
