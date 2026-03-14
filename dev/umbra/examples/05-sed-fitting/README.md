# `05-sed-fitting`

Full SED fitting pipeline: fits stellar temperature, dust extinction (A_V), and
flux normalization simultaneously from UGRIZ photometry. Demonstrates the
composable differentiable pipeline through Spectrum, Extinction, and Photometry.

```bash
dune exec dev/umbra/examples/05-sed-fitting/main.exe
```

## What You'll Learn

- Building a full astrophysical forward model from composable modules
- How the blackbody -> extinction -> photometry pipeline is end-to-end differentiable
- Creating custom bandpasses with `Photometry.tophat`
- Fitting multiple correlated parameters (T, A_V, normalization) simultaneously

## Key Functions

| Function                     | Purpose                                       |
| ---------------------------- | --------------------------------------------- |
| `Spectrum.blackbody`         | Planck spectral radiance at given wavelengths  |
| `Spectrum.scale`             | Scale spectrum values by a factor              |
| `Spectrum.as_flux_density`   | Cast to flux density kind for photometry       |
| `Extinction.ccm89`           | Create CCM89 extinction law with R_V           |
| `Extinction.apply`           | Apply dust reddening to a spectrum             |
| `Photometry.tophat`          | Create a rectangular bandpass                  |
| `Photometry.ab_mag`          | Compute AB magnitude through a bandpass        |
| `Rune.value_and_grads`       | Autodiff through the entire pipeline           |

## How It Works

The forward model constructs a synthetic SED at each optimization step:

1. **Spectrum.blackbody** generates the Planck function at temperature T
2. **Spectrum.scale** applies the flux normalization
3. **Extinction.apply** reddens the spectrum using CCM89 with extinction A_V
4. **Photometry.ab_mag** integrates through each bandpass to produce magnitudes

Since every step is built from Nx tensor operations, Rune computes gradients
of chi-squared with respect to all three parameters (log T, A_V, log norm) in
a single backward pass.

The temperature and normalization are parameterized in log-space for positivity
and better gradient conditioning. A_V is left in linear space since it can
meaningfully be zero or negative (de-reddening).

## Try It

1. Replace tophat filters with real SDSS filters from `Filters.sdss_u`, etc.
2. Add a redshift parameter to fit photometric redshifts.
3. Try `Extinction.fitzpatrick99` instead of `ccm89` and compare results.
4. Increase the photometric noise and observe how parameter uncertainties grow.

## Next Steps

Continue to [06-coordinates-and-time](../06-coordinates-and-time/) to work with
celestial coordinates, time scales, and observing conditions.
