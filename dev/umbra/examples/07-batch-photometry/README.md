# `07-batch-photometry`

Computes SDSS g-r colors for a grid of blackbody templates at different
temperatures and dust extinctions in a single pass using batch operations.
Instead of looping over individual spectra, the values tensor has a leading
batch dimension and all photometry operations broadcast over it.

```bash
cd dev/umbra
dune exec --root . examples/07-batch-photometry/main.exe
```

## What You'll Learn

- Constructing batched spectra by stacking blackbodies into a leading dimension
- Broadcasting extinction across a batch of SEDs with per-spectrum A_V
- Computing synthetic SDSS photometry with AB magnitudes
- Exploring color-temperature and color-extinction relations

## Key Functions

| Function                   | Purpose                                          |
| -------------------------- | ------------------------------------------------ |
| `Spectrum.blackbody`       | Generate Planck spectrum at a given temperature  |
| `Spectrum.create`          | Build a spectrum from wavelength and value arrays |
| `Spectrum.as_flux_density` | Cast to flux density kind for photometry         |
| `Nx.stack`                 | Stack individual spectra into a batch dimension  |
| `Extinction.ccm89`         | Create CCM89 dust extinction law                 |
| `Extinction.apply`         | Apply reddening with per-spectrum A_V broadcast  |
| `Photometry.ab_mag`        | Compute AB magnitude through a bandpass          |
| `Filters.sdss_g`           | SDSS g-band filter response                      |

## How It Works

The example first builds a grid of 20 blackbody spectra from 3000 K to 30000 K
by stacking individual `Spectrum.blackbody` outputs into a `[n_temp; 500]`
values tensor. When this batch spectrum is passed to `Photometry.ab_mag`, the
integration broadcasts over the leading dimension, producing one magnitude per
temperature in a single call.

The second half demonstrates per-spectrum extinction. A T=6000 K blackbody is
replicated into 10 copies, and `Extinction.apply` is called with an A_V tensor
of shape `[n_av; 1]` that broadcasts against the `[n_av; 500]` flux values.
This yields reddened g-r colors across a range of dust columns without any
explicit loop.

## Try It

1. Increase the temperature grid to 100 points and plot the g-r color curve to
   see where the blue turnover occurs.
2. Add a third band (sdss_i) and compute the g-r vs r-i color-color diagram.
3. Replace the blackbody with a power-law spectrum and observe how the color
   trends differ.

## Next Steps

Continue to [08-photometric-redshifts](../08-photometric-redshifts/) to learn
how to estimate galaxy redshifts by combining grid search with gradient-based
refinement through the differentiable photometry pipeline.
