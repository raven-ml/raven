# `04-extinction-and-magnitudes`

Explores three key photometric concepts: magnitude systems (AB, ST, Vega),
K-corrections from redshift, and interstellar dust extinction. Shows how to
compose `Spectrum`, `Extinction`, `Photometry`, and `Filters` modules.

```bash
dune exec dev/umbra/examples/04-extinction-and-magnitudes/main.exe
```

## What You'll Learn

- Computing AB, ST, and Vega magnitudes through real SDSS filters
- Understanding K-corrections from redshift-shifted SEDs
- Applying extinction laws (CCM89, Fitzpatrick99, O'Donnell94)
- Measuring colors and color excess from dust reddening

## Key Functions

| Function                    | Purpose                                        |
| --------------------------- | ---------------------------------------------- |
| `Photometry.ab_mag`         | AB magnitude through a bandpass                |
| `Photometry.st_mag`         | ST magnitude through a bandpass                |
| `Photometry.vega_mag`       | Vega magnitude through a bandpass              |
| `Photometry.color`          | Color index (mag difference between two bands) |
| `Spectrum.blackbody`        | Planck spectral radiance                       |
| `Spectrum.redshift`         | Apply cosmological redshift to an SED          |
| `Spectrum.as_flux_density`  | Cast spectrum to flux density kind             |
| `Extinction.ccm89`          | Cardelli, Clayton & Mathis (1989) dust law     |
| `Extinction.fitzpatrick99`  | Fitzpatrick (1999) dust law                    |
| `Extinction.apply`          | Redden a spectrum by A_V magnitudes            |
| `Filters.sdss_r`            | Pre-built SDSS r-band bandpass                 |

## How It Works

**Magnitude systems** differ in their reference flux:
- AB: constant f_nu = 3631 Jy
- ST: constant f_lambda = 3.63e-9 erg/s/cm^2/A
- Vega: the spectrum of alpha Lyrae

**K-corrections** arise because redshift moves the SED across the bandpass,
changing the measured flux even without distance dimming. K(z) = m_obs - m_rest.

**Extinction** attenuates and reddens starlight. The extinction curve
A_lambda/A_V depends on wavelength and the dust grain properties (encoded in
R_V). Higher A_V means more dimming; bluer bands are affected more, producing
reddening.

## Try It

1. Compare Galactic extinction (CCM89, R_V=3.1) with starburst attenuation
   (`Extinction.calzetti00`).
2. Apply both redshift and extinction to see their combined effect on colors.
3. Use `Extinction.unredden` to recover the intrinsic SED from a reddened
   observation.

## Next Steps

Continue to [05-sed-fitting](../05-sed-fitting/) to fit temperature, extinction,
and normalization simultaneously.
