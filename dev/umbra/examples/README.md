# Umbra Examples

Learn Umbra through progressively complex examples. Start with
`01-constants-and-units` and work through the numbered examples in order.

## Examples

| Example | Concept | Key Functions |
|---------|---------|---------------|
| [`01-constants-and-units`](./01-constants-and-units/) | Type-safe physical quantities, conversions, constants | `Unit.Length.of_m`, `Const.c`, `Unit.Angle.deg` |
| [`02-cosmological-distances`](./02-cosmological-distances/) | LCDM distances, SN Ia fitting | `Cosmo.luminosity_distance`, `Cosmo.distance_modulus` |
| [`03-blackbody-fitting`](./03-blackbody-fitting/) | Fit stellar temperature from photometry | `Spectrum.blackbody`, `Photometry.ab_mag` |
| [`04-extinction-and-magnitudes`](./04-extinction-and-magnitudes/) | Dust extinction, magnitude systems, K-corrections | `Extinction.ccm89`, `Photometry.vega_mag`, `Photometry.color` |
| [`05-sed-fitting`](./05-sed-fitting/) | Full SED pipeline: blackbody, extinction, photometry | `Spectrum.blackbody`, `Extinction.apply`, `Photometry.ab_mag` |
| [`06-coordinates-and-time`](./06-coordinates-and-time/) | Frame transforms, time scales, observer geometry | `Coord.galactic_of_icrs`, `Time.of_iso`, `Altaz.airmass` |
| [`07-batch-photometry`](./07-batch-photometry/) | Batched operations over temperature and extinction grids | `Spectrum.blackbody`, `Extinction.apply`, `Photometry.ab_mag` |
| [`08-photometric-redshifts`](./08-photometric-redshifts/) | Two-stage photo-z: grid search + gradient refinement | `Spectrum.redshift`, `Photometry.ab_mag`, `Rune.value_and_grad` |
| [`09-gravitational-lensing`](./09-gravitational-lensing/) | Point-mass lens model parameter fitting | `Rune.value_and_grad`, `Vega.adam` |
| [`10-uncertainty-propagation`](./10-uncertainty-propagation/) | AD Jacobians for error propagation vs Monte Carlo | `Rune.jacfwd`, `Cosmo.distance_modulus` |
| [`11-bayesian-sed`](./11-bayesian-sed/) | Fisher matrix + HMC posterior sampling | `Rune.jacrev`, `Norn.hmc` |
| [`12-survey-optimization`](./12-survey-optimization/) | Differentiable Fisher forecasting for survey design | `Survey.angular_cl`, `Cosmo.linear_power` |

## Running Examples

All examples can be run with:

```bash
cd dev/umbra
dune exec --root . examples/<name>/main.exe
```

For example:

```bash
cd dev/umbra
dune exec --root . examples/01-constants-and-units/main.exe
```

## Quick Reference

### Cosmological Distances

```ocaml
open Umbra

let cosmo = Cosmo.planck18 in
let z = Nx.scalar Nx.float64 0.5 in
let dl = Cosmo.luminosity_distance cosmo z
```

### Synthetic Photometry

```ocaml
let sed =
  Spectrum.blackbody
    ~temperature:(Unit.Temperature.of_kelvin (Nx.scalar f64 5800.0))
    ~wavelength:wave
  |> Extinction.apply (Extinction.ccm89 ~rv) ~av
  |> Spectrum.as_flux_density
in
let mag = Photometry.ab_mag (Filters.sdss_r ()) sed
```

### Coordinate Transforms

```ocaml
let ra = Unit.Angle.deg 83.633 in
let dec = Unit.Angle.deg (-5.550) in
let l, b = Coord.galactic_of_icrs ra dec
```
