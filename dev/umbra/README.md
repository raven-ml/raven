# Umbra

Computational astronomy for OCaml, powered by [Nx](../../packages/nx/) and [Rune](../../packages/rune/)

Umbra provides dimensionally-typed physical quantities, cosmological distances,
spectral energy distributions, dust extinction, synthetic photometry, coordinate
transforms, time scales, catalog cross-matching, and weak lensing survey science.
All computations operate on Nx tensors and are differentiable through Rune --
fit cosmological parameters, propagate uncertainties via Jacobians, or sample
posteriors with HMC, all from the same forward model.

## Quick Start

Compute the luminosity distance to a galaxy at redshift 0.5:

```ocaml
open Umbra

let () =
  let f64 = Nx.float64 in
  let z = Nx.scalar f64 0.5 in
  let dl = Cosmo.luminosity_distance ~p:Cosmo.planck18 z in
  Printf.printf "d_L(z=0.5) = %.1f Mpc\n"
    (Nx.item [] (Unit.Length.in_mpc dl))
```

Fit stellar temperature from photometry with automatic derivatives:

```ocaml
let model params =
  let temp = Unit.Temperature.of_kelvin (Nx.exp (Nx.slice [ I 0 ] params)) in
  let av = Nx.reshape [||] (Nx.slice [ I 1 ] params) in
  let rv = Nx.scalar Nx.float64 3.1 in
  List.map (fun bp ->
    let wave = Photometry.wavelength bp in
    let sed =
      Spectrum.blackbody ~temperature:temp ~wavelength:wave
      |> Extinction.apply (Extinction.ccm89 ~rv) ~av
      |> Spectrum.as_flux_density
    in
    Photometry.ab_mag bp sed) bands
  |> Nx.stack ~axis:0

(* Rune differentiates through the entire pipeline *)
let loss, grad = Rune.value_and_grad chi2 params
```

## Features

- **Dimensional types**: `Unit.Length`, `Unit.Mass`, `Unit.Time`, `Unit.Angle`, etc. with compile-time safety
- **Physical constants**: CODATA 2022 and IAU 2015 via `Const`
- **Cosmology**: LCDM, wCDM, w0waCDM distances, growth factors, and matter power spectra via `Cosmo`
- **Spectra**: blackbody, power-law, and line profiles (Gaussian, Lorentzian, Voigt) via `Spectrum`
- **Extinction**: CCM89, Fitzpatrick99, O'Donnell94, Calzetti00 dust laws via `Extinction`
- **Photometry**: AB, ST, and Vega magnitudes through standard filter bandpasses via `Photometry`
- **Filters**: SDSS, Johnson-Cousins, 2MASS, Gaia DR3, Rubin/LSST, Euclid via `Filters`
- **Coordinates**: ICRS, Galactic, Ecliptic, Supergalactic frame transforms and kd-tree cross-matching via `Coord`
- **Time**: UTC, TAI, TT, TDB time scales with phantom-typed safety via `Time`
- **Observer geometry**: altitude-azimuth coordinates and airmass via `Altaz`
- **Survey science**: angular power spectra and Fisher forecasting via `Survey`
- **FITS I/O**: image and table read/write via `Umbra_fits`
- **Fully differentiable**: all forward models work with Rune's autodiff, Jacobians, and MCMC

## Examples

| Example | Concept |
|---------|---------|
| [`01-constants-and-units`](examples/01-constants-and-units/) | Type-safe physical quantities and conversions |
| [`02-cosmological-distances`](examples/02-cosmological-distances/) | LCDM distances and SN Ia fitting |
| [`03-blackbody-fitting`](examples/03-blackbody-fitting/) | Fit stellar temperature from photometry |
| [`04-extinction-and-magnitudes`](examples/04-extinction-and-magnitudes/) | Dust extinction, magnitude systems, K-corrections |
| [`05-sed-fitting`](examples/05-sed-fitting/) | Full SED pipeline: blackbody, extinction, photometry |
| [`06-coordinates-and-time`](examples/06-coordinates-and-time/) | Frame transforms, time scales, observer geometry |
| [`07-batch-photometry`](examples/07-batch-photometry/) | Batched operations over parameter grids |
| [`08-photometric-redshifts`](examples/08-photometric-redshifts/) | Two-stage photo-z: grid search + gradient refinement |
| [`09-gravitational-lensing`](examples/09-gravitational-lensing/) | Point-mass lens model parameter fitting |
| [`10-uncertainty-propagation`](examples/10-uncertainty-propagation/) | AD Jacobians for error propagation vs Monte Carlo |
| [`11-bayesian-sed`](examples/11-bayesian-sed/) | Fisher matrix + HMC posterior sampling |
| [`12-survey-optimization`](examples/12-survey-optimization/) | Differentiable Fisher forecasting for survey design |

## Papers

- [**Perlmutter et al. 1999**](papers/perlmutter1999/) -- Reproducing the Nobel Prize-winning discovery of cosmic acceleration using the Pantheon+ dataset

## Contributing

See the [Raven monorepo README](../../README.md) for guidelines.

## License

ISC License. See [LICENSE](../../LICENSE) for details.
