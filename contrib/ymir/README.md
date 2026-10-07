# Ymir

Astronomy for OCaml, built on [Nx](../../packages/nx/). Ymir gives tensors
astronomical meaning: exact units and quantities, celestial frames and
directions, FITS files, maps from pixels to the sky, aperture photometry on
images, and the background cosmology. Every function is a formula of nx
operations, so rune batches, compiles and differentiates it with no rule of its
own.

## Quick start

```ocaml
open Ymir

let () =
  (* Ages of a Planck 2018 universe at four redshifts, in gigayears. *)
  let planck = Cosmology.planck2018 ~codata:Codata.v2022 Nx.float64 in
  let z = Nx.create Nx.float64 [| 4 |] [| 0.; 1.; 3.; 1100. |] in
  let gyr = Unit.giga Units.julian_year in
  Nx.print (Quantity.value gyr (Cosmology.age planck z));

  (* The angle between two directions, in arcseconds. *)
  let deg x = Quantity.v Unit.degree (Nx.scalar Nx.float64 x) in
  let a = Direction.lonlat Frame.icrs ~lon:(deg 10.) ~lat:(deg 20.) in
  let b = Direction.lonlat Frame.icrs ~lon:(deg 10.001) ~lat:(deg 20.) in
  Nx.print (Quantity.value Unit.arcsecond (Direction.separation a b))
```

The [examples](examples/) teach each part on small synthetic data.

## Libraries

- `ymir` is the astronomy; `open Ymir` brings every module below except
  `Fits` into scope.
- `ymir.units` holds the units alone, for code that needs no astronomy.
- `ymir.fits` reads and writes FITS files; `open Ymir_fits` brings `Fits`.
  Neither `ymir` nor `ymir.fits` depends on the other, and a program that
  reads files links both.

## What's inside

- **Units**: `Unit` is a unit as an exact value, a product of primes, π and
  named symbols with rational exponents, with one canonical text. A
  conversion rounds the exact factor once, or raises when the units don't
  convert. `Quantity` is a tensor in a unit. The SI's units, prefixes and
  exact constants are built in, `Codata.v2018` and `Codata.v2022` hold the
  measured constants, and `Vocabulary` spells units with symbols such as
  `MJy sr^-1`.
- **Frames and directions**: `Frame` names ICRS, FK5 J2000, Galactic, the
  J2000 ecliptic and supergalactic, and a frame mismatch is a type error.
  `Direction` holds batches of directions with their separations, position
  angles and rotations between frames.
- **Transforms**: `Transform` maps pixels to the sky as a list of stages
  that compose with `>>`, invert, and print back what they read: axis order,
  shift, linear maps, SIP and TPV distortions, thirteen zenithal and
  cylindrical projections, and rotations between frames.
- **Grids, regions and observations**: `Grid` is an image's cells seen
  through a transform, with each cell's exact area or solid angle;
  `Grid.agree` says whether two grids are the same cells. `Region` places
  circles, annuli, ellipses and polygons on a grid and weighs each cell by
  its exact covered fraction. `Observation` holds data with variance and
  validity; `add`, `sub` and `scale` combine observations on agreeing grids
  with their variances, and `integrate` sums one over a region: aperture
  photometry, differentiable in the aperture's centre and size.
- **FITS**: `ymir.fits`'s `Fits` reads and writes headers, images (tile
  compression included), binary and ASCII tables, and `BUNIT` and `TUNIT`
  units. `Wcs` reads a header's world coordinates as a `Transform` and
  writes one back as keyword edits; a program composes an image's data,
  error and world coordinates into an `Observation`.
- **Cosmology**: `Cosmology.t` is one record of tensors for flat and curved
  ΛCDM, wCDM and w0waCDM with radiation and massive neutrinos. Expansion
  rate, density parameters, distances, volumes and times are fixed
  Gauss–Legendre sums with a stated error bound (2⁻⁴⁶ relative in float64),
  batched over models and redshifts. The Planck and WMAP fits are built in.

## Validation

Two results on public data reproduce published ones. They download their
data, so they run outside `runtest`; `runtest` checks the same paths on
bundled cutouts and simulated data.

- `test/nircam`: aperture photometry on a JWST NIRCam mosaic agrees with
  photutils within 10⁻⁶ of the sums, with gradients in the aperture's centre
  and radius checked against finite differences.
- `test/pantheon`: fitting Ω_m and Ω_Λ to the Pantheon+ and SH0ES
  supernovae with their full covariance reproduces Brout et al. (2022),
  Table 3.
