# Changelog

All notable changes to Ymir are documented in this file.

## Unreleased

- `Ymir.Transform` maps pixel coordinates to the sky as a list of stages that
  invert and print back what they read: `axes`, `shift`, `linear`, `scale` and
  `celestial` with the TAN and ARC projections, `about` and `gnomonic` for
  offsets about a direction. `apply` raises at a point outside a projection's
  domain, naming it; `covers` gives the mask.
- `Ymir.Grid` is an image's cells seen through a transform, with `centres`,
  `corners`, each cell's exact `measure` (solid angle or area), and windows of
  static shape at traced, batched starts (`window`, `around`). `Grid.cell`
  marks data per cell.
- `Ymir.Region` places circles and annuli on a grid's world and weighs each
  cell by its exact covered fraction, differentiable in centre and radius.
- `Ymir.Observation` holds data on a grid with variance, validity and the
  area its pipeline states, and `integrate` sums them over a region: a cell
  counts by its area for a field and as one cell for data per cell.
- New library `ymir.fits`, FITS files without the compiler. `Fits.read`
  copies a file's headers, checks every data unit's extent and reads no data;
  `Fits.Header` keeps every record as the file held it, reading a value with
  `Fits.Value` when asked, so one bad card fails alone. `Fits.Image.raw` and
  `values` read stored numbers or physical values in the dtype the caller
  names, with windows that read only the rows they cover; `Fits.Image.hdu`
  and `Fits.write` write images whose structure, `DATASUM` and `CHECKSUM`
  the writer computes, and `Fits.verify` checks them.
- `Fits.Image` reads tile-compressed images (Rice, gzip, uncompressed and
  quantized tiles under every dither) as it reads plain ones, decoding only
  the tiles a window meets; HCOMPRESS and PLIO tiles are an `Error` naming
  funpack. `Fits.Image.hdu ~tiles` writes them losslessly, and
  `Fits.Image.quantized` writes floats Rice-coded in steps of each tile's
  noise, with cfitsio's quantizer and a dither seed derived from the pixels.
- `Fits.Unit` reads and writes FITS unit strings over `ymir.units` at the
  standard's values, and `Fits.unit` reads `BUNIT`, scoping `pix`, `chan`,
  `voxel` and `beam` to the file's digest and HDU.
- `Fits.Table` reads binary and ASCII tables: `raw` and `values` give a
  column as a tensor of `[rows] @ cell`, `ragged` gives text and heap arrays
  as `Nx_ragged.t`, `validity` marks TNULL, NaN and undefined logicals, and
  `read` returns every column in one pass. `Fits.Table.hdu` writes a
  binary table, numbering each column's cards and choosing its TNULL.
- `Ymir.Frame` names the fixed celestial frames, ICRS, FK5 at J2000, Galactic,
  the J2000 ecliptic and supergalactic, with one value per frame type.
  `Frame.matrix` converts between any two from their standards' orientations,
  each rounded once, so a conversion never depends on a path.
- `Ymir.Direction` holds float64 directions in a frame: `lonlat`, `of_xyz`,
  `lon`, `lat`, `rotate`, `separation` and `position_angle`. Readers accept any
  finite vector as its direction, keep relative accuracy at any separation and
  give derivative 0, never NaN, where no derivative exists.
- New library `ymir`, astronomy over nx: `open Ymir` brings `ymir.units`'
  modules and `Ymir.Units`, astronomy's units, with `Units.parsec`
  (648000/π au) and `Units.julian_year` (31557600 s), both exact.
- New package: exact physical units. `Ymir_units.Unit` is a unit as an exact
  monomial over primes, π and named symbols with rational exponents, with one
  canonical text (`to_string`, strict `of_string`), the SI's units, prefixes
  and exact defining constants, and `Unit.ratio`, a conversion factor
  correctly rounded to any nx dtype. `Ymir_units.Quantity` is a tensor in a
  unit, an `Nx.Ptree.S` that reports its unit's canonical text, with `value`
  and `convert` multiplying by that factor and raising on integer overflow.
- `Ymir_units.Constant` is a measured constant, a published decimal with its
  uncertainty, rounded once to a payload's dtype. `Ymir_units.Codata` holds
  the CODATA 2018 and 2022 releases, transcribed from NIST's tables.
- `Ymir_units.Vocabulary` names units: `lookup` reads a symbol with an SI
  prefix, `spell` and `pp` write a unit with a vocabulary's symbols, and
  `Vocabulary.si` holds the SI's.
