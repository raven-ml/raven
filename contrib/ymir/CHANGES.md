# Changelog

All notable changes to Ymir are documented in this file.

## Unreleased

- Gzip tiles decode and encode through `lz`. `Fits.Image.raw` and
  `Fits.Image.values` inflate a tile without copying it first, and a broken
  tile's `Error` ends as `lz` prints it (`malformed at byte 25: ...`).
  `Fits.Image.hdu ~tiles` writes other gzip bytes, which read the same.
- `Region.weights` weighs a region at a float narrower than float32
  (float16, bfloat16, float8) at float32 and rounds the weights once. The
  kernel ran at the narrow float, which cannot hold a cell's offsets or their
  squares: at float8 every 40-cell grid raised that the transforms fold it,
  and float16 offsets past 65504 gave NaN weights.
- `Direction.separation` and `Direction.position_angle` keep their accuracy
  on rows of unequal norms, as `Nx.Ptree.map` can build, and at separations
  far below float64's rounding: both read `a × b`, computed with fused
  multiply-adds and scaled by a power of two. Rows of norms 1 and 1.3 at 1 µas
  were 3e-6 off, and a separation of 1e-160 rad 6e-6 off.
- `Vocabulary.spell` is `None` where an item of the spelling, the power of ten
  left or a word to its exponent, is past the unit algebra's 4096-bit bound.
  It returned spellings such as `1e-1236 m^309` with `m` = 1e4 π, whose items
  no unit holds, so their text did not read back.
- `Region.weights` keeps a cell's derivative where its weight rounds to 0 or
  1. A circle just past tangent to a cell edge cuts a sliver whose area rounds
  to 0, and the clamp dropped its arc from the derivative in the radius: 1e-6
  of the circumference at float64, 2e-3 at float32, for a radius of 3.5 cells.
- `Wcs.read` refuses a header that omits LATPOLE where LATPOLE alone fixes
  the celestial pole (θ₀ = 0, CRVAL's latitude 0, LONPOLE 90° from φ₀), as
  FITS 4.0 gives it no default there; it read as 90°. astropy reads such a
  header, with the pole at 90°. `Wcs.write` always writes LATPOLE there.
- `Fits.write` works on Windows, where it raised
  `Invalid_argument "Unix.fchmod not implemented"`: Windows has no file
  modes, and the file stays writable there.
- `Fits.Value.float` reads and prints the same on every system. On Windows,
  msvcrt's strtod misread some decimals, such as `3.839223843572815E-239`,
  and its printf wrote three exponent digits (`6.02214076E+023`) and
  rounded some digits twice. Printing now gives the nearest of the fewest
  digits on every system, so a few values print differently: `2^56` prints
  as `72057594037927940.0`, not `72057594037927936.0`.
- An ASCII table's real field past float64's range, such as `1E400`, is an
  `Error` naming its row, as a header value past it is; it read as infinity.
- Quantized float tiles decode to the same values on every platform: on
  arm64, `Fits.Image.raw` and `Fits.Image.values` rounded `I × ZSCALE +
  ZZERO` once instead of twice, and could differ in the last bit.
  `Fits.Image.quantized` rounds its zero point the same way.
- `Fits.Image.hdu ~tiles` raises `Invalid_argument` on an int64 or uint64
  image: cfitsio, behind funpack, imcopy and ds9, cannot decompress 64-bit
  integer tiles. Write such an image plain; reading these tiles still works.
- `Rune.grad` over a record of quantities, such as a `Cosmology.t`, returns
  the record with each field holding its derivative in the field's unit, the
  payload per that unit. A derivative per another unit is the gradient at the
  field converted to it; `Quantity.value` on a gradient converts a step.
- `Ymir.Wcs.read` turns a FITS header's celestial description (FITS's
  thirteen zenithal and cylindrical projections with their PV terms; SIP,
  TPV and SCAMP's TAN with PV terms; CD, PC, CDELT and CROTA2; ICRS, FK5
  J2000, Galactic, ecliptic and supergalactic) into a `Transform.t`, asking
  a `Wcs.keywords` view for each keyword's decoded value, and `Wcs.write`
  returns the typed keyword edits that print one back. `ymir` does not
  depend on `ymir.fits`: a program links both, builds the view from
  `Fits.Header.find`, and composes an observation from `Fits.Image.values`,
  `Grid.pixels` and `Observation.v`.
- `Ymir.Cosmology`, the background of a homogeneous expanding universe: one
  record of tensors for flat and curved ΛCDM, wCDM and w0waCDM with radiation
  and massive neutrinos, and its expansion rate, density parameters,
  distances, volumes and times as fixed Gauss–Legendre sums, batched,
  compiled and differentiable in every parameter, NaN where the universe has
  no past. `planck2018` to `wmap1` transcribe each paper's fit.
- `Ymir.Transform` maps pixel coordinates to the sky as a list of stages that
  invert and print back what they read: `axes`, `shift`, `linear`, `scale`,
  the `sip` and `tpv` distortions, whose inverses solve with jera,
  `celestial` with FITS's thirteen zenithal and cylindrical projections and
  any native reference point, `rotation` between fixed frames, and `about`
  and `gnomonic` for offsets about a direction. `apply` raises at a point
  outside a stage's domain, naming it; `covers` gives the mask.
- `Ymir.Grid` is an image's cells seen through a transform, with `centres`,
  `corners`, each cell's exact `measure` (solid angle or area), and windows of
  static shape at traced, batched starts (`window`, `around`). `Grid.cell`
  marks data per cell. `map_world` sees the cells in another world and keeps
  their measures; `agree` tells whether two grids are the same cells, exactly.
- `Ymir.Region` places circles, annuli, ellipses and polygons on a grid's
  world and weighs each cell by its exact covered fraction, differentiable in
  centre and size and continuous in a polygon's vertices.
- `Quantity.t` is declared injective (`type !'p t`), so a GADT
  can be indexed by quantities.
- `Ymir.Observation` holds data on a grid with variance, validity and the
  area its pipeline states, and `integrate` sums them over a region: a cell
  counts by its area for a field and as one cell for data per cell. `add`,
  `sub` and `scale` combine observations on agreeing grids with exact
  variances, and `map_world` moves one to another world.
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
