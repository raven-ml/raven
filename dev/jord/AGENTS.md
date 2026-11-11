# jord

jord places arrays on the Earth: the coordinate reference systems
TESSERA's data use and the transforms between them, grids that tie an
array's first two axes to a system, polygons in a system and the cells
they cover, the GeoTIFF tags that state a grid, what a band's stored
numbers measure, and the STAC items that list them. It lives in `dev/`,
raven's sandbox, on the `tessera` branch: never published (no opam package,
no CHANGES entry). Its design is the eo-library adjudication of the tessera
campaign.

- The same rules as raven's libraries: `doc/dev/design.md`, AGENTS.md's code
  style, raven's public interfaces only (`Jord_geotiff`'s tags are nx.io's
  `Tiff.value`). The one dependency outside raven is `jsont`, because
  `Jsont.json` is `Jord_geojson`'s and `Jord_stac`'s input, as it is
  brot's and kaun.hf's. No Python, weights or data in the repository beyond
  small test fixtures, each recording its source and the command that cut
  it (`test/tm_coords.dat`).
- jord does no I/O. Readers take bytes or values (a shapefile's bytes, a
  `Jsont.json`, a `.prj`'s text, TIFF tags) and return `Error` for data
  that breaks their format; misuse raises `Invalid_argument`.
- `lib/` is the library `jord`, which links nx alone: `Crs` (WGS 84 and its
  UTM zones, ETRS89's TM35FIN, Krüger's series to n⁶, `of_epsg`, `of_wkt`,
  WKT being a system's own text form), `Grid` (an affine map in (row, col)
  order, GeoTIFF's corner convention), `Polygons` (GeoArrow's layout,
  even-odd regions), `Rasterize` (the (feature, cell) relation of `centres`
  and `touched`), `Warp` (a grid's centres in another system, pixel points,
  the nearest and bilinear samplers, the source window they read), `Band`
  (STAC's band: scale, offset, nodata and saturated values, one decode,
  never applied to bitfield bands), `Sentinel2` (ESA's Level-2A: bands and
  resolutions, scene classes, reflectance as a `Band` from the product's
  stated offset, the acquisition key of a datastrip). `Wkt` and `Points`
  are private.
- Each file or document format is a library of its own, `lib/<format>/`,
  named `jord_<format>` (`jord.<format>` once jord has a package):
  `jord_geotiff` (nx.io: tags to and from a grid, GDAL's nodata),
  `jord_geojson` (jsont), `jord_shapefile` (no dependency), `jord_stac`
  (jsont: items, their assets' grids and bands, in STAC 1.0's and 1.1's
  spellings; an item's footprint is `Jord_geojson.polygons` of the same
  JSON, so it needs no `jord_geojson`). `jord` never depends on them, so a
  compiled step links no file reader; its dune file holds that.
- `test/` holds the laws and the spec's values, each sub-library's suite in
  `test/<format>/`, with a cut of Karney's test set and GDAL's warps of
  small rasters onto finer grids (`test/warp_gdal.dat`, cut outside the
  repository by the script its header names), and items of Microsoft's
  Planetary Computer (`test/stac/`, fetched by the command its dune file
  names).
  `test/rasterio/parity.exe` compares the rules with rasterio on
  tessera-eval's Austria labels and GeoTessera's landmasks, which live
  outside the repository, so no test runs it; `parity.exe landmask` reads
  real landmasks through `Nx_io.load_tiff` and `Jord_geotiff`. `bench/` times
  each rule on a tile (`dune build @dev/jord/bench/bench`).

## Boundaries

- **nx.io** owns the TIFF container (directories, strips and tiles,
  codecs, BigTIFF). `Jord_geotiff` reads and makes `Nx_io.Tiff.value` tags:
  `Nx_io.load_tiff` gives them, `Nx_io.save_tiff ~tags` writes them, and
  nothing in jord parses a TIFF file.
- **ymir** is astronomy's; jord does not depend on it. Its `Region`
  computes exact cell-polygon areas, the candidate mechanism for overlap
  weights, which would be copied, not shared.
- **The future geometry package** (spatial identity) may take
  `Rasterize`'s planar core and overlap weights when it lands; jord keeps
  the systems, grids and formats.
- **dev/tessera** keeps TESSERA's recipe: tile names and the tile-grid
  rule, the UTM zone exceptions of its ROI script, which classes it masks,
  band order, the mosaic and duplicate rules, its storage codes, fetching.
  A tile's grid comes from its landmask's tags, never from its corners
  (they differ by up to 1 µm).
- Later, in their own campaigns: ranged COG reads, overlap weights as a
  rule, and the area rules (mean, mode) that would use them. Never: TLS
  or HTTP, GeoPackage, PROJ pipelines and datum grids, GDAL bit parity.

## Traps

- The transforms are float64 by design: float32 errs by metres.
- `Grid.equal` is exact; grids computed from corners are not equal to the
  landmasks' by design.
- A warp takes a source's pixel points by `Warp.pixel`, not `Grid.pixel`:
  only it wraps a `Lonlat` source's longitudes into the grid's 360°.
- `Warp.centres` skips the transform for equal systems. A round trip
  through `Lonlat` moves points by about 1e-9 m, which breaks the exact
  1/4 and 3/4 weights of a lattice-aligned warp.
