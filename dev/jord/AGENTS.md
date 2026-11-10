# jord

jord places arrays on the Earth: the coordinate reference systems
TESSERA's data use and the transforms between them, grids that tie an
array's first two axes to a system, polygons in a system and the cells
they cover, and the GeoTIFF tags that state a grid. It lives in `dev/`,
raven's sandbox, on the `tessera` branch: never published (no opam package,
no CHANGES entry). Its design is the eo-library adjudication of the tessera
campaign.

- The same rules as raven's libraries: `doc/dev/design.md`, AGENTS.md's code
  style, raven's public interfaces only (`Geotiff`'s tags are nx.io's
  `Tiff.value`). The one dependency outside raven is `jsont`, because
  `Jsont.json` is `Geojson.polygons`'s input, as it is brot's and
  kaun.hf's. No Python, weights or data in the repository beyond small
  test fixtures, each recording its source and the command that cut it
  (`test/tm_coords.dat`).
- jord does no I/O. Readers take bytes or values (a shapefile's bytes, a
  `Jsont.json`, a `.prj`'s text, TIFF tags) and return `Error` for data
  that breaks their format; misuse raises `Invalid_argument`.
- `lib/` is the library `jord`: `Crs` (WGS 84 and its UTM zones, Krüger's
  series to n⁶, `of_epsg`, `of_wkt`), `Grid` (an affine map in (row, col)
  order, GeoTIFF's corner convention), `Polygons` (GeoArrow's layout,
  even-odd regions), `Shapefile`, `Geojson`, `Rasterize` (the (feature,
  cell) relation of `centres` and `touched`), `Geotiff` (tags to and from
  a grid, GDAL's nodata). `Wkt` and `Points` are private.
- `test/` holds the laws and the spec's values, with a cut of Karney's
  test set. `test/rasterio/parity.exe` compares the rules with rasterio on
  tessera-eval's Austria labels and GeoTessera's landmasks, which live
  outside the repository, so no test runs it; `parity.exe landmask` reads
  real landmasks through `Nx_io.load_tiff` and `Geotiff`. `bench/` times each rule on
  a tile (`dune build @dev/jord/bench/bench`).

## Boundaries

- **nx.io** owns the TIFF container (directories, strips and tiles,
  codecs, BigTIFF). `Geotiff` reads and makes `Nx_io.Tiff.value` tags:
  `Nx_io.load_tiff` gives them, `Nx_io.save_tiff ~tags` writes them, and
  nothing in jord parses a TIFF file.
- **ymir** is astronomy's; jord does not depend on it. Its `Region`
  computes exact cell-polygon areas, the candidate mechanism for overlap
  weights, which would be copied, not shared.
- **The future geometry package** (spatial identity) may take
  `Rasterize`'s planar core and overlap weights when it lands; jord keeps
  the systems, grids and formats.
- **dev/tessera** keeps TESSERA's recipe: tile names and the tile-grid
  rule, the UTM zone exceptions of its ROI script, STAC, masking, scaling,
  time. A tile's grid comes from its landmask's tags, never from its
  corners (they differ by up to 1 µm).
- Later, in their own campaigns: the warp, ranged COG reads, overlap
  weights as a rule. Never: TLS or HTTP, GeoPackage, PROJ pipelines and
  datum grids, GDAL bit parity.

## Traps

- The transforms are float64 by design: float32 errs by metres.
- `Grid.equal` is exact; grids computed from corners are not equal to the
  landmasks' by design.
