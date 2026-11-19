# jord

jord places arrays on the Earth: the coordinate reference systems
Earth-observation products use and the transforms between them, grids that
tie an array's first two axes to a system, polygons in a system and the
cells they cover, the GeoTIFF tags that state a grid, what a band's stored
numbers measure, and the STAC items that list them. It is a project of its
own in `contrib/`: its `dune-project` declares the opam package `jord`, and
it keeps its own `CHANGES.md`, `README.md` and guide pages in `doc/`, which
run under mdx.

- Changes go in `contrib/jord/CHANGES.md`.

- The same rules as raven's libraries: `docs/dev/design.md`, AGENTS.md's code
  style, raven's public interfaces only (`Jord_geotiff`'s tags are nx.io's
  `Tiff.value`). Two dependencies come from outside raven: `jsont`,
  because `Jsont.json` is `Jord_geojson`'s and `Jord_stac`'s input, as it
  is brot's and kaun.hf's; and `xmlm`, which `Jord_safe` reads ESA's
  metadata with, and whose types it does not expose. No weights or data in
  the repository beyond small test fixtures, each recording its source and
  the command that made it; `test/README.md` gives each one's terms, and
  its generator is in `test/gen/`, run through `uv run`.
- The preamble of `lib/jord.mli` states the defaults every value follows
  unless its doc says otherwise: no I/O, `Error` for input a reader cannot
  read, `Invalid_argument` for misuse, results where the inputs lie. A new
  value keeps to them or states its exception in its doc.
- The docs follow raven's `docs/dev/documentation.md`. A value sits under
  its section's heading: the section's preamble states each rule in a
  sentence, and a subsection with a rule's detail comes after the values
  it governs, whose docs link it (`Crs.of_wkt` and its WKT rules,
  `Warp.average` and the antimeridian, `Jord_safe.l1c` and its refusals).
- `lib/` is the library `jord`, which links nx alone. `lib/jord.mli`
  lists its public modules in four sections, each with its synopsis, and
  each module's own `.mli` says what it holds. A new public module goes
  into `jord.mli` and `jord.ml`, in its section. The modules that
  `lib/dune` names in `private_modules` are private.
- The module boundaries follow one rule. A noun is a module with its `t`
  (`Crs`, `Grid`, `Polygons`, `Band`), and a family of rules over them is
  a module of its own (`Rasterize`, `Warp`). A producer's product is a
  module (`Sentinel2`, `Landsat`). Inside a product, each quality layer is
  a submodule named after its band's constructor (`Landsat.Qa_pixel`,
  `Modis.State`), or after the producer's name for a layer outside the
  module's bands (`Sentinel2.Scl`). A layer's named one-bit flags are its
  type `flag`, read by `bit` and `any`, and a classified layer's classes
  its type `t`, read by `code` and `mem`.
- Each file or document format is a library of its own, `lib/<format>/`,
  public as `jord.<format>`, its module `Jord_<format>`. `README.md`'s
  Libraries table gives each one's dependency and links its `.mli`.
  `jord.stac` does not depend on `jord.geojson`, since an item's footprint
  is `Jord_geojson.polygons` of the same JSON. `jord` never depends on
  them, so a compiled step links no file reader; its dune file holds
  that.
- `test/` holds the laws and the spec's values, each sub-library's suite in
  `test/<format>/`, with a cut of Karney's test set and GDAL's warps of
  small rasters onto finer grids (`test/warp_gdal.dat`), the USGS's Landsat
  spectral responses around each half-maximum crossing
  (`test/landsat_rsr.dat`), each written by its generator in `test/gen/`,
  and items of Microsoft's Planetary Computer (`test/stac/` and
  `test/safe/`, fetched by the commands their dune files name).
  `test/rasterio/parity.exe` compares the rules with rasterio on labels and
  landmasks that live outside the repository (its dune file names them), so
  no test runs it; `parity.exe landmask` reads real landmasks through
  `Nx_io.load_tiff` and `Jord_geotiff`. `bench/` times each rule on a tile
  (`dune build @contrib/jord/bench/bench`).

## Boundaries

- **nx.io** owns the TIFF container (directories, strips and tiles,
  codecs, BigTIFF). `Jord_geotiff` reads and makes `Nx_io.Tiff.value` tags:
  `Nx_io.load_tiff` gives them, `Nx_io.save_tiff ~tags` writes them, and
  nothing in jord parses a TIFF file.
- **ymir** is astronomy's; jord does not depend on it. Its `Region`
  computes exact cell-polygon areas; `Rasterize.areas` computes its own,
  by slabs.
- **The future geometry package** (spatial identity) may take
  `Rasterize`'s planar core and overlap weights when it lands; jord keeps
  the systems, grids and formats.
- **A consumer** keeps its recipe: its tile names and tile-grid rule,
  which classes it masks, its band order, its mosaic and duplicate rules,
  its storage codes and its fetching. jord holds the facts a recipe rests
  on, such as the UTM grid's zone exceptions (`Crs.utm_grid_at`). A tile's
  grid comes from its landmask's tags, never from its corners, which differ
  by up to 1 µm.
- Later, as designs of their own: ranged COG reads, and zonal rules over
  `Rasterize.areas` beyond its mli's mean. Never: TLS
  or HTTP, GeoPackage, PROJ pipelines and datum grids, GDAL bit parity.

## Traps

- The transforms take float64 only, since float32 errs by metres (the
  Accuracy section of `lib/crs.mli`).
- `Grid.equal` compares exactly (`lib/grid.mli`, `equal`), so a grid
  computed from corners differs from the landmask grid it approximates.
- A warp takes a source's pixel points by `Warp.pixel`, which wraps a
  geographic source's longitudes into the grid's 360°, where `Grid.pixel`
  leaves them (`lib/warp.mli`, `pixel`).
- `Warp.centres` skips the transform for equal systems (`lib/warp.mli`,
  `centres`). A round trip through the datum's longitude and latitude
  moves points by its rounding, which breaks the exact 1/4 and 3/4 weights
  of a lattice-aligned warp.
