# jord

jord gives Earth-observation data their meaning on the Earth. It says where
each value of an array lies, through coordinate reference systems, grids,
resampling and polygons, and what a product measured there: the physical
quantity, its special values and the acquisition that holds it. The
transforms, the grids' maps, the samplers and the decodes are nx
operations. Unless a value's doc says otherwise, it does no I/O, a reader
returns an `Error` for input it cannot read, a misuse raises
`Invalid_argument`, and a result lies where its input arrays lie or its
`~at` places it (the preamble of [`jord.mli`](../lib/jord.mli)).

A *system* says what a pair of numbers means on the Earth, a *point* is the
last axis of a float64 tensor, `[...; 2]`, in its system's axis order, and
a *grid* places an array's first two axes in a system. The preambles of
[`crs.mli`](../lib/crs.mli) and [`grid.mli`](../lib/grid.mli) define these
terms.

## Install

[README.md](../README.md) gives jord's status. In the raven repository,
add `jord` to an executable's `libraries`, each format library it reads,
and `nx.io`, which reads the image files. A program that goes from a
catalogue's item to a product's band, as the preamble of
[`jord.mli`](../lib/jord.mli) lists its steps, reads the item with
`jord.stac` and the product's metadata with `jord.safe`, and decodes the
item's JSON itself with `jsont` and `jsont.bytesrw`:

```dune
(executable
 (name main)
 (libraries
  jord
  jord.geotiff
  jord.stac
  jord.safe
  nx
  nx.io
  jsont
  jsont.bytesrw))
```

## A first program

This program finds the cell of a 10 m grid that holds a point of Berlin:

```ocaml
# open Jord;;
# let () =
    let lonlat = Nx.create Nx.float64 [| 1; 2 |] [| 13.4; 52.5 |] in
    let utm = Crs.utm_at ~lon:13.4 ~lat:52.5 in
    let grid =
      Grid.v utm ~rows:1000 ~cols:1000 ~origin:(388000., 5823000.)
        ~row_step:(0., -10.) ~col_step:(10., 0.)
    in
    let xy = Crs.transform ~src:Crs.lonlat ~dst:utm lonlat in
    Nx.print (Grid.pixel grid xy);;
float64 [1; 2]
[[514.4759182668291, 339.0731339950929]]
```

The point lies in cell (514, 339), about 5 km south and 3 km east of the
grid's corner.

`open Jord` brings jord's modules, such as `Crs` and `Grid`, into scope.
Each call, in order:

- `Nx.create Nx.float64 [| 1; 2 |] [| 13.4; 52.5 |]` is a float64 tensor
  of shape `[1; 2]` in host memory, one point of `Crs.lonlat`, longitude
  then latitude in degrees. Its last axis holds the point, and it is
  float64 since the transforms take float64 only (the Accuracy section of
  [`crs.mli`](../lib/crs.mli)).
- `Crs.utm_at ~lon:13.4 ~lat:52.5` is the UTM zone that holds the point,
  33N here, since zones are 6° wide from 180°W and northern from the
  equator. `Crs.utm_grid_at` also applies the zone exceptions between
  56°N and 64°N and between 72°N and 84°N (`utm_at` and `utm_grid_at` in
  the Systems section of [`crs.mli`](../lib/crs.mli)).
- `Grid.v utm ~rows ~cols ~origin ~row_step ~col_step` is a grid of 1000 ×
  1000 cells in that zone. `origin` is the corner of the first cell, its
  easting and northing in metres. `row_step` is the move in the plane,
  easting and northing, from one row to the next, `(0., -10.)` because rows
  run south, and `col_step` the move from one column to the next,
  `(10., 0.)` because columns run east. So the grid is north-up and each
  cell is 10 m square (the preamble of [`grid.mli`](../lib/grid.mli)).
- `Crs.transform ~src:Crs.lonlat ~dst:utm lonlat` takes the points
  `lonlat` from the system `~src` to the system `~dst`. It gives the point
  in the zone, easting then northing, to transverse Mercator's error ε of
  6 nm, which the Accuracy section of [`crs.mli`](../lib/crs.mli) defines.
  It raises if a point lies outside the projection's domain, the points
  within 40° of longitude of the zone's central meridian (the Domains
  section).
- `Grid.pixel grid xy` is the point as a pixel point `(row, col)`, which
  `Nx.print` prints. The point lies in cell `(⌊row⌋, ⌊col⌋)` (`pixel` in
  [`grid.mli`](../lib/grid.mli)), and cell `(i, j)` has its centre at
  `(i + 0.5, j + 0.5)` (the preamble).

## The libraries

| Library | Page | What it holds |
|---|---|---|
| [`jord`](../lib/jord.mli) | [Places and measurements](01-places.md) | Earth-observation data placed on the Earth |
| [`jord.geotiff`](../lib/geotiff/jord_geotiff.mli) | [GeoTIFF tags](02-geotiff.md) | The TIFF tags of OGC GeoTIFF 1.1 (19-008r4) that place an image on the Earth, with the grids of a Cloud Optimized GeoTIFF's overviews and GDAL's nodata tag |
| [`jord.geojson`](../lib/geojson/jord_geojson.mli) | [GeoJSON](03-geojson.md) | GeoJSON's polygons and points, by RFC 7946 |
| [`jord.shapefile`](../lib/shapefile/jord_shapefile.mli) | [Shapefiles](04-shapefile.md) | ESRI Shapefiles of polygons and their dBASE attributes |
| [`jord.stac`](../lib/stac/jord_stac.mli) | [STAC items](05-stac.md) | The STAC items that list a catalogue's acquisitions, with their files, grids and bands |
| [`jord.safe`](../lib/safe/jord_safe.mli) | [Sentinel-2 metadata](06-safe.md) | ESA's Sentinel-2 product metadata, from the bytes of its XML |

Each library's name links its main module, whose first sentence the row
gives, and [`jord.mli`](../lib/jord.mli) lists the modules of `jord`.

`jord` depends on nx alone and never on the format libraries, so a program
that only computes links no file reader.

[A Sentinel-2 scene, from catalogue to cloud mask](07-scene.md) traces one
program through these libraries.
