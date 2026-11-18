# Jord

Earth observation for OCaml, built on [Nx](../../packages/nx/). jord gives
Earth-observation data their meaning on the Earth: where each value lies
(coordinate reference systems, grids, resampling, polygons and the cells
they cover) and what a product measured there (the physical quantity, its
special values, its acquisition). It does no I/O, so a program fetches and
opens files with the tools it likes. [`jord.mli`](lib/jord.mli) states what
every value does unless its doc says otherwise. The transforms, the grids'
maps, the warps that move an array from one grid to another and the
decodes are nx operations. A warp reads the array at a point of each cell,
by a *sampler*, or over each cell's area, by an *area rule*, and rune
differentiates the bilinear sampler and the average area rule in their
points and corners (the Gradients section of
[`warp.mli`](lib/warp.mli)).

jord is a contrib package: it builds and tests against `main`, and has no
API stability guarantee ([`contrib/README.md`](../README.md)).

## When to use jord

jord serves an OCaml program that puts Earth-observation products on grids
of its own, such as a model's input: it reads where a product's arrays lie
and what their DNs mean, moves the arrays onto the program's grid and masks
their quality, in nx operations that rune can compile and differentiate.

jord does not do the following. It knows five datums, WGS 84, NAD83,
ETRS89-FIN, NZGD2000 and GDA94, the four besides WGS 84 each taken to it
by a step that changes no coordinate, refuses data on any other, and
reads no datum grid such as NADCON or NTv2 (the Datums section of
[`crs.mli`](lib/crs.mli)). Its projections are transverse
Mercator, Albers' equal-area conic, Pseudo-Mercator and the sinusoid, so
it has no conformal conic system, such as Lambert's, and no polar
stereographic one, such as UPS (the Projections section). It reads no
JPEG 2000, which nx.io does not read either, so no pixels of ESA's SAFE
products or of the Copernicus Data Space Ecosystem's Sentinel-2 assets
(the preamble of [`jord_safe.mli`](lib/safe/jord_safe.mli)). Its area
rules are provisional (the Area rules section of
[`warp.mli`](lib/warp.mli)). A program that needs one of these takes it
from another library.

## Install

jord is not in the opam repository yet, and it builds against raven's
`main`. In the raven repository, an executable adds `jord` to its
`libraries` ([the quick start](doc/index.md) lists the others it may
need). Outside it, opam pins raven's packages at `main`, and jord from
its directory, and installs jord with the nx it needs:

<!-- $MDX skip -->
```bash
opam pin add -n git+https://github.com/raven-ml/raven.git#main
opam pin add -n jord git+https://github.com/raven-ml/raven.git#main --subpath contrib/jord
opam install jord
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

[The quick start](doc/index.md) explains each call, and [`doc/`](doc/)
holds a page for each library, with worked examples that the test suite
runs.

## Libraries

| Library | Depends on | What it holds |
|---|---|---|
| [`jord`](lib/jord.mli) | nx | Earth-observation data placed on the Earth |
| [`jord.geotiff`](lib/geotiff/jord_geotiff.mli) | nx.io | The TIFF tags of OGC GeoTIFF 1.1 (19-008r4) that place an image on the Earth, with the grids of a Cloud Optimized GeoTIFF's overviews and GDAL's nodata tag |
| [`jord.geojson`](lib/geojson/jord_geojson.mli) | jsont | GeoJSON's polygons and points, by RFC 7946 |
| [`jord.shapefile`](lib/shapefile/jord_shapefile.mli) | | ESRI Shapefiles of polygons and their dBASE attributes |
| [`jord.stac`](lib/stac/jord_stac.mli) | jsont | The STAC items that list a catalogue's acquisitions, with their files, grids and bands |
| [`jord.safe`](lib/safe/jord_safe.mli) | xmlm | ESA's Sentinel-2 product metadata, from the bytes of its XML |

Each library's name links its main module, whose first sentence the row
gives, and [`jord.mli`](lib/jord.mli) lists the modules of `jord`. `jord`
never depends on the format libraries, so a program that only computes
links no file reader.

## Licence

ISC, as [LICENSE](LICENSE) states.
