# Places and measurements

The library `jord` holds the rules that need no file format.
[`jord.mli`](../lib/jord.mli) lists its modules, and this page runs most
of them on small arrays.

```ocaml
# open Jord;;
```

## Systems

A file names its system by an EPSG code, which `Crs.of_epsg` reads. jord
knows the datums WGS 84, NAD83, ETRS89-FIN [EUREF-FIN], NZGD2000 and GDA94
(the Datums section of [`crs.mli`](../lib/crs.mli)). `Crs.of_epsg` reads
their longitude and latitude, WGS 84's and NAD83's UTM zones, Finland's
and New Zealand's national grids, NAD83's and GDA94's Albers' conics and
WGS 84's Pseudo-Mercator, and refuses every other code (the Codes and text
section), so that no data is read as if it were in one of these:

```ocaml
# Crs.of_epsg 32633;;
- : (Crs.t, string) result =
Ok
 {Crs.datum = Crs.Wgs84;
  projection =
   Some
    (Crs.Transverse_mercator
      {Jord.Crs.lat0 = 0.; lon0 = 15.; k0 = 0.9996; false_easting = 500000.;
       false_northing = 0.})}
# let () = Format.printf "%a@." Crs.pp (Crs.utm ~zone:33 ~north:false);;
WGS 84 / UTM zone 33S
# Crs.of_epsg 25835;;
- : (Crs.t, string) result =
Error "EPSG:25835 is not one of the systems Jord.Crs.of_epsg reads"
```

A transform maps each point alone, through the longitude and latitude of
each system's datum and the datum steps between them (the Transforms
section of [`crs.mli`](../lib/crs.mli)). Zones 33N and 32N share WGS 84,
so a point of zone 33N goes to zone 32N as its longitude and latitude
would:

```ocaml
# let () =
    let p = Nx.create Nx.float64 [| 1; 2 |] [| 500000.; 5800000. |] in
    let z32 = Crs.utm ~zone:32 ~north:true in
    let z33 = Crs.utm ~zone:33 ~north:true in
    Nx.print (Crs.transform ~src:z33 ~dst:Crs.lonlat p);
    Nx.print (Crs.transform ~src:z33 ~dst:z32 p);;
float64 [1; 2]
[[15, 52.35029334885088]]
float64 [1; 2]
[[908490.3303809444, 5816961.726947372]]
```

The point lies on zone 33's central meridian, 15°E, and 6° from zone 32's,
9°E, inside the 40° of the projections' domain (the Domains section of
[`crs.mli`](../lib/crs.mli)).

From here on the toplevel prints each system as `Crs.pp` does, by its EPSG
name:

```ocaml
# #install_printer Crs.pp;;
```

## Grids

A grid's cells are squares of pixel points, and its affine map takes a
pixel point to its system's plane (the preamble of
[`grid.mli`](../lib/grid.mli)). `Grid.window` is a block of a grid's
cells, as a tile of a larger image:

```ocaml
# let grid =
    Grid.v (Crs.utm ~zone:33 ~north:true) ~rows:4 ~cols:4
      ~origin:(500000., 5800040.) ~row_step:(0., -10.) ~col_step:(10., 0.);;
val grid : Grid.t =
  {Grid.crs = WGS 84 / UTM zone 33N; rows = 4; cols = 4;
   origin = (500000., 5800040.); row_step = (0., -10.); col_step = (10., 0.)}
# let tile = Grid.window ~row:2 ~col:2 ~rows:2 ~cols:2 grid;;
val tile : Grid.t =
  {Grid.crs = WGS 84 / UTM zone 33N; rows = 2; cols = 2;
   origin = (500020., 5800020.); row_step = (0., -10.); col_step = (10., 0.)}
# let () = Nx.print (Grid.centres ~at:Nx.Host.on tile);;
float64 [2; 2; 2]
[[[500025, 5800015],
  [500035, 5800015]],
 [[500025, 5800005],
  [500035, 5800005]]]
```

The window's origin is the world point of pixel point `(2, 2)`, 20 m east
and 20 m south of the grid's, and its centres are those of the grid's last
two rows and columns.

## Polygons and the cells they cover

A feature is a list of rings, in GeoArrow's layout of one tensor of
points and offsets that cut it into rings and the rings into features (the
preamble of [`polygons.mli`](../lib/polygons.mli)). This square of 24 m in
`grid`'s system starts at the grid's corner:

```ocaml
# let square =
    let pts =
      [| 500000.; 5800040.; 500024.; 5800040.; 500024.; 5800016.;
         500000.; 5800016. |]
    in
    Polygons.v grid.crs (Nx.create Nx.float64 [| 4; 2 |] pts)
      ~rings:[| 0; 4 |] ~features:[| 0; 1 |];;
val square : Polygons.t =
  {Polygons.crs = WGS 84 / UTM zone 33N; points = <abstr>; rings = [|0; 4|];
   features = [|0; 1|]}
```

`Rasterize.centres` gives the cells whose centre lies in the square, and
`Rasterize.touched` also the cells whose inside an edge meets (the Rules
section of [`rasterize.mli`](../lib/rasterize.mli)). Each is a relation,
the features' indices and the cells' flat indices, `row × cols + col` (the
preamble):

```ocaml
# let print_relation (f, c) = Nx.print f; Nx.print c;;
val print_relation : ('a, 'b, 'c) Nx.t * ('d, 'e, 'f) Nx.t -> unit = <fun>
# let () = print_relation (Rasterize.centres grid square);;
int64 [4] [0, 0, 0, 0]
int64 [4] [0, 1, 4, 5]
# let () = print_relation (Rasterize.touched grid square);;
int64 [9] [0, 0, 0, 0, 0, 0, 0, 0, 0]
int64 [9] [0, 1, 2, 4, 5, 6, 8, 9, 10]
```

The square covers the centres of the 2 × 2 cells in its corner, at 5 m and
15 m from its western and northern edges. Its eastern and southern edges,
at 24 m, cross the third row and column, so `touched` adds those 5 cells.
The preamble of [`rasterize.mli`](../lib/rasterize.mli) builds a raster of
owners from a relation.

`Rasterize.areas` gives `touched`'s pairs with the share of each cell the
square covers, the weights of an area-weighted mean, which the
Compositions paragraph of [`rasterize.mli`](../lib/rasterize.mli)
composes:

```ocaml
# let () = let _, _, area = Rasterize.areas grid square in Nx.print area;;
float64 [9]
[1, 1, 0.3999999999999999, 1, 1, 0.3999999999999999, 0.3999999999999999,
 0.3999999999999999, 0.15999999999999992]
```

The third row and column hold 0.4 of a cell, to the rounding of the edges'
pixel points: 24 m is pixel 2.4, which float64 holds as
2.399999999999999911.

## Warping

A warp gives each cell of a destination grid the source's value at the
cell's centre. `Warp.centres` takes the centres into the source's system,
`Warp.pixel` to its pixel points, and a sampler reads the array there (the
preamble of [`warp.mli`](../lib/warp.mli)). Here a source of 2 × 2 cells
of 20 m is read on `grid`, whose cells are 10 m:

```ocaml
# let src =
    Grid.v grid.crs ~rows:2 ~cols:2 ~origin:(500000., 5800040.)
      ~row_step:(0., -20.) ~col_step:(20., 0.);;
val src : Grid.t =
  {Grid.crs = WGS 84 / UTM zone 33N; rows = 2; cols = 2;
   origin = (500000., 5800040.); row_step = (0., -20.); col_step = (20., 0.)}
# let () =
    let x = Nx.create Nx.float64 [| 2; 2 |] [| 0.; 1.; 2.; 3. |] in
    let p = Warp.pixel src (Warp.centres ~at:Nx.Host.on src.crs grid) in
    let near, _ = Warp.nearest src p x in
    let lin, ok = Warp.bilinear src p x in
    Nx.print near;
    Nx.print lin;
    Nx.print ok;;
float64 [4; 4]
[[0, 0, 1, 1],
 [0, 0, 1, 1],
 [2, 2, 3, 3],
 [2, 2, 3, 3]]
float64 [4; 4]
[[0, 0.25, 0.75, 1],
 [0.5, 0.75, 1.25, 1.5],
 [1.5, 1.75, 2.25, 2.5],
 [2, 2.25, 2.75, 3]]
bool [4; 4]
[[true, true, true, true],
 [true, true, true, true],
 [true, true, true, true],
 [true, true, true, true]]
```

Nearest copies each source cell onto the four cells it holds. Bilinear
interpolates between the source's centres, and at the grid's edge it
interpolates along the edge, so every destination cell has data (the
Samplers section of [`warp.mli`](../lib/warp.mli)).

Onto a grid coarser than the source, an area rule reads the source over
each cell's footprint, where a sampler would alias, and an equal or
finer grid takes a sampler (the preamble of
[`warp.mli`](../lib/warp.mli)). `Warp.average` is the mean weighted by each
source cell's exact area in the footprint, and `Warp.mode` the class that
covers the most of it, each with the fraction of the footprint that holds
data (the Area rules section). Each footprint reads a block of `k × k`
source cells, which the call states as `~k`, and `Warp.span` gives the
least `k` for a set of corners, 3 from 10 m to 20 m in one system: a
20 m cell covers 2 × 2 cells of 10 m, and its span also holds the row
and column after them, where its area is 0 (the span section).

## Bands

A band's stored numbers, its DNs, measure `scale · DN + offset`, except
its special values (the preamble of [`band.mli`](../lib/band.mli)).
`Band.decode` gives the values and a mask of the DNs that are not
special:

```ocaml
# let band =
    { Band.name = Some "B04"; common_name = Some "red";
      center_wavelength = Some 0.665; full_width_half_max = Some 0.038;
      unit = None; kind = Measurement; scale = 1e-4; offset = -0.1;
      nodata = Some 0.; saturated = Some 65535. };;
val band : Band.t =
  {Band.name = Some "B04"; common_name = Some "red";
   center_wavelength = Some 0.665; full_width_half_max = Some 0.038;
   unit = None; kind = Band.Measurement; scale = 0.0001; offset = -0.1;
   nodata = Some 0.; saturated = Some 65535.}
# let () =
    let dn = Nx.create Nx.uint16 [| 4 |] [| 0; 1000; 3000; 65535 |] in
    let v, ok = Band.decode band Nx.float32 dn in
    Nx.print v;
    Nx.print ok;
    Nx.print (Band.saturated band dn);;
float32 [4] [0, 0, 0.2, 0]
bool [4] [false, true, true, false]
bool [4] [false, false, false, true]
```

DN 0 is nodata and 65535 saturated. Both are masked and read 0, and
`Band.saturated` tells them apart. The centre wavelength and width are
what a band's source states, in micrometres, each by the source's own
definition (`center_wavelength` in [`band.mli`](../lib/band.mli)). These
come from Microsoft's Planetary Computer for B04, the midpoint and span
of the product metadata's 646 to 684 nm.

## Sentinel-2, Sentinel-1 and Landsat

A Sentinel-2 product, at Level-1C or Level-2A, states its quantification
value and, from processing baseline 04.00, an offset for each band (the
Reflectance section of [`sentinel2.mli`](../lib/sentinel2.mli)).
`Sentinel2.reflectance` makes the band that decodes a band's DNs into
reflectance. [STAC items](05-stac.md) give the DNs' grids and
[Sentinel-2 metadata](06-safe.md) reads these values from a product:

```ocaml
# let metadata =
    { Sentinel2.baseline = "05.10"; quantification = 10000.;
      offsets = [ (Sentinel2.B04, -1000) ]; nodata = 0; saturated = 65535;
      wavelengths = [] };;
val metadata : Sentinel2.metadata =
  {Sentinel2.baseline = "05.10"; quantification = 10000.;
   offsets = [(Sentinel2.B04, -1000)]; nodata = 0; saturated = 65535;
   wavelengths = []}
# let red : Band.t = Result.get_ok (Sentinel2.reflectance metadata B04);;
val red : Band.t =
  {Band.name = Some "B04"; common_name = None; center_wavelength = None;
   full_width_half_max = None; unit = None; kind = Band.Measurement;
   scale = 0.0001; offset = -0.1; nodata = Some 0.; saturated = Some 65535.}
# Sentinel2.reflectance metadata B08;;
- : (Band.t, string) result =
Error "baseline 05.10 states offsets; none is given for B08"
```

The product states no offset for B08, and from baseline 04.00 an offset
must be stated, so its reflectance is an `Error` (`offset` in the
Reflectance section). A scene classification masks the cells of the
classes a model rejects, here those of no data, defective data, and cloud
of medium and high probability, as in the example of
[`jord.mli`](../lib/jord.mli):

```ocaml
# let () =
    let scl = Nx.create Nx.uint8 [| 5 |] [| 0; 4; 8; 9; 6 |] in
    let masked =
      Sentinel2.Scl.
        [ No_data; Saturated_or_defective; Cloud_medium_probability;
          Cloud_high_probability ]
    in
    Nx.print (Sentinel2.Scl.mem masked scl);;
bool [5] [true, false, true, true, false]
```

`Sentinel1` names the polarisations of a radar product, as STAC's
`sar:polarizations` lists them, and states that the RTC products of the
providers it names hold γ⁰ as linear power (the Linear power paragraph of
[`sentinel1.mli`](../lib/sentinel1.mli)):

```ocaml
# Sentinel1.of_name "VH";;
- : Sentinel1.polarisation option = Some Sentinel1.VH
```

`Landsat` states, for each sensor of Collection 2 Level-2, its bands as
`Band.t` values, which `Band.decode` turns into reflectance or temperature,
and the bits of its quality bands (the Quality bits section of
[`landsat.mli`](../lib/landsat.mli)), which are `Bitfield` bands and are
read as bits (the Bitfield bands section of
[`band.mli`](../lib/band.mli)).

## HLS

`Hls` states NASA's HLS v2.0 products, L30 from Landsat and S30 from
Sentinel-2, whose layers are files named by a code that the two products
assign to different bands:

```ocaml
# (Hls.name L30 Nir_narrow, Hls.name S30 Nir_narrow);;
- : string * string = ("B05", "B8A")
```

`Hls.band` makes each layer a `Band.t` with the scale and fill that the
Quantities paragraph of [`hls.mli`](../lib/hls.mli) states, so
`Band.decode` turns a reflective layer's DNs into reflectance:

```ocaml
# let () =
    let dn = Nx.create Nx.int16 [| 3 |] [| -9999; 0; 1234 |] in
    let v, ok = Band.decode (Hls.band S30 Red) Nx.float32 dn in
    Nx.print v;
    Nx.print ok;;
float32 [3] [0, 0, 0.1234]
bool [3] [false, true, true]
```

DN -9999 is the reflective layers' fill, and DN 0 is a reflectance of 0.

`Hls.Fmask` reads the bits of the quality layer, Fmask. This program
keeps the pixels that are neither cloud nor shadow, and masks
`Hls.Fmask.fill` with them, as the Bitfield bands section of
[`band.mli`](../lib/band.mli) says a program must:

```ocaml
# let () =
    let fmask = Nx.create Nx.uint8 [| 4 |] [| 0; 2; 100; 255 |] in
    let cloud = Hls.Fmask.(any [ Cloud; Cloud_shadow ] fmask) in
    let clear =
      Nx.logical_not (Nx.logical_or (Hls.Fmask.fill fmask) cloud)
    in
    Nx.print cloud;
    Nx.print clear;;
bool [4] [false, true, false, true]
bool [4] [true, false, true, false]
```

DN 2 sets the cloud bit, and DN 255 is fill. DN 100 sets the bits of
adjacency, water and a low aerosol level, none of them cloud or shadow.

A product's identifier names its product, its tile and an instant
(the Identity section of [`hls.mli`](../lib/hls.mli) says what the instant
is):

```ocaml
# let id = Result.get_ok (Hls.id "HLS.S30.T21JVF.2026306T134221.v2.0");;
val id : Hls.id =
  {Hls.product = Hls.S30; tile = "21JVF"; stamp = 1793626941.}
# Time.date id.stamp;;
- : int * int * int = (2026, 11, 2)
```

## MODIS

`Modis` states NASA's MODIS Collection 6.1 products MOD13Q1, vegetation
indices at 250 m, and MOD09A1, surface reflectance at 500 m. Each product
is one tile of MODLAND's sinusoidal tiling, and `Modis.tile` is that tile's
grid on `Crs.modis` (the Tiles section of [`modis.mli`](../lib/modis.mli)):

```ocaml
# let () =
    let tile = Modis.tile Mod09a1 ~h:18 ~v:4 in
    let x, y = tile.origin in
    Printf.printf "%d x %d cells from (%.3f, %.3f)\n" tile.rows tile.cols x y;;
2400 x 2400 cells from (0.000, 5559752.598)
```

Tile h18v04's corner lies on the central meridian, four rows of tiles below
the grid's northern edge.

`Modis.band` makes a layer a `Band.t`, and `Modis.valid_range` gives the
range of DNs its guide calls valid (the Bands section of
[`modis.mli`](../lib/modis.mli)). This program keeps the NDVI values that
decode and whose DNs lie in that range:

```ocaml
# let () =
    let dn = Nx.create Nx.int16 [| 3 |] [| -3000; 5000; 12000 |] in
    let ndvi, ok = Band.decode (Modis.band Mod13q1 Ndvi) Nx.float32 dn in
    let lo, hi = Modis.valid_range Mod13q1 Ndvi in
    let in_range =
      Nx.logical_and (Nx.greater_equal_s dn lo) (Nx.less_equal_s dn hi)
    in
    Nx.print ndvi;
    Nx.print (Nx.logical_and ok in_range);;
float32 [3] [0, 0.5, 1.2]
bool [3] [false, true, false]
```

DN -3000 is the indices' fill. DN 12000 decodes to 1.2, and lies past the
range's 10000.

`Modis.State.cloud` reads MOD35's cloud states from MOD09A1's state layer
(`Modis.State` in the Quality section of [`modis.mli`](../lib/modis.mli)).
This program keeps the pixels that are clear or whose state is not set,
and masks `Modis.State.fill`:

```ocaml
# let () =
    let state = Nx.create Nx.uint16 [| 4 |] [| 0; 1; 2; 65535 |] in
    let clear = Modis.State.(cloud [ Clear; Not_set ] state) in
    let fill = Modis.State.fill state in
    Nx.print clear;
    Nx.print (Nx.logical_and clear (Nx.logical_not fill));;
bool [4] [true, false, false, true]
bool [4] [true, false, false, false]
```

DN 65535 is fill in this layer. Its cloud bits read as `Not_set`, so it is
clear until the program masks the fill, as the Bitfield bands section of
[`band.mli`](../lib/band.mli) says a program must.
