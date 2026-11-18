# A Sentinel-2 scene, from catalogue to cloud mask

This page traces one program through jord's libraries. A catalogue's item
gives the files and their grids, the product's metadata gives the band
that decodes the DNs, a warp puts the values on the program's grid, and
the scene classification masks the clouds. It is the life cycle of the
preamble of [`jord.mli`](../lib/jord.mli). jord does no I/O, so the
program fetches the files with its own tools. The item and the metadata
below are the real files that jord's tests hold. The DNs are made up, two
small blocks that stand for the windows `Nx_io.Tiff.window` would read
from the band's and the scene classification's Cloud Optimized GeoTIFFs.

```ocaml
# open Jord;;
```

## The item and its grids

Microsoft's Planetary Computer's search for tile 33UUU on 2024-06-29
returned one item, which jord's tests hold. `Jord_stac.items` reads the
page, and each asset's grid comes from its projection fields (the Grids
section of [`jord_stac.mli`](../lib/stac/jord_stac.mli)):

```ocaml
let item =
  let text =
    In_channel.with_open_bin "../test/stac/mpc_sentinel2_l2a.json"
      In_channel.input_all
  in
  let json = Result.get_ok (Jsont_bytesrw.decode_string Jsont.json text) in
  List.hd (Result.get_ok (Jord_stac.items json))

let grid key =
  Result.get_ok (Option.get (List.assoc key item.Jord_stac.assets).grid)

let red_grid = grid "B04" and scl_grid = grid "SCL"
```

```ocaml
# (red_grid.rows, scl_grid.rows);;
- : int * int = (10980, 5490)
```

The band's grid has 10 m cells and the scene classification's 20 m cells,
both from the tile's corner at (300000, 5900040) in WGS 84 / UTM zone 33N.

## The band that decodes the DNs

The item states no scale, offset or nodata for its bands. The product's
metadata, the asset `product-metadata`, states them, and
`Sentinel2.reflectance` makes the band (the Reflectance section of
[`sentinel2.mli`](../lib/sentinel2.mli)):

```ocaml
let red_band =
  let xml =
    In_channel.with_open_bin "../test/safe/mtd_n0510.xml" In_channel.input_all
  in
  let metadata = Result.get_ok (Jord_safe.l2a xml) in
  Result.get_ok (Sentinel2.reflectance metadata B04)
```

```ocaml
# (red_band.scale, red_band.offset, red_band.nodata);;
- : float * float * float option = (0.0001, -0.1, Some 0.)
```

## The program's grid and the windows it reads

The program's grid has 3 × 3 cells of 10 m in the tile's corner, shifted
half a cell east and south, so its centres fall on the corners between the
band's cells. Both source grids are in the tile's system, so one `centres`
serves both, and `Warp.window` gives the block of each source that a
sampler reads there (`window` in [`warp.mli`](../lib/warp.mli)). A grid of
equal or finer resolution takes a sampler (the preamble of
[`warp.mli`](../lib/warp.mli)), bilinear for the reflectance and nearest
for the classes.

```ocaml
let dst =
  Grid.v red_grid.crs ~rows:3 ~cols:3 ~origin:(300005., 5900035.)
    ~row_step:(0., -10.) ~col_step:(10., 0.)

let centres = Warp.centres ~at:Nx.Host.on red_grid.crs dst
let p = Warp.pixel red_grid centres and p' = Warp.pixel scl_grid centres
let w = Warp.window Bilinear red_grid p
and w' = Warp.window Nearest scl_grid p'
```

```ocaml
# (w, w');;
- : (int * int * int * int) * (int * int * int * int) =
((0, 0, 4, 4), (0, 0, 2, 2))
```

The program reads those blocks, the band's first 4 × 4 cells and the
scene classification's first 2 × 2, with `Nx_io.Tiff.window` (the
[GeoTIFF tags](02-geotiff.md) page reads a window of a real file). The DNs
are made up for this page:

```ocaml
let dn =
  Nx.create Nx.uint16 [| 4; 4 |]
    [| 0; 2100; 2200; 2300; 2000; 2100; 2200; 2300;
       2000; 2100; 2200; 2300; 2000; 2100; 2200; 2300 |]

and scl = Nx.create Nx.uint8 [| 2; 2 |] [| 4; 4; 9; 4 |]
```

In these made-up blocks, the band's first DN is 0, NODATA, and the scene
classification's third cell, the lower left 20 m cell, is class 9, cloud
of high probability.

## Onto the program's grid

Each sampler reads its block with `~row ~col`, the block's first cell, at
the whole source's grid and pixel points, which gives what it gives on the
whole source, bit for bit (`window` in [`warp.mli`](../lib/warp.mli)).
Both blocks start at (0, 0) here. Bilinear carries the decode's mask,
nearest gives where the scene classification has a cell, and the program
joins them:

```ocaml
let red, ok =
  let row, col, _, _ = w and v, valid = Band.decode red_band Nx.float32 dn in
  Warp.bilinear ~row ~col ~valid red_grid p v

let classes, seen =
  let row, col, _, _ = w' in
  Warp.nearest ~row ~col scl_grid p' scl

let masked =
  Sentinel2.Scl.(
    mem
      [ No_data; Saturated_or_defective; Cloud_medium_probability;
        Cloud_high_probability ])
    classes
```

```ocaml
# let () =
    Nx.print red;
    Nx.print Nx.(logical_and (logical_and ok seen) (logical_not masked));;
float32 [3; 3]
[[0.10666667, 0.114999995, 0.125],
 [0.105000004, 0.114999995, 0.125],
 [0.105000004, 0.114999995, 0.125]]
bool [3; 3]
[[true, true, true],
 [false, true, true],
 [false, true, true]]
```

The reflectance is `(DN - 1000) / 10000`, and each value is the mean of
the four band cells around the centre, which lies on their common corner.
The first leaves out the NODATA cell, so it is the mean of the other
three, 0.1067 (`bilinear` in [`warp.mli`](../lib/warp.mli)). The program
masks the classes of no data, of saturated or defective cells and of
cloud of medium and high probability. The mask is false at the two cells
whose centres nearest reads from the scene classification's lower left
20 m cell, class 9. One centre lies inside that cell, and one on its top
edge, where nearest reads the cell below the edge (`nearest` in
[`warp.mli`](../lib/warp.mli)). The mask is true elsewhere, where the
band has data and the class is none of those. The program keeps the
reflectance where the mask is true. A model chooses which classes it
masks (the preamble of [`band.mli`](../lib/band.mli)).

## A coarser grid

Onto a grid coarser than the band's, the program takes an area rule (the
Area rules section of [`warp.mli`](../lib/warp.mli)). Its grid here is
one cell of 40 m in the tile's corner, whose corners `Warp.corners`
gives. `Warp.area_window` gives the block of the band's cells the rule
reads, and `Warp.span` its `k` (the Windows and span sections), here the
first 5 × 5 cells and 5, which the scene classification's first 3 × 3
cells cover. The made-up blocks hold the same NODATA cell and one cell of
cloud:

```ocaml
let cell =
  Grid.v red_grid.crs ~rows:1 ~cols:1 ~origin:(300000., 5900040.)
    ~row_step:(0., -40.) ~col_step:(40., 0.)

let q = Warp.pixel red_grid (Warp.corners ~at:Nx.Host.on red_grid.crs cell)
let row, col, rows, cols = Warp.area_window red_grid q
let k = Warp.span red_grid q
let block = Grid.window ~row ~col ~rows ~cols red_grid

let dn =
  Nx.init Nx.uint16 [| 5; 5 |] (fun a ->
      if a.(0) = 0 && a.(1) = 0 then 0 else 2000 + (100 * a.(1)))

and scl = Nx.create Nx.uint8 [| 3; 3 |] [| 4; 4; 4; 9; 4; 4; 4; 4; 4 |]
```

The program takes the classes onto the band's cells with nearest, and
folds the cells they mask, and the cells the scene classification does not
cover, into the decode's mask, so the fraction the rule gives is the share
of the cell that is clear and has data:

```ocaml
let classes, seen =
  Warp.nearest scl_grid
    (Warp.pixel scl_grid (Warp.centres ~at:Nx.Host.on scl_grid.crs block))
    scl

let clear =
  Nx.logical_and seen
    (Nx.logical_not
       Sentinel2.Scl.(
         mem
           [ No_data; Saturated_or_defective; Cloud_medium_probability;
             Cloud_high_probability ]
           classes))
```

```ocaml
# let () =
    let v, valid = Band.decode red_band Nx.float32 dn in
    let red, fraction =
      Warp.average ~k ~row ~col ~valid:(Nx.logical_and valid clear) red_grid
        q v
    in
    let modal, _ = Warp.mode ~k ~row ~col red_grid q classes in
    Nx.print red;
    Nx.print fraction;
    Nx.print modal;;
float32 [1; 1]
[[0.12]]
float64 [1; 1]
[[0.6875]]
uint8 [1; 1]
[[4]]
```

The rule takes the band's whole grid and the corners as they are, and the
block's first cell as `~row` and `~col`, here (0, 0). The fraction,
0.6875, is the 11 of the cell's 16 band cells that are clear and have
data, and the value is their mean, the reflectance of a DN of 2200.
`Warp.mode` of the classes gives 4, vegetation, since cloud covers a
quarter of the cell, so the program masks the clouds through `valid`
before the rule, and keeps the cells whose fraction passes a threshold of
its own.

## Onto another zone

A program whose grid lies in another system warps the same way, and
`Warp.corners` takes the corners through the datum's longitude and
latitude. Its grid here is one cell of 40 m in zone 32, near the tile's
corner. The rule's block comes from `Warp.area_window`, and the scene
classification's window is the block nearest reads at the band block's
centres:

```ocaml
let zone32 = Crs.utm ~zone:32 ~north:true

let cell32 =
  let o = Nx.create Nx.float64 [| 2 |] [| 300060.; 5899980. |] in
  let o = Crs.transform ~src:red_grid.crs ~dst:zone32 o in
  let at i = Float.round (Nx.item [ i ] o) in
  Grid.v zone32 ~rows:1 ~cols:1 ~origin:(at 0, at 1) ~row_step:(0., -40.)
    ~col_step:(40., 0.)

let q = Warp.pixel red_grid (Warp.corners ~at:Nx.Host.on red_grid.crs cell32)

let ((row, col, rows, cols) as b) = Warp.area_window red_grid q
let block = Grid.window ~row ~col ~rows ~cols red_grid
let ps = Warp.pixel scl_grid (Warp.centres ~at:Nx.Host.on scl_grid.crs block)
let ((srow, scol, srows, scols) as b') = Warp.window Nearest scl_grid ps
```

```ocaml
# (b, b');;
- : (int * int * int * int) * (int * int * int * int) =
((5, 5, 6, 5), (2, 2, 4, 3))
```

The program reads those blocks. The made-up DNs are 2000 plus 100 a
column, and the scene classification's cell (1, 1) of its window is
cloud:

```ocaml
let dn =
  Nx.init Nx.uint16 [| rows; cols |] (fun a -> 2000 + (100 * a.(1)))

and scl =
  Nx.init Nx.uint8 [| srows; scols |] (fun a ->
      if a.(0) = 1 && a.(1) = 1 then 9 else 4)
```

```ocaml
# let () =
    let classes, seen = Warp.nearest ~row:srow ~col:scol scl_grid ps scl in
    let masked =
      Sentinel2.Scl.(
        mem
          [ No_data; Saturated_or_defective; Cloud_medium_probability;
            Cloud_high_probability ]
          classes)
    in
    let clear = Nx.logical_and seen (Nx.logical_not masked) in
    let v, valid = Band.decode red_band Nx.float32 dn in
    let red, fraction =
      Warp.average ~k:(Warp.span red_grid q) ~row ~col
        ~valid:(Nx.logical_and valid clear) red_grid q v
    in
    Nx.print red;
    Nx.print fraction;;
float32 [1; 1]
[[0.12533192]]
float64 [1; 1]
[[0.7594922127036363]]
```

The band's block starts at cell (5, 5), which the rule takes as `~row`
and `~col` with the band's whole grid and corners, and the scene
classification's window starts at (2, 2), which nearest takes the same
way with the scene classification's whole grid and pixel points (the
Windows section of [`warp.mli`](../lib/warp.mli)). A block that leaves
out a cell a rule reads raises. The fraction, 0.76, is the share of the
cell that is clear and has data, and the cloud's 20 m cell covers the
rest. The value is the mean over that share.
