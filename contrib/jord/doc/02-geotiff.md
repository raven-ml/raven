# GeoTIFF tags

A GeoTIFF file is a TIFF file whose tags place its image on the Earth.
nx.io reads and writes the file and its tags (`Nx_io.load_tiff` and
`Nx_io.save_tiff`), and `jord.geotiff` turns tags into a grid and a grid
into tags ([`jord_geotiff.mli`](../lib/geotiff/jord_geotiff.mli)). Add
`jord.geotiff` and `nx.io` to an executable's `libraries`.

```ocaml
# open Jord;;
```

## Writing and reading a grid

This program writes a mask of 2 × 3 cells with its grid and a nodata value,
then reads them back from the file:

```ocaml
# let () =
    let grid =
      Grid.v (Crs.utm ~zone:33 ~north:true) ~rows:2 ~cols:3
        ~origin:(500000., 5800020.) ~row_step:(0., -10.) ~col_step:(10., 0.)
    in
    let mask = Nx.zeros Nx.uint8 [| 2; 3; 1 |] in
    let written = Jord_geotiff.tags ~nodata:255. grid in
    Nx_io.save_tiff ~tags:written "mask.tiff" mask;
    let tags = Nx_io.Tiff.tags (List.hd (Nx_io.load_tiff "mask.tiff")) in
    let back = Result.get_ok (Jord_geotiff.grid tags) in
    Printf.printf "same grid: %b\n" (Grid.equal grid back);
    match Jord_geotiff.nodata tags with
    | Ok (Some v) -> Printf.printf "nodata: %g\n" v
    | Ok None -> print_endline "no nodata"
    | Error e -> print_endline e;;
same grid: true
nodata: 255
```

`Jord_geotiff.tags` gives the north-up grid as a pixel scale and one
tiepoint, its system as an EPSG code, and the nodata value as GDAL's tag
(`tags` in [`jord_geotiff.mli`](../lib/geotiff/jord_geotiff.mli)).
`Nx_io.save_tiff` adds the image's width and length, and from them
`Jord_geotiff.grid` gives back the same grid, compared exactly by
`Grid.equal`.

## Reading a window

A program that warps part of a large image reads only the block of it
that the warp reads. `Warp.window` gives that block, `Nx_io.Tiff.window`
reads it from the byte ranges it names, and the sampler reads the block
with `~row ~col`, its first cell, at the whole image's pixel points, which
gives the values it gives on the whole image, bit for bit (`window` in
[`warp.mli`](../lib/warp.mli)). Here the bytes come from a COG of 51 × 75
cells that jord's tests hold, through a partial file, as a program would
fetch byte ranges of a remote one:

```ocaml
let bytes =
  In_channel.with_open_bin "../test/geotiff/cog_overviews.tif"
    In_channel.input_all

let fetched = ref 0

let fill p rs =
  List.fold_left
    (fun p (at, n) ->
      fetched := !fetched + n;
      Nx_io.Partial.add p ~at (String.sub bytes at n))
    p rs

let rec ifd p =
  match Nx_io.decode_tiff p with
  | Error rs -> ifd (fill p rs)
  | Ok ds -> (p, List.hd ds)

let rec block p ~row ~col ~rows ~cols =
  let p, d = ifd p in
  match Nx_io.Tiff.window d ~row ~col ~rows ~cols with
  | Ok x -> Nx.unpack Nx.uint8 x
  | Error rs -> block (fill p rs) ~row ~col ~rows ~cols

let p, d =
  ifd (Nx_io.Partial.v ~name:"cog_overviews.tif" ~length:(String.length bytes))

let src = Result.get_ok (Jord_geotiff.grid (Nx_io.Tiff.tags d))
```

The program's grid is 5 × 5 cells of 10 m inside the image, off its
lattice, so each point lies inside a cell. The values are decoded to
float64 and compared by their bits:

```ocaml
# let () =
    let dst =
      Grid.v src.crs ~rows:5 ~cols:5 ~origin:(500103.7, 5299896.9)
        ~row_step:(0., -10.) ~col_step:(10., 0.)
    in
    let q = Warp.pixel src (Warp.centres ~at:Nx.Host.on src.crs dst) in
    let row, col, rows, cols = Warp.window Bilinear src q in
    let x' = Nx.cast Nx.float64 (block p ~row ~col ~rows ~cols) in
    let v', ok' = Warp.bilinear ~row ~col src q x' in
    let whole = Nx_io.load_tiff "../test/geotiff/cog_overviews.tif" in
    let x = Nx.unpack Nx.uint8 (Nx_io.Tiff.samples (List.hd whole)) in
    let v, ok = Warp.bilinear src q (Nx.cast Nx.float64 x) in
    let bits a = Array.map Int64.bits_of_float (Nx.to_array a) in
    Printf.printf "block (%d, %d) of %d x %d, %d of %d bytes\n" row col rows
      cols !fetched (String.length bytes);
    Printf.printf "the same bits: %b, the same mask: %b\n" (bits v = bits v')
      (Nx.to_array ok = Nx.to_array ok');;
block (10, 10) of 6 x 6, 1355 of 5898 bytes
the same bits: true, the same mask: true
```

The values and the mask are the whole image's, from 1355 of the file's
5898 bytes. Only the index of a cell in the block depends on the block,
and a point that reads a cell outside it raises, so a short block never
gives a wrong value (the Windows section of
[`warp.mli`](../lib/warp.mli)).

## Onto another zone

The image lies in UTM zone 33, at 15° E. A program whose grid is in zone
34 warps it across the systems, and `Warp.centres` takes each point
through the datum's longitude and latitude (`centres` in
[`warp.mli`](../lib/warp.mli)). The grid here has cells of 60 m, six
times the image's, so an area rule reads each cell's footprint, whose
corners `Warp.corners` gives (the Area rules section). Its origin is
the image's top left corner in zone 34, rounded to the metre and moved
one cell inside, and its last row passes the image's lower edge:

```ocaml
let x =
  let whole = Nx_io.load_tiff "../test/geotiff/cog_overviews.tif" in
  let x = Nx.unpack Nx.uint8 (Nx_io.Tiff.samples (List.hd whole)) in
  Nx.cast Nx.float64 (Nx.reshape [| 51; 75 |] x)

let zone34 = Crs.utm ~zone:34 ~north:true

let ox, oy =
  let x0, y0 = src.origin in
  let o = Nx.create Nx.float64 [| 2 |] [| x0; y0 |] in
  let o = Crs.transform ~src:src.crs ~dst:zone34 o in
  let at i = Float.round (Nx.item [ i ] o) in
  (at 0 +. 60., at 1 -. 60.)

let dst =
  Grid.v zone34 ~rows:9 ~cols:4 ~origin:(ox, oy) ~row_step:(0., -60.)
    ~col_step:(60., 0.)

let q = Warp.pixel src (Warp.corners ~at:Nx.Host.on src.crs dst)
```

```ocaml
# let () =
    let v, fraction = Warp.average ~k:(Warp.span src q) src q x in
    let cents a = Nx.div_s (Nx.round (Nx.mul_s a 100.)) 100. in
    Printf.printf "origin (%.0f, %.0f)\n" ox oy;
    Nx.print (Nx.round v);
    Nx.print (cents fraction);;
origin (51302, 5317392)
float64 [9; 4]
[[110, 120, 130, 133],
 [142, 141, 131, 115],
 [98, 100, 110, 121],
 [123, 133, 143, 154],
 [118, 124, 134, 131],
 [124, 115, 125, 134],
 [135, 129, 120, 125],
 [128, 92, 108, 118],
 [0, 0, 0, 0]]
float64 [9; 4]
[[1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [1, 1, 1, 1],
 [0.67, 0.75, 0.82, 0.9],
 [0, 0, 0, 0]]
```

The fraction is the share of each footprint over the image's cells. It
is 1 inside the image, rises from 0.67 to 0.9 along the row that crosses
its lower edge, which zone 34's grid meets at an angle, and is 0 on the
row past it, whose value is 0 (`average` in
[`warp.mli`](../lib/warp.mli)). A program keeps the cells whose
fraction passes a threshold of its own.

The span `k` is the side of the square of cells each footprint reads,
and `Warp.span` gives the least that holds every footprint's cells, from
the corners and after the rule's own placement (the span section of
[`warp.mli`](../lib/warp.mli)):

```ocaml
# Warp.span src q;;
- : int = 8
```

The widest footprint spans 8 cells, for a cell six times the image's
turned by the 4.5° between the two zones' grids, so `~k:7` raises.

## An area rule on a window

`Warp.area_window` gives the block of the image an area rule reads at the
corners, and the rule runs on that block with its first cell as `~row`
and `~col`, at the whole image's grid and corners, which gives the whole
image's bits (the Windows section of [`warp.mli`](../lib/warp.mli)):

```ocaml
# let () =
    let row, col, rows, cols = Warp.area_window src q in
    fetched := 0;
    let x' = block p ~row ~col ~rows ~cols in
    let x' = Nx.cast Nx.float64 (Nx.reshape [| rows; cols |] x') in
    let k = Warp.span src q in
    let v', f' = Warp.average ~k ~row ~col src q x' in
    let v, f = Warp.average ~k src q x in
    let bits a = Array.map Int64.bits_of_float (Nx.to_array a) in
    Printf.printf "block (%d, %d) of %d x %d, %d of %d bytes\n" row col rows
      cols !fetched (String.length bytes);
    Printf.printf "the same bits: %b, the same fraction: %b\n"
      (bits v = bits v') (bits f = bits f');;
block (3, 6) of 48 x 29, 2079 of 5898 bytes
the same bits: true, the same fraction: true
```

A block that leaves out a cell a footprint reads raises, so a short block
never gives a wrong value.

## Across the antimeridian

On a source whose columns span 360°, an area rule reads the columns on
both sides of its west edge, and `Warp.area_window`'s block runs past the
last column and on from column 0 (`area_window` in
[`warp.mli`](../lib/warp.mli)). A file's window lies in its raster, so
the program reads such a block in two parts and joins them. Here a global
grid of 1° cells from 180° W is written as a tiled file, and a grid of
2° cells from 178° E to 178° W is averaged from it:

```ocaml
# let () =
    let world =
      Grid.v Crs.lonlat ~rows:180 ~cols:360 ~origin:(-180., 90.)
        ~row_step:(0., -1.) ~col_step:(1., 0.)
    in
    let x =
      Nx.init Nx.uint8 [| 180; 360; 1 |] (fun a ->
          (a.(0) + (3 * a.(1))) mod 251)
    in
    let tags = Jord_geotiff.tags world in
    Nx_io.save_tiff ~tile:(32, 32) ~tags "world.tiff" x;
    let d = List.hd (Nx_io.load_tiff "world.tiff") in
    let dst =
      Grid.v Crs.lonlat ~rows:2 ~cols:2 ~origin:(178., 2.) ~row_step:(0., -2.)
        ~col_step:(2., 0.)
    in
    let q = Warp.pixel world (Warp.corners ~at:Nx.Host.on world.crs dst) in
    let k = Warp.span world q in
    let row, col, rows, cols = Warp.area_window world q in
    let part col cols =
      match Nx_io.Tiff.window d ~row ~col ~rows ~cols with
      | Ok x -> Nx.cast Nx.float64 (Nx.unpack Nx.uint8 x)
      | Error _ -> assert false
    in
    let x' =
      if col + cols <= world.cols then part col cols
      else
        Nx.concatenate ~axis:1
          [ part col (world.cols - col); part 0 (col + cols - world.cols) ]
    in
    let v', _ = Warp.average ~k ~row ~col world q x' in
    let v, _ = Warp.average ~k world q (Nx.cast Nx.float64 x) in
    let bits a = Array.map Int64.bits_of_float (Nx.to_array a) in
    Printf.printf "k %d, block (%d, %d) of %d x %d\n" k row col rows cols;
    Printf.printf "the same bits: %b\n" (bits v = bits v');;
k 3, block (88, 358) of 5 x 5
the same bits: true
```

The block is columns 358 and 359, the last two, then 0 to 2, read as two
windows of the file. The footprints cover columns 358 to 1 and rows 88
to 91. Their greatest corners lie on the lines of cell edges after column
1 and row 91, so their spans hold column 2 and row 92, where their area is
0, and the rule reads them (the span section of
[`warp.mli`](../lib/warp.mli)). The least block without the wrap would be
the raster's whole width.

## Tags that state no grid

`Jord_geotiff.grid` refuses tags that do not state a grid jord can read,
such as an image with no georeference at all:

```ocaml
# let () =
    Nx_io.save_tiff "plain.tiff" (Nx.zeros Nx.uint8 [| 2; 3; 1 |]);
    let tags = Nx_io.Tiff.tags (List.hd (Nx_io.load_tiff "plain.tiff")) in
    match Jord_geotiff.grid tags with
    | Ok _ -> print_endline "a grid"
    | Error e -> print_endline e;;
GeoKeyDirectory (34735) is missing
```

The Grids section of [`jord_geotiff.mli`](../lib/geotiff/jord_geotiff.mli)
lists the tags and keys a grid is read from.
