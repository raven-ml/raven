# Morphology

This page builds the morphological operators that mynd leaves to
composition over `Mynd.erode`, `Mynd.dilate`, `Mynd.components` and
`Mynd.reconstruct`: footprint shapes, openings and closings, top-hats,
the morphological gradient, hit-or-miss, hole filling, clearing the
border, areas and small objects. The rules each call follows, the
footprint, the reads at the border and the adjunction, are in the
Morphology and the Selection sections of [`mynd.mli`](../lib/mynd.mli).

```ocaml
let img =
  Nx.create Nx.float64 [| 7; 9; 1 |]
    (Array.init 63 (fun k -> Float.of_int ((k * 29) mod 47) /. 46.))

let floats x = Nx.to_array (Nx.cast Nx.float64 x)
let bools m = Nx.to_array m

(* [mask rows] is the mask drawn by strings of ['#'] and ['.']. *)
let mask rows =
  let h = List.length rows and w = String.length (List.hd rows) in
  let a = Array.of_list rows in
  Nx.init Nx.bool [| h; w; 1 |] (fun i -> a.(i.(0)).[i.(1)] = '#')

(* [all le a b] is [le] at every element of two float images. *)
let all le a b = Array.for_all2 le (floats a) (floats b)
```

## Footprints

A footprint is a host boolean array whose `true` taps select a window's
pixels, its tap `(kh / 2, kw / 2)` on the pixel. Shapes are predicates on
the offset `(dy, dx)` from that tap:

```ocaml
let shape r f =
  let k = (2 * r) + 1 in
  Nx.init Nx.bool [| k; k |] (fun i -> f (i.(0) - r) (i.(1) - r))

let rectangle kh kw = Nx.ones Nx.bool [| kh; kw |]
let cross r = shape r (fun dy dx -> dy = 0 || dx = 0)
let diamond r = shape r (fun dy dx -> abs dy + abs dx <= r)
let disk r = shape r (fun dy dx -> (dy * dy) + (dx * dx) <= r * r)

let count fp = Array.fold_left (fun n b -> if b then n + 1 else n) 0 (bools fp)

let () = assert (List.map (fun r -> count (disk r)) [ 1; 2; 3; 4; 5 ] = [ 5; 13; 29; 49; 81 ])
```

A footprint of an even extent, such as `rectangle 2 2`, has its origin
tap at `(1, 1)`: erosion reads the pixel and those above and to its left,
and dilation, which reads the window reflected about the pixel, those
below and to its right.

## Openings and closings

The opening `dilate fp (erode fp x)` removes the bright details that the
footprint does not fit in, and the closing `erode fp (dilate fp x)` the
dark ones. Both use the same footprint in their two steps: `dilate`
already reads the reflected window, which is what makes the two adjoint,
so a footprint flipped by hand for the second step breaks the laws
below for every footprint that is not symmetric about its origin.

```ocaml
let opening fp x = Mynd.dilate fp (Mynd.erode fp x)
let closing fp x = Mynd.erode fp (Mynd.dilate fp x)

let () =
  List.iter
    (fun fp ->
      let o = opening fp img and c = closing fp img in
      assert (all ( <= ) o img && all ( >= ) c img);
      assert (floats (opening fp o) = floats o);
      assert (floats (closing fp c) = floats c))
    [ rectangle 2 2; rectangle 1 4; cross 1; disk 2; Nx.create Nx.bool [| 2; 3 |] [| true; false; false; true; true; true |] ]
```

The opening is at most the image and the closing at least it, and a
second application changes nothing, at every pixel, the border included,
because no step reads past the border.

## Top-hats and the morphological gradient

The white top-hat `x - opening x` keeps the bright details the footprint
does not fit in, and the black top-hat `closing x - x` the dark ones;
both are at least `0` by the laws above. The morphological gradient
`dilate fp x - erode fp x` is the range of each window; where the
footprint holds its origin both terms read the pixel, so it is at least
`0`:

```ocaml
let white fp x = Nx.sub x (opening fp x)
let black fp x = Nx.sub (closing fp x) x
let gradient fp x = Nx.sub (Mynd.dilate fp x) (Mynd.erode fp x)

let () =
  let fp = disk 1 in
  let zero = Nx.zeros_like img in
  assert (all ( >= ) (white fp img) zero);
  assert (all ( >= ) (black fp img) zero);
  assert (all ( >= ) (gradient fp img) zero)
```

## Hit-or-miss

Hit-or-miss finds a pattern of foreground pixels `fg` and background
pixels `bg`, two footprints of one shape that share no tap. It erodes the
mask by `fg` and its complement by `bg`, so it uses erosions alone:

```ocaml
let hit_or_miss fg bg m = Nx.logical_and (Mynd.erode fg m) (Mynd.erode bg (Nx.logical_not m))

(* Isolated pixels: the pixel itself set, its eight neighbours clear. *)
let () =
  let centre = shape 1 (fun dy dx -> dy = 0 && dx = 0) in
  let ring = shape 1 (fun dy dx -> dy <> 0 || dx <> 0) in
  let m = mask [ "......"; ".#..##"; "....##"; "......" ] in
  let found = hit_or_miss centre ring m in
  assert (bools found = bools (mask [ "......"; ".#...."; "......"; "......" ]))
```

Past the border no tap is read, so a pixel at the border passes on the
neighbours it has.

## Holes

A hole is a component of the background that reaches no border pixel.
`reconstruct` grows the background from the frame of border pixels; what
it does not reach is a hole, and the mask with its holes filled is the
complement of what it reaches. The background's connectivity pairs with
the foreground's: a foreground joined through corners (`` `Eight ``)
encloses the background components joined only through edges
(`` `Four ``).

```ocaml
(* The frame is a predicate on the index, so that masks of one row or
   column are all frame. *)
let frame m =
  let h = Nx.dim (-3) m and w = Nx.dim (-2) m in
  Nx.init Nx.bool (Nx.shape m) (fun i ->
      let r = Array.length i in
      i.(r - 3) = 0 || i.(r - 3) = h - 1 || i.(r - 2) = 0 || i.(r - 2) = w - 1)

let fill_holes background m =
  Nx.logical_not (Mynd.reconstruct background ~marker:(frame m) (Nx.logical_not m))

let () =
  let ring = mask [ "......."; "..###.."; ".#...#."; "..###.."; "......." ] in
  let filled = mask [ "......."; "..###.."; ".#####."; "..###.."; "......." ] in
  assert (bools (fill_holes `Four ring) = bools filled);
  assert (bools (fill_holes `Eight ring) = bools ring)
```

The ring's corners touch only diagonally, so under an 8-connected
background its inside leaks out through them, and it holds no hole.

Clearing the objects that touch the border is reconstruction from the
frame too, removed from the mask:

```ocaml
let clear_border c m = Nx.logical_and m (Nx.logical_not (Mynd.reconstruct c ~marker:(frame m) m))

let () =
  let m = mask [ "##...."; "....#."; "....#."; "......" ] in
  assert (bools (clear_border `Eight m) = bools (mask [ "......"; "....#."; "....#."; "......" ]))
```

## Areas and small objects

`Mynd.components` labels each mask in its own indices: a component's
label is the least flat index `i w + j` of its pixels. Counting each
label's pixels by a scatter along the last axis of the masks flattened
to `[…; h w]` gives every mask its own counts, in a batch too, and a
gather at the labels reads each pixel's area back:

```ocaml
let areas c m =
  let s = Nx.shape m in
  let r = Array.length s in
  let flat x = Nx.reshape (Array.append (Array.sub s 0 (r - 3)) [| s.(r - 3) * s.(r - 2) |]) x in
  let l = flat (Mynd.components c m) in
  let ones = Nx.ones_like l in
  let counts = Nx.scatter ~combine:Add ~axis:(-1) l ones (Nx.zeros_like l) in
  Nx.reshape s (Nx.take_along_axis ~axis:(-1) l counts)

let () =
  let a = mask [ "##.."; "##.#"; "...." ] and b = mask [ "#..."; "...."; "..##" ] in
  let both = Nx.stack [ a; b ] in
  assert (floats (areas `Eight both) = Array.append (floats (areas `Eight a)) (floats (areas `Eight b)));
  assert (floats (areas `Eight a) = [| 4.; 4.; 0.; 0.; 4.; 4.; 0.; 1.; 0.; 0.; 0.; 0. |])
```

A label of `-1`, a pixel outside every component, writes nothing and
reads `0`.

Removing the components of at most `n` pixels, and filling the holes of
at most `n` pixels, cleans a predicted mask of specks. A hole here is any
component of the background, the ones that touch the border included, so
that a gap at the edge of the image fills too; take the holes of the
previous section to keep those.

```ocaml
let remove_small c n m = Nx.logical_and m (Nx.greater_s (areas c m) (Int64.of_int n))

let fill_small c n m =
  let bg = Nx.logical_not m in
  Nx.logical_or m (Nx.logical_and bg (Nx.less_equal_s (areas c bg) (Int64.of_int n)))

let () =
  let m = mask [ "#....###"; ".....#.#"; "..#..###"; "........" ] in
  assert (bools (remove_small `Eight 1 m) = bools (mask [ ".....###"; ".....#.#"; ".....###"; "........" ]));
  assert (bools (fill_small `Eight 1 m) = bools (mask [ "#....###"; ".....###"; "..#..###"; "........" ]))
```

Each reads the areas of the mask it is given, so where a speck sits in a
small hole the order of the two decides the result. Computing both from
the same mask and joining them avoids the choice.

## Opening by reconstruction

An opening removes the objects the footprint does not fit in and also
erodes the corners of those it keeps. Reconstructing the mask from its
erosion keeps every object that survives the erosion, whole:

```ocaml
let open_by_reconstruction c fp m = Mynd.reconstruct c ~marker:(Mynd.erode fp m) m

let () =
  let m = mask [ "###....."; "###..#.."; "####...."; "........" ] in
  assert (bools (open_by_reconstruction `Eight (rectangle 3 3) m)
          = bools (mask [ "###....."; "###....."; "####...."; "........" ]))
```
