# Moving annotations with an image

An image moved by a transform carries its boxes, masks and keypoints
with it. This page moves each kind by one transform, writes the array
operations of augmentation as transforms, decides which boxes survive,
draws transforms for batches and videos, builds two training recipes,
fits a transform to four point pairs and trains a box with
`Mynd.Box.giou`. The rules it uses are in [`mynd.mli`](../lib/mynd.mli):
transforms and the warp in the preambles of `Mynd.Transform` and
`Mynd.Sample`, boxes in the preamble of `Mynd.Box`, and the pairing of
arrays in its section Batches.

```ocaml
let f64 shape a = Nx.create Nx.float64 shape a
let floats x = Nx.to_array (Nx.cast Nx.float64 x)
let mat a = f64 [| 3; 3 |] a
let near a b = Float.abs (a -. b) < 1e-9

(* [reads m (h, w)] is the positions a warp by [m] reads for an output of
   [h] rows and [w] columns: [m]'s inverse at the output's centres. *)
let reads m (h, w) =
  Mynd.Transform.apply (Nx.inv m) (Mynd.Sample.centres Nx.float64 (h, w))
```

## One transform for every kind

A transform `m` maps the positions of the input image to those of the
output. The warp reads the input at `m`'s inverse of the output's
centres, and the annotations move by `m` itself. An image is read
through a kernel with `Mynd.Sample.at`, a mask with
`Mynd.Sample.nearest`, which keeps its values `true` and `false`, a box
with `Mynd.Box.transform` and keypoints with `Mynd.Transform.apply`.

The image below is 12 rows by 16 columns. Its mask is a block of pixels
whose box is `4 3 10 7`, and it has two keypoints:

```ocaml
let h, w = (12, 16)

let image =
  Nx.init Nx.float64 [| h; w; 3 |] (fun i ->
      Float.of_int ((i.(0) * w) + i.(1) + i.(2)) /. 200.)

let mask =
  Nx.init Nx.bool [| h; w; 1 |] (fun i ->
      i.(0) >= 3 && i.(0) < 7 && i.(1) >= 4 && i.(1) < 10)

let box = f64 [| 1; 4 |] [| 4.; 3.; 10.; 7. |]
let keypoints = f64 [| 2; 2 |] [| 5.5; 4.5; 9.; 6. |]
let () = assert (floats (Mynd.Box.of_mask Nx.float64 mask) = floats box)
```

`Mynd.Transform.rotation` turns about the origin, the image's top-left
corner. A turn about the image's centre moves the centre to the origin,
turns and moves it back:

```ocaml
let about (cx, cy) m =
  let t x y = Mynd.Transform.translation (f64 [| 2 |] [| x; y |]) in
  Nx.matmul (t cx cy) (Nx.matmul m (t (-.cx) (-.cy)))

let turn a = Mynd.Transform.rotation (Nx.scalar Nx.float64 a)
let m = about (8., 6.) (turn (Float.pi /. 6.))
let at = reads m (h, w)
let image' = Mynd.Sample.at ~edge:`Zero `Triangle image at
let mask' = Mynd.Sample.nearest ~edge:`Zero mask at
let box' = Mynd.Box.transform m box
let keypoints' = Mynd.Transform.apply m keypoints
```

A pixel of the moved mask is `true` when its centre's preimage lies in
the mask, so its centre lies in the moved box, and its cell within half a
pixel of it. The moved mask's box therefore lies in the moved box grown
by half a pixel on each side:

```ocaml
let () =
  let b = floats (Mynd.Box.of_mask Nx.float64 mask') and t = floats box' in
  assert (b.(0) >= t.(0) -. 0.5 && b.(1) >= t.(1) -. 0.5);
  assert (b.(2) <= t.(2) +. 0.5 && b.(3) <= t.(3) +. 0.5)
```

## Array operations as transforms

An array operation that moves whole pixels is a warp, and the preamble
of [`Mynd.Sample`](../lib/mynd.mli) gives the transforms of flips, axis
exchanges and crops. A flip of an image `w` columns wide is `x ↦ w - x`.
The program checks each operation on an image of 3 rows and 5 columns:
`nearest` at the positions the warp reads gives the operation's result,
bit for bit.

```ocaml
let x =
  Nx.init Nx.float64 [| 3; 5; 1 |] (fun i ->
      Float.of_int ((i.(0) * 5) + i.(1)))

let is_warp m size y =
  let r = Mynd.Sample.nearest ~edge:`Zero x (reads (mat m) size) in
  Nx.shape r = Nx.shape y && Nx.to_array r = Nx.to_array y

let exchange x = Nx.transpose ~axes:[ 1; 0; 2 ] x

let () =
  (* Flips of the columns and of the rows. *)
  assert (is_warp [| -1.; 0.; 5.; 0.; 1.; 0.; 0.; 0.; 1. |] (3, 5)
            (Nx.flip ~axes:[ 1 ] x));
  assert (is_warp [| 1.; 0.; 0.; 0.; -1.; 3.; 0.; 0.; 1. |] (3, 5)
            (Nx.flip ~axes:[ 0 ] x));
  (* Quarter turns, clockwise (x, y) ↦ (h - y, x) and anticlockwise
     (x, y) ↦ (y, w - x), onto an image of 5 rows and 3 columns. *)
  assert (is_warp [| 0.; -1.; 3.; 1.; 0.; 0.; 0.; 0.; 1. |] (5, 3)
            (Nx.flip ~axes:[ 1 ] (exchange x)));
  assert (is_warp [| 0.; 1.; 0.; -1.; 0.; 5.; 0.; 0.; 1. |] (5, 3)
            (Nx.flip ~axes:[ 0 ] (exchange x)));
  (* A crop of rows 1 and 2, columns 1 to 3. *)
  assert (is_warp [| 1.; 0.; -1.; 0.; 1.; -1.; 0.; 0.; 1. |] (2, 3)
            (Nx.slice [ Nx.R (1, 3); Nx.R (1, 4) ] x));
  (* A zoom-out: the image on a canvas of 6 by 8, 2 rows down and 1 column
     right. *)
  assert (is_warp [| 1.; 0.; 1.; 0.; 1.; 2.; 0.; 0.; 1. |] (6, 8)
            (Nx.pad [| (2, 1); (1, 2); (0, 0) |] 0. x));
  (* Enlarging 3 times across and 2 times down. *)
  assert (is_warp [| 3.; 0.; 0.; 0.; 2.; 0.; 0.; 0.; 1. |] (6, 15)
            (Mynd.Sample.resize ~edge:`Replicate `Box (6, 15) x))
```

Resizing an image of `h` rows and `w` columns to `h'` and `w'` scales
its positions by `w' / w` and `h' / h` under every kernel (the section
Resizing of [`Mynd.Sample`](../lib/mynd.mli)), so its boxes move by
`Mynd.Transform.scaling`. An enlarging resize under `` `Box `` reads
each new pixel from one old pixel, as `nearest` does at the same
positions, to their rounding. A shrinking resize removes detail finer
than the new pixels, and a warp that shrinks does not.

## Boxes

`Mynd.Box.transform` moves a box to the box of its mapped corners. Two
moves in a row hold the corners' box of the first move, which can be
larger than the box the two moves give at once, so compose transforms
first and move the box once. A square of side 2 turned twice by 45° has
side 4, and turned once by 90° keeps side 2:

```ocaml
let side b = let b = floats b in b.(2) -. b.(0)
let square = f64 [| 1; 4 |] [| -1.; -1.; 1.; 1. |]

let () =
  let q = turn (Float.pi /. 4.) in
  let twice = Mynd.Box.transform q (Mynd.Box.transform q square) in
  let once = Mynd.Box.transform (Nx.matmul q q) square in
  assert (near (side twice) 4. && near (side once) 2.)
```

A box holds its object, but turning the box gives the box of the turned
box, which can be much larger than the box of the turned object. A disc
of radius 4 turned by 45° about its centre keeps a box of side 8, while
its box's box grows to about 11.3. Moving the object's mask with the
image and taking `Mynd.Box.of_mask` gives the object's box:

```ocaml
let () =
  let disc =
    Nx.init Nx.bool [| 16; 16; 1 |] (fun i ->
        let dx = Float.of_int i.(1) +. 0.5 -. 8. in
        let dy = Float.of_int i.(0) +. 0.5 -. 8. in
        (dx *. dx) +. (dy *. dy) < 16.)
  in
  let m = about (8., 8.) (turn (Float.pi /. 4.)) in
  let turned = Mynd.Sample.nearest ~edge:`Zero disc (reads m (16, 16)) in
  let moved = Mynd.Box.transform m (Mynd.Box.of_mask Nx.float64 disc) in
  assert (side (Mynd.Box.of_mask Nx.float64 turned) <= 8.);
  assert (side moved > 11.)
```

A projective transform sends the line `d = 0` to infinity, so a box
that meets that line has an unbounded image and moves to NaN (the doc of
`Mynd.Box.transform` in [`mynd.mli`](../lib/mynd.mli)). Below, `d` is `1 - x / 4`. The
first box lies left of `x = 4` and moves to a finite box, and the second
crosses the line:

```ocaml
let horizon = mat [| 1.; 0.; 0.; 0.; 1.; 0.; -0.25; 0.; 1. |]

let () =
  let b =
    floats
      (Mynd.Box.transform horizon
         (f64 [| 2; 4 |] [| 1.; 0.; 3.; 2.; 3.; 0.; 6.; 2. |]))
  in
  assert (near b.(2) 12. && near b.(3) 8.);
  assert (Float.is_nan b.(4))
```

The warped image may still show part of that object. Its mask, or a
polygon drawn into a mask, moved with the image gives the box of the
part that shows. The second box covers columns 3 to 5, which the warp
shows from column 12 of a canvas 16 columns wide:

```ocaml
let () =
  let object_mask =
    Nx.init Nx.bool [| 8; 8; 1 |] (fun i ->
        i.(0) < 2 && i.(1) >= 3 && i.(1) < 6)
  in
  let at = reads horizon (16, 16) in
  let moved = Mynd.Sample.nearest ~edge:`Zero object_mask at in
  let b = floats (Mynd.Box.of_mask Nx.float64 moved) in
  assert (b.(0) = 12. && b.(2) = 16.)
```

## What survives

A box is clipped to the image by the composition in the preamble of
[`Mynd.Box`](../lib/mynd.mli), and a box wholly outside becomes empty.
The program chooses which boxes to keep. Three rules recur: a least
width and height of the clipped box, a least fraction of the box's area
that stays inside, and a box whose centre lies inside the crop, the rule
of the SSD detector's random crops (Liu et al., "SSD: Single Shot
MultiBox Detector", ECCV 2016). Each is a keep mask:

```ocaml
let clip (h, w) b =
  let h = Float.of_int h and w = Float.of_int w in
  Nx.maximum_s (Nx.minimum b (f64 [| 4 |] [| w; h; w; h |])) 0.

let column k b = Nx.slice [ Nx.A; Nx.I k ] b
let ( &&& ) = Nx.logical_and

let large_enough ~least b =
  let s = Mynd.Box.to_xywh b in
  Nx.greater_equal_s (column 2 s) least
  &&& Nx.greater_equal_s (column 3 s) least

let mostly_inside ~share size b =
  let inside = Mynd.Box.area (clip size b) in
  Nx.greater_equal inside (Nx.mul_s (Mynd.Box.area b) share)

let centred (h, w) b =
  let c = Mynd.Box.to_cxcywh b in
  let inside k n = Nx.greater_s (column k c) 0. &&& Nx.less_s (column k c) n in
  inside 0 (Float.of_int w) &&& inside 1 (Float.of_int h)
```

Each rule is written as comparisons that hold for a box to keep. A NaN
fails every comparison, so a NaN box is dropped. A rule written as the
negation of a rule that drops, such as "not narrower than one pixel",
keeps it. The boxes below have been moved onto an image of 10 by 10: one
inside, one mostly outside, one wholly outside and one NaN:

```ocaml
let moved =
  f64 [| 4; 4 |]
    [| 2.; 2.; 6.; 5.; 8.; 1.; 14.; 4.; 12.; 0.; 15.; 3.; nan; nan; nan; nan |]

let size = (10, 10)
let clipped = clip size moved
let keep =
  large_enough ~least:1. clipped
  &&& mostly_inside ~share:0.5 size moved
  &&& centred size moved

let () =
  assert (Nx.to_array keep = [| true; false; false; false |]);
  let narrow = Nx.less_s (column 2 (Mynd.Box.to_xywh moved)) 1. in
  assert (Nx.item [ 3 ] (Nx.logical_not narrow))
```

On the host, the kept rows are taken by their indices:

```ocaml
let kept = Nx.take ~axis:0 (List.hd (Nx.nonzero keep)) clipped
let () = assert (Nx.shape kept = [| 1; 4 |])
```

A batch whose images keep different numbers of boxes is filled out to
one number of rows, and carries its keep mask beside them. A loss over
all the rows replaces the targets of the dropped rows with `Nx.where`
before the loss, then masks the loss. A loss masked alone has a NaN
gradient: where a target is NaN, so is the loss's derivative, and the
cotangent `0` that `Nx.where` sends there times NaN is NaN (Selections,
in the section Tangents and cotangents of
[`rune.mli`](../../../packages/rune/lib/rune.mli)). Replacing the targets
first gives a finite gradient:

```ocaml
let pred = f64 [| 2; 1; 4 |] [| 0.; 0.; 2.; 2.; 1.; 1.; 3.; 3. |]
let target = f64 [| 2; 1; 4 |] [| 0.5; 0.; 2.; 2.5; nan; nan; nan; nan |]
let kept_rows = Nx.create Nx.bool [| 2; 1; 1 |] [| true; false |]
let unit_box = f64 [| 4 |] [| 0.; 0.; 1.; 1. |]

let masked t p =
  let l = Nx.rsub_s 1. (Mynd.Box.giou p t) in
  Nx.sum (Nx.where kept_rows l (Nx.zeros_like l))

let () =
  let g = floats (Rune.grad' (masked target) pred) in
  assert (Array.exists Float.is_nan g);
  let target = Nx.where kept_rows target unit_box in
  let g = floats (Rune.grad' (masked target) pred) in
  assert (Array.for_all Float.is_finite g)
```

## Masks and the image's fill

A warp reads outside the input wherever the output's positions map
there. Under `` `Zero ``, `Mynd.Sample.at` reads zero there and a mask
read by `nearest` reads `false`. An image filled with a constant `c`
reads `x - c` and adds `c` back, so that the area outside reads `c`:

```ocaml
let filled c x at =
  Nx.add_s (Mynd.Sample.at ~edge:`Zero `Triangle (Nx.sub_s x c) at) c

let () =
  let away = Mynd.Transform.translation (f64 [| 2 |] [| 40.; 0. |]) in
  let r = filled 0.5 image (reads away (h, w)) in
  assert (Array.for_all (fun v -> v = 0.5) (floats r))
```

A warp that shrinks reads positions more than a pixel apart, so a thin
part of a mask can vanish. Halving a mask reads the odd rows and columns
of the input, and a pixel in an even row disappears. A program drops
such an annotation or draws the transform again:

```ocaml
let () =
  let dot = Nx.init Nx.bool [| 8; 8; 1 |] (fun i -> i.(0) = 2 && i.(1) = 2) in
  let half = Mynd.Transform.scaling (f64 [| 2 |] [| 0.5; 0.5 |]) in
  let r = Mynd.Sample.nearest ~edge:`Zero dot (reads half (4, 4)) in
  assert (not (Nx.item [] (Nx.any r)))
```

## Keypoints

Keypoints move by `Mynd.Transform.apply`. Datasets that store a keypoint
as the integer indices of its pixel mean the pixel's centre. Reading adds
½ to each coordinate, and writing subtracts it.

A program that hides the keypoints outside the image tests the closed
canvas, `0 <= x <= w` and `0 <= y <= h`. A flip or a quarter turn maps
that canvas onto its output's, so every visible point stays visible.
Tested on `[0, w)` instead, a point at `x = 0` would vanish under a
flip. A point at `x = w` lies in no pixel, so a heatmap of `w` columns
writes it to column `min ⌊x⌋ (w - 1)`. A point on the line where a crop
cuts the canvas is visible in the crops on both sides.

```ocaml
let visible (h, w) p =
  let p = floats p in
  let w = Float.of_int w and h = Float.of_int h in
  let ok = ref true in
  for k = 0 to (Array.length p / 2) - 1 do
    let x = p.(2 * k) and y = p.((2 * k) + 1) in
    ok := !ok && x >= 0. && x <= w && y >= 0. && y <= h
  done;
  !ok

let () =
  let edges = f64 [| 3; 2 |] [| 0.; 4.; 16.; 0.; 3.5; 12. |] in
  let flip = mat [| -1.; 0.; 16.; 0.; 1.; 0.; 0.; 0.; 1. |] in
  assert (visible (h, w) edges);
  assert (visible (h, w) (Mynd.Transform.apply flip edges))
```

A reflection exchanges left and right, so the keypoints named left and
right swap names by the dataset's permutation. An affine transform
reflects where the determinant of its top-left 2 by 2 block is below 0:

```ocaml
let reflects m =
  Nx.item [] (Nx.det (Nx.slice [ Nx.R (0, 2); Nx.R (0, 2) ] m)) < 0.

let () =
  assert (reflects (mat [| -1.; 0.; 16.; 0.; 1.; 0.; 0.; 0.; 1. |]));
  assert (not (reflects m))
```

## Draws

A transform is an array, and its constructors take arrays of
parameters, so one draw of parameters gives one transform per image.
Transforms `[n; 3; 3]` move the boxes of `n` images, `[n; q; 4]`, each by
its own, since the boxes pair with the transforms at their leading axes
(the section Batches of [`Mynd`](../lib/mynd.mli)). For `v` videos of
`t` frames, transforms `[v; 1; 3; 3]` move boxes `[v; t; q; 4]`, one
transform for every frame of a video, and one `[3; 3]` moves them all:

```ocaml
let key = Nx.Rng.key 7
let n = 4
let angles = Nx.mul_s (Nx.Rng.uniform ~key Nx.float64 [| n |]) (Float.pi /. 6.)
let per_image = Mynd.Transform.rotation angles

let boxes =
  f64 [| n; 2; 4 |] (Array.init 32 (fun k -> Float.of_int (k mod 4 + (k / 8))))

let () =
  assert (Nx.shape per_image = [| n; 3; 3 |]);
  assert (Nx.shape (Mynd.Box.transform per_image boxes) = [| n; 2; 4 |]);
  let two = Nx.slice [ Nx.R (0, 2) ] per_image in
  let per_video = Nx.reshape [| 2; 1; 3; 3 |] two in
  let frames = Nx.reshape [| 2; 2; 2; 4 |] boxes in
  assert (Nx.shape (Mynd.Box.transform per_video frames) = [| 2; 2; 2; 4 |])
```

## A detection recipe

A detector's training recipe chains array operations: a colour
adjustment ([Adjusting colour](colour.md)), a zoom-out onto a larger
canvas filled with zeros, a crop, a flip, a resize to the model's input
and a conversion of the boxes to centres and sizes divided by the image's.
Each operation that moves pixels is a transform, so the boxes move once
by their product, and the rules of the previous sections drop the boxes
that leave the image.

The image below is 40 by 40. It is placed on a canvas of 80 by 80 at an
offset, cropped to 60 by 60, flipped and resized to 120 by 120:

```ocaml
let shift x y = Mynd.Transform.translation (f64 [| 2 |] [| x; y |])
let scaling sx sy = Mynd.Transform.scaling (f64 [| 2 |] [| sx; sy |])

let detection_step ~top ~left ~crop:(ci, cj) ~flip x boxes =
  let x = Nx.pad [| (top, 40 - top); (left, 40 - left); (0, 0) |] 0. x in
  let x = Nx.slice [ Nx.R (ci, ci + 60); Nx.R (cj, cj + 60) ] x in
  let x = if flip then Nx.flip ~axes:[ 1 ] x else x in
  let x = Mynd.Sample.resize ~edge:`Replicate (`Lanczos 3) (120, 120) x in
  let fl = Float.of_int in
  let m =
    List.fold_left
      (fun m s -> Nx.matmul s m)
      (shift (fl left) (fl top))
      [
        shift (fl (-cj)) (fl (-ci));
        (if flip then mat [| -1.; 0.; 60.; 0.; 1.; 0.; 0.; 0.; 1. |]
         else mat [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |]);
        scaling 2. 2.;
      ]
  in
  let moved = Mynd.Box.transform m boxes in
  let clipped = clip (120, 120) moved in
  let keep =
    large_enough ~least:2. clipped &&& mostly_inside ~share:0.5 (120, 120) moved
  in
  let normalised = Nx.div_s (Mynd.Box.to_cxcywh clipped) 120. in
  (x, normalised, keep)

let () =
  let x =
    Nx.init Nx.float64 [| 40; 40; 3 |] (fun i -> Float.of_int i.(0) /. 40.)
  in
  let b = f64 [| 2; 4 |] [| 2.; 2.; 12.; 10.; 36.; 30.; 40.; 39. |] in
  let x', b', keep =
    detection_step ~top:10 ~left:30 ~crop:(5, 5) ~flip:true x b
  in
  assert (Nx.shape x' = [| 120; 120; 3 |] && Nx.shape b' = [| 2; 4 |]);
  assert (Nx.to_array keep = [| true; false |])
```

A recipe that trains at several input sizes draws one size per batch and
resizes the batch's images to it, and its boxes move by the scaling of
that resize. A random crop that must hold some of the objects is drawn
again on the host until the boxes it keeps satisfy the recipe's rule.

## A video recipe

A video segmentation recipe moves every frame of a video by one draw of
a flip, a turn and a shear, then resizes the frames, jitters their
colour and converts them to grey. The transforms `[v; 1; 3; 3]` pair
with frames `[v; t; h; w; c]` through positions `[v; 1; h; w; 2]`, the
masks are read by `nearest` with `` `Zero ``, and the images are filled
with a constant colour, such as the mean colour of the training images.
A shear along `x` and then along `y` is the product of two
`Mynd.Transform.shear` transforms. The program below draws the turn and
a shear along `x`.

```ocaml
let video_step ~key ~fill frames masks =
  let v = Nx.dim 0 frames and h = Nx.dim 2 frames and w = Nx.dim 3 frames in
  let draw k lo hi =
    let u = Nx.Rng.uniform ~key:(Nx.Rng.fold_in key k) Nx.float64 [| v |] in
    Nx.add_s (Nx.mul_s u (hi -. lo)) lo
  in
  let degrees = Float.pi /. 180. in
  let angle = draw 0 (-25. *. degrees) (25. *. degrees) in
  let shear = Float.tan (20. *. degrees) in
  let sx = draw 1 (-.shear) shear in
  let zero = Nx.zeros_like sx in
  let m =
    about (Float.of_int w /. 2., Float.of_int h /. 2.)
      (Nx.matmul (Mynd.Transform.rotation angle)
         (Mynd.Transform.shear (Nx.stack ~axis:(-1) [ sx; zero ])))
  in
  let m = Nx.reshape [| v; 1; 3; 3 |] m in
  let c = Mynd.Sample.centres Nx.float64 (h, w) in
  let c = Nx.reshape [| 1; 1; h; w; 2 |] c in
  let at = Mynd.Transform.apply (Nx.inv m) c in
  let read x = Mynd.Sample.at ~edge:`Zero `Triangle x at in
  let frames = Nx.add (read (Nx.sub frames fill)) fill in
  (frames, Mynd.Sample.nearest ~edge:`Zero masks at)

let () =
  let frames =
    Nx.init Nx.float64 [| 2; 3; 8; 8; 3 |] (fun i -> Float.of_int i.(3) /. 8.)
  in
  let masks =
    Nx.init Nx.bool [| 2; 3; 8; 8; 1 |] (fun i -> i.(2) >= 2 && i.(2) < 6)
  in
  let fill = f64 [| 3 |] [| 0.48; 0.46; 0.41 |] in
  let f, m = video_step ~key:(Nx.Rng.key 3) ~fill frames masks in
  assert (Nx.shape f = [| 2; 3; 8; 8; 3 |] && Nx.shape m = [| 2; 3; 8; 8; 1 |])
```

## A transform from four point pairs

A projective transform has eight parameters once `m₂₂` is `1`, so four
points and their images fix it. Each pair `(x, y) ↦ (u, v)` gives two
equations linear in the parameters,
`m₀₀ x + m₀₁ y + m₀₂ - u m₂₀ x - u m₂₁ y = u` and the same for `v` with
the second row, and `Nx.solve` solves the eight. Four pairs need no
rescaling of the coordinates first, and float64 holds the fitted
transform to well below a pixel at the sizes of photographs:

```ocaml
let fit src dst =
  let row k =
    let x = src.(2 * k) and y = src.((2 * k) + 1) in
    let u = dst.(2 * k) and v = dst.((2 * k) + 1) in
    [|
      [| x; y; 1.; 0.; 0.; 0.; -.u *. x; -.u *. y |];
      [| 0.; 0.; 0.; x; y; 1.; -.v *. x; -.v *. y |];
    |]
  in
  let rows = List.concat_map (fun k -> Array.to_list (row k)) [ 0; 1; 2; 3 ] in
  let a = Array.concat rows in
  let p = Nx.solve (f64 [| 8; 8 |] a) (f64 [| 8 |] dst) in
  mat (Array.append (floats p) [| 1. |])

let () =
  let src = [| 0.; 0.; 4000.; 0.; 4000.; 3000.; 0.; 3000. |] in
  let dst = [| 120.; 80.; 3900.; 10.; 4050.; 2950.; -30.; 3100. |] in
  let m = fit src dst in
  let got = floats (Mynd.Transform.apply m (f64 [| 4; 2 |] src)) in
  assert (Array.for_all2 (fun a b -> Float.abs (a -. b) < 1e-6) got dst)
```

## GIoU as a loss

A detector matched to its targets trains on `1 - giou` of each matched
pair. Matched pairs `[k; 4]`, reshaped to `[k; 1; 4]`, give `[k; 1; 1]`,
since `Mynd.Box.giou` pairs every box of one array with every box of the
other. A head whose outputs are raw corners can predict `x₁` below `x₀`,
and the doc of `Mynd.Box.giou` in [`mynd.mli`](../lib/mynd.mli) says
such outputs must be ordered first. Below, an inverted box inside its
target's span has loss 1 and derivative 0, so descent leaves it there,
while its ordered corners have loss 0.75 and train. A head that predicts
a centre and a positive width and height, such as through a sigmoid,
gives boxes that are never inverted.

```ocaml
let ordered b =
  let c k = column k b in
  Nx.stack ~axis:(-1)
    [ Nx.minimum (c 0) (c 2); Nx.minimum (c 1) (c 3);
      Nx.maximum (c 0) (c 2); Nx.maximum (c 1) (c 3) ]

let giou_loss pred target =
  let k = Nx.dim 0 pred in
  let pairs x = Nx.reshape [| k; 1; 4 |] x in
  let g = Mynd.Box.giou (pairs pred) (pairs target) in
  Nx.mean (Nx.rsub_s 1. g)

let () =
  let raw = f64 [| 1; 4 |] [| 2.5; 0.5; 1.5; 1.5 |] in
  let target = f64 [| 1; 4 |] [| 1.; 0.; 3.; 2. |] in
  assert (near (Nx.item [] (giou_loss raw target)) 1.);
  let g = floats (Rune.grad' (fun p -> giou_loss p target) raw) in
  assert (Array.for_all (fun d -> d = 0.) g);
  assert (near (Nx.item [] (giou_loss (ordered raw) target)) 0.75)
```

## What mynd does and what the program does

| mynd | the program |
|---|---|
| transforms and their product, `Mynd.Transform` | the draw of their parameters, per image, video or batch |
| the positions a warp reads, `Mynd.Sample.at` and `nearest` | the fill outside, and the kernel |
| boxes moved by a transform, `Mynd.Box.transform` | which boxes survive, and what replaces a NaN box |
| boxes of masks, `Mynd.Box.of_mask` | redrawing a transform whose mask vanished |
| keypoints moved, `Mynd.Transform.apply` | their visibility and the left and right permutation |
| `Mynd.Box.giou` | ordering raw outputs, matching, and the loss's weights |
