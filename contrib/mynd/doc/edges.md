# Gradients and edges

This page builds edge detectors and derivative filters from
`Mynd.gradient`, `Mynd.gradient_maxima`, `Mynd.gaussian_blur` and
`Mynd.reconstruct`: Canny's edges, wider derivative stencils, the
Laplacian of a Gaussian, thresholds, the border, and the structure
tensor. The rules each call follows are in the Gradients and edges, the
Filtering and the Selection sections of [`mynd.mli`](../lib/mynd.mli).

The programs below work on grey images of `float32` values in `[0, 1]`:

```ocaml
(* A dark image with a light square turned by 30°, [[48; 48; 1]]. *)
let square =
  let c = Float.cos (Float.pi /. 6.) and s = Float.sin (Float.pi /. 6.) in
  Nx.init Nx.float32 [| 48; 48; 1 |] (fun i ->
      let u = Float.of_int i.(1) +. 0.5 -. 24. and v = Float.of_int i.(0) +. 0.5 -. 24. in
      let a = (c *. u) +. (s *. v) and b = (c *. v) -. (s *. u) in
      if Float.abs a < 12. && Float.abs b < 12. then 0.8 else 0.1)

let largest a = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. a
let floats x = Nx.to_array (Nx.cast Nx.float64 x)
let distance a b = largest (floats (Nx.sub a b))

(* [pieces m] is the number of 8-connected components of a mask. *)
let pieces m =
  Mynd.components `Eight m |> floats |> Array.to_list
  |> List.filter (fun l -> l >= 0.)
  |> List.sort_uniq compare |> List.length
```

## Gradients

`Mynd.gradient` gives the pair `(gx, gy)` in values per pixel. A ramp
that grows by `a` a column and `b` a row reads `a` and `b` under every
stencil, so thresholds on the strength `Nx.hypot gx gy` mean the same
under each:

```ocaml
let ramp = Nx.init Nx.float32 [| 6; 7; 1 |] (fun i -> (0.25 *. Float.of_int i.(1)) -. (0.125 *. Float.of_int i.(0)))

let () =
  List.iter
    (fun s ->
      let gx, gy = Mynd.gradient ~edge:`Antisymmetric_whole s ramp in
      assert (distance gx (Nx.full_like gx 0.25) = 0.);
      assert (distance gy (Nx.full_like gy (-0.125)) = 0.))
    [ `Central; `Sobel; `Scharr ]
```

Under `` `Antisymmetric_whole `` the image continues past its border as
the line through its last two pixels, so the slope holds to the border.

`` `Central `` takes the difference alone. After `Mynd.gaussian_blur` of
deviation `σ` it is a Gaussian derivative whose scale is `σ` alone.
`` `Sobel `` and `` `Scharr `` smooth across the difference too, which
adds a little blur along the edge and makes the gradient's direction
more accurate on detail a few pixels wide.

A wider derivative stencil is a binomial blur before `` `Sobel ``. Each
blur by `¼ ½ ¼` along both axes widens the stencil by two pixels, so one
gives a stencil of five pixels and two one of seven. The blur's weights
sum to `1`, so the slope stays in values per pixel:

```ocaml
let binomial edge x =
  let b = Nx.create Nx.float32 [| 3 |] [| 0.25; 0.5; 0.25 |] in
  x
  |> Mynd.correlate ~edge (Nx.reshape [| 1; 3 |] b)
  |> Mynd.correlate ~edge (Nx.reshape [| 3; 1 |] b)

let sobel5 edge x = Mynd.gradient ~edge `Sobel (binomial edge x)

let () =
  let gx, _ = sobel5 `Antisymmetric_whole ramp in
  assert (distance gx (Nx.full_like gx 0.25) < 1e-6)
```

## The Laplacian of a Gaussian

The Laplacian `∂²/∂x² + ∂²/∂y²` of an image blurred by a Gaussian of
deviation `σ` responds to blobs of radius about `σ √2`. Multiplied by
`σ²`, its response to a blob of matching size is the same at every
scale, so responses at several deviations compare:

```ocaml
let laplacian edge x =
  let w = Nx.create Nx.float32 [| 3; 3 |] [| 0.; 1.; 0.; 1.; -4.; 1.; 0.; 1.; 0. |] in
  Mynd.correlate ~edge w x

let log ~radius sigma x =
  let edge = `Symmetric_whole in
  let b = Mynd.gaussian_blur ~edge ~radius (Nx.scalar Nx.float32 sigma) x in
  Nx.mul_s (laplacian edge b) (sigma *. sigma)

let () =
  let flat = Nx.full Nx.float32 [| 8; 8; 1 |] 0.5 in
  assert (distance (log ~radius:6 1.5 flat) (Nx.zeros_like flat) < 1e-6)
```

## Canny's edges

Canny's edges are the pixels where the strength of a Gaussian derivative
is a maximum along the gradient and above a low threshold, kept where
they join one above a high threshold. Each step is a call:

```ocaml
(* Canny's edges of grey images [x], [[…; h; w; 1]], float32, values in
   [0, 1]. [sigma] is [[b'…]], in [x]'s dtype; [low] and [high] are
   strengths in values per pixel. A step of height [a] has strength about
   [a / (σ √(2π))], so thresholds scale with [1 / σ]. One symmetric edge
   rule for every step computes what the detector computes on the image
   mirrored past its border. *)
let canny ~radius ~sigma ~low ~high x =
  let edge = `Symmetric_whole in
  let b = Mynd.gaussian_blur ~edge ~radius sigma x in
  let ((gx, gy) as g) = Mynd.gradient ~edge `Central b in
  let m = Nx.hypot gx gy in
  let candidates = Nx.logical_and (Mynd.gradient_maxima g) (Nx.greater_s m low) in
  Mynd.reconstruct `Eight ~marker:(Nx.greater_s m high) candidates

let () =
  let e = canny ~radius:6 ~sigma:(Nx.scalar Nx.float32 1.4) ~low:0.05 ~high:0.15 square in
  assert (pieces e = 1)
```

The square's step has height `0.7`, so at `σ = 1.4` its strength is
about `0.2`, above both thresholds. The outline is one piece because
`gradient_maxima` interpolates the strength along the gradient at every
direction, so a contour turning a corner keeps a pixel there.

`reconstruct` keeps the candidates joined to a strong pixel through
8-connected neighbours. It ignores a marker pixel outside the
candidates, so the strong pixels need no masking first.

Images in a narrow float format lose the strength's low bits: convert
them to `float32` first, as `canny`'s comment asks. A colour image gives
edges per channel. `reconstruct` takes masks of one channel, so either
move the channels into the batch, or join them before hysteresis with
`Nx.any ~axes:[ -1 ] ~keepdims:true`, of the candidates and of the strong
pixels each, which gives one edge map.

## Thresholds

Strengths are in values per pixel, so a threshold reads as the height of
a step over the pixels that blur it: a step of height `a` has strength
`a (g(0) + g(1)) / 2` beside it, `g` the blur's weights, about
`a / (σ √(2π))`. Halving `σ` doubles the strength of every step.

A threshold relative to each image's own strengths takes a quantile of
`m` per image. The batch axes pair with the quantiles by a reshape:

```ocaml
let relative ~radius ~sigma q x =
  let edge = `Symmetric_whole in
  let b = Mynd.gaussian_blur ~edge ~radius sigma x in
  let gx, gy = Mynd.gradient ~edge `Central b in
  let m = Nx.hypot gx gy in
  let n = Nx.dim 0 m in
  let flat = Nx.reshape [| n; Nx.numel m / n |] m in
  let sorted, _ = Nx.sort ~axis:(-1) flat in
  let k = Float.to_int (q *. Float.of_int (Nx.dim 1 flat - 1)) in
  Nx.reshape [| n; 1; 1; 1 |] (Nx.slice [ Nx.A; Nx.I k ] sorted)

let () =
  let batch = Nx.stack [ square; Nx.mul_s square 0.5 ] in
  let t = relative ~radius:6 ~sigma:(Nx.scalar Nx.float32 1.4) 0.9 batch in
  assert (Nx.shape t = [| 2; 1; 1; 1 |]);
  let t = floats t in
  assert (Float.abs (t.(1) -. (t.(0) /. 2.)) < 1e-6)
```

Where an image is a linear ramp, the strength is constant up to
rounding, and rounding decides which pixels are maxima. Those pixels
have the ramp's strength, so a low threshold above the ramp's slope
removes them:

```ocaml
let () =
  let ramp = Nx.init Nx.float32 [| 32; 48; 1 |] (fun i -> 0.004 *. Float.of_int i.(1)) in
  let e = canny ~radius:6 ~sigma:(Nx.scalar Nx.float32 1.4) ~low:0.008 ~high:0.01 ramp in
  assert (not (Nx.item [] (Nx.any e)))
```

## The border

Under `` `Symmetric_whole `` every step reads the image mirrored about
its border pixels. The derivative across the border is then `0` on the
border row itself, so an edge that runs along the border between rows
`0` and `1` is marked on row `1`.

To run the detector on the image continued by another rule, extend the
image by that rule first, by the blur's radius and the two pixels that
the gradient and the thinning read, and crop the candidates and the
strong pixels before hysteresis, so that edges join only through the
image's own pixels:

```ocaml
let canny_extended rule ~radius ~sigma ~low ~high x =
  let r = radius + 2 in
  let x' = Nx.extend ~axis:(-3) rule (r, r) (Nx.extend ~axis:(-2) rule (r, r) x) in
  let edge = `Zero in
  let b = Mynd.gaussian_blur ~edge ~radius sigma x' in
  let ((gx, gy) as g) = Mynd.gradient ~edge `Central b in
  let m = Nx.hypot gx gy in
  let h = Nx.dim (-3) x and w = Nx.dim (-2) x in
  let crop y = Nx.slice [ Nx.R (r, r + h); Nx.R (r, r + w) ] y in
  let candidates = crop (Nx.logical_and (Mynd.gradient_maxima g) (Nx.greater_s m low)) in
  Mynd.reconstruct `Eight ~marker:(crop (Nx.greater_s m high)) candidates

let () =
  let sigma = Nx.scalar Nx.float32 1.4 in
  let a = canny ~radius:6 ~sigma ~low:0.05 ~high:0.15 square in
  let b = canny_extended `Symmetric_whole ~radius:6 ~sigma ~low:0.05 ~high:0.15 square in
  assert (Nx.to_array a = Nx.to_array b)
```

Under `` `Symmetric_whole `` the two agree, since every step's reads past
the border are the mirrored image's own pixels.

## The structure tensor

The structure tensor is the blur of the gradient's products `gx²`,
`gx gy` and `gy²`. Its eigenvalues measure how much the image changes
along its two principal directions: both large at a corner, one at an
edge. Stacking the three products puts them on a batch axis, so one
blur covers them, and Harris' response `det - k trace²` follows:

```ocaml
let harris ~k x =
  let gx, gy = Mynd.gradient ~edge:`Symmetric_whole `Scharr x in
  let products = Nx.stack [ Nx.mul gx gx; Nx.mul gx gy; Nx.mul gy gy ] in
  let t = Mynd.gaussian_blur ~edge:`Symmetric_whole ~radius:4 (Nx.scalar Nx.float32 1.) products in
  let at i = Nx.slice [ Nx.I i ] t in
  let det = Nx.sub (Nx.mul (at 0) (at 2)) (Nx.square (at 1)) in
  let trace = Nx.add (at 0) (at 2) in
  Nx.sub det (Nx.mul_s (Nx.square trace) k)

let () =
  let r = harris ~k:0.05 square in
  assert (Nx.shape r = [| 48; 48; 1 |])
```

A gradient transforms as a covector: under a warp `p ↦ A p` of the
image, `(gx, gy)` maps by the inverse transpose of `A`, so it is not
a position for `Mynd.Transform`.
