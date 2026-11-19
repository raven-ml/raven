# Filtering images

This page builds the filters that mynd leaves to composition over
`Mynd.correlate`, `Mynd.windows` and `Mynd.gaussian_blur`: box blurs,
unsharp masks, a constant other than zero past the edge, the Laplacian,
rank, minimum and maximum filters, the bilateral filter and the discrete
Gaussian. The rules each call follows, windows, taps, edges and weights
paired with batch axes, are in the Filtering section of
[`mynd.mli`](../lib/mynd.mli).

```ocaml
let img =
  Nx.create Nx.float64 [| 6; 8; 1 |]
    (Array.init 48 (fun k -> Float.of_int ((k * 29) mod 47) /. 46.))

let largest a = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. a
let floats x = Nx.to_array (Nx.cast Nx.float64 x)
let distance a b = largest (floats (Nx.sub a b))
```

## Box blur

A box blur of extent `k` is the mean of a `k` by `k` window. Its weights
are a column of `1 / k` times a row of `1 / k`, so it is two calls of
`k` products a pixel each:

```ocaml
let box edge k x =
  let w = Nx.full Nx.float64 [| k; 1 |] (1. /. Float.of_int k) in
  Mynd.correlate ~edge (Nx.reshape [| 1; k |] w) (Mynd.correlate ~edge w x)
```

For a large `k`, running sums cost two subtractions a pixel at any
extent. The image is extended as the window reads it, summed along each
axis from a zero, and each window's sum is the difference of two running
sums `k` apart:

```ocaml
(* [cut ~axis lo hi x] is [x]'s elements [lo] to [hi - 1] along [axis]. *)
let cut ~axis lo hi x =
  Nx.slice (List.init (Nx.ndim x + axis) (fun _ -> Nx.A) @ [ Nx.R (lo, hi) ]) x

let window_sums ~axis k x =
  let n = Nx.dim axis x - k + 1 in
  let zero = Nx.zeros_like (cut ~axis 0 1 x) in
  let s = Nx.cumsum ~axis (Nx.concatenate ~axis [ zero; x ]) in
  Nx.sub (cut ~axis k (k + n) s) (cut ~axis 0 n s)

let box_sums edge k x =
  let x = Nx.extend ~axis:(-3) edge (k / 2, (k - 1) / 2) x in
  let x = Nx.extend ~axis:(-2) edge (k / 2, (k - 1) / 2) x in
  Nx.div_s (window_sums ~axis:(-2) k (window_sums ~axis:(-3) k x)) (Float.of_int (k * k))

let () = assert (distance (box `Replicate 5 img) (box_sums `Replicate 5 img) < 1e-12)
```

The running sums grow along the image, so their differences carry the
rounding of the largest of them. In float32, take the sums in float64.

## Unsharp mask

An unsharp mask adds back `a` times the detail that a blur removes,
`x + a (x - blur x)`, which is `Nx.lerp (blur x) x (1 + a)`. A flat
region has no detail and stays flat:

```ocaml
let sharpen a x =
  let blur = Mynd.gaussian_blur ~edge:`Symmetric_half ~radius:4 (Nx.scalar Nx.float64 1.2) x in
  Nx.lerp blur x (Nx.full_like x (1. +. a))

let () =
  let flat = Nx.full Nx.float64 [| 5; 5; 1 |] 0.3 in
  assert (distance (sharpen 0.8 flat) flat < 1e-15)
```

## A constant past the edge

`` `Zero `` reads zero past the edge. To read a constant `v` there,
correlate the image less `v` and add back `v` times the weights' sum,
which a correlation of a constant image gives:

```ocaml
let correlate_with_fill v k x =
  let s = Nx.item [] (Nx.sum k) in
  Nx.add_s (Mynd.correlate ~edge:`Zero k (Nx.sub_s x v)) (v *. s)

let () =
  let k = Nx.full Nx.float64 [| 3; 3 |] (1. /. 9.) in
  let flat = Nx.full Nx.float64 [| 4; 4; 1 |] 0.7 in
  assert (distance (correlate_with_fill 0.7 k flat) flat < 1e-15)
```

## Laplacian

The Laplacian `∂²/∂x² + ∂²/∂y²` of a smooth image is approximated by its
second differences, the weights below. On `x² + y²` it is `4`, exactly,
at every pixel whose window lies inside the image:

```ocaml
let laplacian edge x =
  let w = Nx.create Nx.float64 [| 3; 3 |] [| 0.; 1.; 0.; 1.; -4.; 1.; 0.; 1.; 0. |] in
  Mynd.correlate ~edge w x

let () =
  let bowl = Nx.init Nx.float64 [| 6; 7; 1 |] (fun i -> Float.of_int ((i.(0) * i.(0)) + (i.(1) * i.(1)))) in
  let l = laplacian `Replicate bowl in
  let inside = Nx.slice [ Nx.R (1, 5); Nx.R (1, 6) ] l in
  assert (distance inside (Nx.full_like inside 4.) = 0.)
```

Second differences amplify noise; blurring first, by `gaussian_blur`,
gives the Laplacian of the Gaussian.

## Rank, minimum and maximum filters

`Mynd.windows` gives each pixel's window as two trailing axes. Any
reduction over them is a filter: the rank `r` of each window sorted, its
least or its greatest element. Rank `(n - 1) / 2` of a window of `n` taps
is the lower median, which `Mynd.median_blur` computes in bands of rows
to hold less memory.

```ocaml
let rank edge (kh, kw) r x =
  let w = Mynd.windows ~edge (kh, kw) x in
  let s = Nx.shape w in
  let flat = Nx.reshape (Array.append (Array.sub s 0 (Array.length s - 2)) [| kh * kw |]) w in
  let sorted, _ = Nx.sort ~axis:(-1) flat in
  Nx.slice [ Nx.A; Nx.A; Nx.A; Nx.I r ] sorted

let minimum edge size x = Nx.min ~axes:[ -2; -1 ] (Mynd.windows ~edge size x)
let maximum edge size x = Nx.max ~axes:[ -2; -1 ] (Mynd.windows ~edge size x)

let () =
  assert (distance (rank `Periodic (3, 3) 4 img) (Mynd.median_blur ~edge:`Periodic (3, 3) img) = 0.);
  assert (distance (rank `Replicate (3, 3) 0 img) (minimum `Replicate (3, 3) img) = 0.)
```

Sorting stores `kh kw` elements a pixel, and the minimum and maximum read
that many. A NaN in a window makes its minimum and maximum NaN, where a
rank in the sort order is NaN only past the window's numbers.

## Bilateral filter

A bilateral filter averages a pixel's window with weights that fall with
distance in the image and with difference in colour, so it smooths
regions and keeps their edges. The colour difference is one Euclidean
distance across the channels, so the channels of a pixel share their
weights and no colour bleeds across an edge into another channel.

```ocaml
let bilateral edge ~radius ~space ~range x =
  let k = (2 * radius) + 1 in
  let w = Mynd.windows ~edge (k, k) x in
  let centre = Nx.unsqueeze ~axes:[ -2; -1 ] x in
  let d2 = Nx.sum ~axes:[ -3 ] ~keepdims:true (Nx.square (Nx.sub w centre)) in
  let spatial =
    Nx.init Nx.float64 [| k; k |] (fun i ->
        let dy = Float.of_int (i.(0) - radius) and dx = Float.of_int (i.(1) - radius) in
        Float.exp (-.((dx *. dx) +. (dy *. dy)) /. (2. *. space *. space)))
  in
  let weight = Nx.mul spatial (Nx.exp (Nx.div_s d2 (-2. *. range *. range))) in
  Nx.div (Nx.sum ~axes:[ -2; -1 ] (Nx.mul weight w)) (Nx.sum ~axes:[ -2; -1 ] weight)
```

With an infinite range every colour weighs alike, and the filter is the
Gaussian blur of the same deviation:

```ocaml
let () =
  let b = bilateral `Replicate ~radius:3 ~space:1.1 ~range:Float.infinity img in
  let g = Mynd.gaussian_blur ~edge:`Replicate ~radius:3 (Nx.scalar Nx.float64 1.1) img in
  assert (distance b g < 1e-12)
```

## The discrete Gaussian

`Mynd.gaussian_blur` samples the Gaussian at the pixels. Below a
deviation of about one pixel those samples blur less than the deviation
states, and their derivative in it vanishes as it falls to zero. Scale
space wants another kernel: Lindeberg's discrete analogue of the
Gaussian, whose blurs compose, a blur by `σ₁` then by `σ₂` being
the blur by `√(σ₁² + σ₂²)`, and whose variance is `σ²` at every
deviation.

Its transform is `e^(σ² (cos θ - 1))` at each frequency `θ`. Its taps on
`2r + 1` pixels follow by an inverse discrete Fourier transform at
`M = 2r + 1` frequencies, which folds the tails past `r` back onto the
taps, so they sum to `1` exactly. The deviation is data, so the taps
have a derivative in it, at zero too.

```ocaml
let discrete ~radius sigma =
  let m = (2 * radius) + 1 in
  let theta j = 2. *. Float.pi *. Float.of_int j /. Float.of_int m in
  let cosines =
    Nx.init Nx.float64 [| m; m |] (fun i ->
        Float.cos (Float.of_int (i.(0) - radius) *. theta i.(1)) /. Float.of_int m)
  in
  let c = Nx.init Nx.float64 [| m |] (fun i -> Float.cos (theta i.(0)) -. 1.) in
  let spectrum = Nx.exp (Nx.mul (Nx.square (Nx.unsqueeze ~axes:[ -1 ] sigma)) c) in
  Nx.matmul spectrum (Nx.transpose cosines)

let discrete_blur edge ~radius sigma x =
  let g = discrete ~radius sigma in
  Mynd.correlate ~edge (Nx.unsqueeze ~axes:[ -1 ] g)
    (Mynd.correlate ~edge (Nx.unsqueeze ~axes:[ -2 ] g) x)

let () =
  let g = discrete ~radius:12 (Nx.scalar Nx.float64 1.5) in
  let n = Nx.init Nx.float64 [| 25 |] (fun i -> Float.of_int (i.(0) - 12)) in
  assert (Float.abs (Nx.item [] (Nx.sum g) -. 1.) < 1e-12);
  assert (Float.abs (Nx.item [] (Nx.sum (Nx.mul g (Nx.square n))) -. 2.25) < 1e-8);
  let impulse = floats (discrete ~radius:2 (Nx.scalar Nx.float64 0.)) in
  assert (largest (Array.mapi (fun i v -> if i = 2 then v -. 1. else v) impulse) < 1e-15);
  let d = Rune.grad' (fun s -> Nx.sum (discrete_blur `Periodic ~radius:3 s img)) (Nx.scalar Nx.float64 0.) in
  assert (Float.is_finite (Nx.item [] d))
```

The taps past `r` fold back onto those within it, so the radius must
hold the kernel's tails, as for the sampled Gaussian.

## A deviation for each image

`gaussian_blur` pairs a deviation with each image of a batch, so an
augmentation that draws a deviation for each image, here uniformly from
`(0.1, 2)`, blurs the batch in one call:

```ocaml
let () =
  let batch = Nx.broadcast_to [| 4; 6; 8; 1 |] img in
  let sigmas = Nx.create Nx.float64 [| 4 |] (Array.init 4 (fun _ -> 0.1 +. Random.float 1.9)) in
  let blurred = Mynd.gaussian_blur ~edge:`Symmetric_whole ~radius:4 sigmas batch in
  assert (Nx.shape blurred = [| 4; 6; 8; 1 |])
```

A small deviation blurs little. Below `0.3` a neighbour one pixel away
weighs under `e^(-1/2σ²)`, `0.39 %`, of the centre, and the two passes
give all the neighbours together under `1.6 %` of a pixel's result, so the
blur moves a pixel of an 8-bit image by under 4 levels.
