# Adjusting colour

A colour adjustment moves each colour toward a reference colour or away
from it, or turns its hue. [`Mynd.Colour`](../lib/mynd.mli) leaves these
to composition, and the programs below build brightness, contrast,
saturation, hue, a hue rotation that keeps luminance, and CIE Lab
relative to D50. Each rule a program relies on is linked where it is
used.

The programs below work on a small image of encoded sRGB values in
`[0, 1]`, as a decoder returns them after
[`Mynd.convert`](../lib/mynd.mli), whose fractions the preamble of `Mynd`
defines.

```ocaml
let img =
  Nx.create Nx.float32 [| 4; 5; 3 |]
    (Array.init 60 (fun k -> Float.of_int ((k * 37) mod 61) /. 60.))

(* [largest a] is the largest magnitude of a float array. *)
let largest a = Array.fold_left (fun m v -> Float.max m (Float.abs v)) 0. a
let floats x = Nx.to_array (Nx.cast Nx.float64 x)
let distance a b = largest (floats (Nx.sub a b))
```

## Brightness, contrast and saturation

Each adjustment blends the image with a reference image by a factor `f`.
`Nx.lerp r x f` is `r + f (x - r)` (`Nx.lerp` in
[`nx.mli`](../../../packages/nx/lib/nx.mli)), so `f = 1` keeps the
image, `f = 0` gives the reference, and a factor above `1` moves away
from it. The references are black for brightness, the image's mean luma
for contrast, and each pixel's luma for saturation. Luma is a weighted
sum of encoded values, which these are (the preamble of
[`Mynd.Colour`](../lib/mynd.mli)).

```ocaml
let clamp x = Nx.clamp ~min:0. ~max:1. x

let factor x f = Nx.full_like x f

let brightness f x = clamp (Nx.lerp (Nx.zeros_like x) x (factor x f))

let contrast f x =
  let mean = Nx.mean ~axes:[ -3; -2; -1 ] ~keepdims:true (Mynd.Colour.luma x) in
  clamp (Nx.lerp (Nx.broadcast_to (Nx.shape x) mean) x (factor x f))

let saturation f x =
  let grey = Nx.broadcast_to (Nx.shape x) (Mynd.Colour.luma x) in
  clamp (Nx.lerp grey x (factor x f))
```

A saturation of `0` gives every pixel its luma in all three channels. A
grey has no colour to remove, and stays as it is under any saturation,
since `Nx.lerp a a w` is `a` and the weighted sum of a grey is that grey
(the preamble of [`Mynd.Colour`](../lib/mynd.mli)):

```ocaml
let () =
  let grey = saturation 0. img in
  let g = Nx.split ~axis:(-1) 3 grey in
  assert (distance (List.nth g 0) (List.nth g 1) = 0.);
  assert (distance (List.nth g 1) (List.nth g 2) = 0.);
  assert (distance (saturation 1.7 grey) grey = 0.)
```

## Alpha in linear light

An image with alpha is interpolated premultiplied, and averaged in linear
light. The transfer function applies to the colours alone, since alpha
is linear ([`Mynd.srgb_to_linear`](../lib/mynd.mli)), and it applies to
straight colours, before `premultiply` and after `unpremultiply`
([`Mynd.premultiply`](../lib/mynd.mli)). So a program splits off alpha,
makes the colours linear, joins them and premultiplies, and reverses the
steps after.

```ocaml
let split x =
  match Nx.split_at ~axis:(-1) [ Nx.dim (-1) x - 1 ] x with
  | [ colour; alpha ] -> (colour, alpha)
  | _ -> assert false

let to_working x =
  let colour, alpha = split x in
  Mynd.premultiply (Nx.concatenate ~axis:(-1) [ Mynd.srgb_to_linear colour; alpha ])

let of_working y =
  let colour, alpha = split (Mynd.unpremultiply y) in
  Nx.concatenate ~axis:(-1) [ Mynd.linear_to_srgb colour; alpha ]

let () =
  let alpha = Nx.full Nx.float32 [| 4; 5; 1 |] 0.6 in
  let rgba = Nx.concatenate ~axis:(-1) [ img; alpha ] in
  assert (distance (of_working (to_working rgba)) rgba < 1e-5)
```

## Hue

The hue of the HSV form is in turns (the preamble of
[`Mynd.Colour`](../lib/mynd.mli)), and
[`Mynd.Colour.of_hsv`](../lib/mynd.mli) reads any finite hue modulo `1`,
so turning every colour by `d` turns is an addition. A whole turn gives
the image back.

```ocaml
let turn d x =
  match Nx.split ~axis:(-1) 3 (Mynd.Colour.to_hsv x) with
  | [ h; s; v ] -> Mynd.Colour.of_hsv (Nx.concatenate ~axis:(-1) [ Nx.add h d; s; v ])
  | _ -> assert false

let () =
  let by d = turn (Nx.scalar Nx.float32 d) img in
  assert (distance (by 0.) img < 1e-6);
  assert (distance (by 1.) img < 1e-6);
  assert (distance (by 0.25) (by (-0.75)) < 1e-6)
```

The turn is differentiable in `d` and in the image. Where a colour
crosses a sextant boundary of the hexagon of hues (the preamble of
[`Mynd.Colour`](../lib/mynd.mli)), the derivative in
`d` is the mean of its two one-sided derivatives
([`Mynd.Colour.of_hsv`](../lib/mynd.mli)), so a loss that turns colours
by a learned `d` receives a gradient at every value of `d`:

```ocaml
let () =
  let loss d = Nx.sum (turn d img) in
  let g = Rune.grad' loss (Nx.scalar Nx.float32 0.1) in
  assert (Float.is_finite (Nx.item [] g))
```

The turn keeps each colour's value and saturation. It changes its luma
and its luminance, since the primaries differ in brightness. A red
turned to green brightens.

### Turning hue at constant luminance

The W3C's Filter Effects Module Level 1 defines a hue rotation by a
matrix (`feColorMatrix`, type `hueRotate`) that keeps the weighted sum
`0.213 r + 0.715 g + 0.072 b`, the weights of
[`Mynd.Colour.luminance`](../lib/mynd.mli) to three digits. It applies to
linear values, where that sum is luminance. Each row sums to
`1`, so greys stay grey.

```ocaml
let hue_rotate degrees x =
  let a = Float.pi *. degrees /. 180. in
  let c = Float.cos a and s = Float.sin a in
  let base = [| 0.213; 0.715; 0.072 |] in
  let cos_part =
    [| [| 0.787; -0.715; -0.072 |]; [| -0.213; 0.285; -0.072 |]; [| -0.213; -0.715; 0.928 |] |]
  and sin_part =
    [| [| -0.213; -0.715; 0.928 |]; [| 0.143; 0.140; -0.283 |]; [| -0.787; 0.715; 0.072 |] |]
  in
  let m =
    Array.init 9 (fun k ->
        let i = k / 3 and j = k mod 3 in
        base.(j) +. (c *. cos_part.(i).(j)) +. (s *. sin_part.(i).(j)))
  in
  let m = Nx.create (Nx.dtype x) [| 3; 3 |] m in
  Nx.matmul x (Nx.transpose m)

let () =
  let lin = Mynd.srgb_to_linear img in
  let turned = hue_rotate 120. lin in
  assert (distance (Mynd.Colour.luminance turned) (Mynd.Colour.luminance lin) < 1e-3);
  let grey = Nx.full Nx.float32 [| 2; 3 |] 0.4 in
  assert (distance (hue_rotate 75. grey) grey < 1e-6)
```

The weights are rounded to three digits, so luminance is kept to that
precision. Where exact luminance matters, turn by the HSV form and scale
each colour by the ratio of its luminances before and after.

## Lab relative to D50

[`Mynd.Colour.linear_to_lab`](../lib/mynd.mli) measures colours against
D65, the white of linear sRGB. ICC profiles and image editors that offer
Lab measure against D50, and reach it by adapting the colour's XYZ with
the Bradford transform, the usual matrix of chromatic adaptation.

Lab depends on XYZ only through XYZ divided by the white. So Lab
relative to D50 of a colour adapted to D50 is
`Mynd.Colour.linear_to_lab` of the linear sRGB colour whose XYZ, divided
by D65, equals the adapted XYZ divided by D50. That colour is one 3×3
matrix away, the product of sRGB's matrix `M` to XYZ (the preamble of
[`Mynd.Colour`](../lib/mynd.mli)), the Bradford adaptation, the ratio of
the whites, and `M`'s inverse. The constants are CSS Color Module Level
4's.

```ocaml
let mat a b =
  Array.init 3 (fun i ->
      Array.init 3 (fun j ->
          a.(i).(0) *. b.(0).(j) +. a.(i).(1) *. b.(1).(j) +. a.(i).(2) *. b.(2).(j)))

let srgb_to_xyz =
  [|
    [| 506752. /. 1228815.; 87881. /. 245763.; 12673. /. 70218. |];
    [| 87098. /. 409605.; 175762. /. 245763.; 12673. /. 175545. |];
    [| 7918. /. 409605.; 87881. /. 737289.; 1001167. /. 1053270. |];
  |]

let xyz_to_srgb =
  [|
    [| 12831. /. 3959.; -329. /. 214.; -1974. /. 3959. |];
    [| -851781. /. 878810.; 1648619. /. 878810.; 36519. /. 878810. |];
    [| 705. /. 12673.; -2585. /. 12673.; 705. /. 667. |];
  |]

let d65_to_d50 =
  [|
    [| 1.0479297925449969; 0.022946870601609652; -0.05019226628920524 |];
    [| 0.02962780877005599; 0.9904344267538799; -0.017073799063418826 |];
    [| -0.009243040646204504; 0.015055191490298152; 0.7518742814281371 |];
  |]

let d65 = [| 0.3127 /. 0.3290; 1.; (1. -. 0.3127 -. 0.3290) /. 0.3290 |]
let d50 = [| 0.3457 /. 0.3585; 1.; (1. -. 0.3457 -. 0.3585) /. 0.3585 |]

let to_d50 =
  let ratio =
    Array.init 3 (fun i -> Array.init 3 (fun j -> if i = j then d65.(i) /. d50.(i) else 0.))
  in
  mat xyz_to_srgb (mat ratio (mat d65_to_d50 srgb_to_xyz))

let linear_to_lab_d50 x =
  let m = Nx.create (Nx.dtype x) [| 3; 3 |] (Array.concat (Array.to_list to_d50)) in
  Mynd.Colour.linear_to_lab (Nx.matmul x (Nx.transpose m))

let () =
  let white = Nx.ones Nx.float64 [| 3 |] in
  let lab = floats (linear_to_lab_d50 white) in
  assert (Float.abs (lab.(0) -. 100.) < 1e-9);
  assert (Float.abs lab.(1) < 1e-9 && Float.abs lab.(2) < 1e-9)
```

The Bradford matrix maps D65's white onto D50's, so sRGB's white lands on
`(100, 0, 0)` to rounding.
