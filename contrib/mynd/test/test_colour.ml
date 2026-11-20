(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module V = Mynd

type flt = F : string * (float, _) Nx.dtype -> flt

let wide =
  [
    F ("float16", Nx.float16);
    F ("bfloat16", Nx.bfloat16);
    F ("float32", Nx.float32);
    F ("float64", Nx.float64);
  ]

let all_floats =
  [
    F ("float4_e2m1fn", Nx.float4_e2m1fn);
    F ("float8_e5m2", Nx.float8_e5m2);
    F ("float8_e4m3fn", Nx.float8_e4m3fn);
  ]
  @ wide

let floats x = Nx.to_array (Nx.cast Nx.float64 x)
let eps (type s) (dt : (float, s) Nx.dtype) = (Nx.Dtype.float_format dt).epsilon

let colours dt cs =
  Nx.cast dt
    (Nx.create Nx.float64
       [| List.length cs; 3 |]
       (Array.of_list (List.concat_map (fun (r, g, b) -> [ r; g; b ]) cs)))

let one (r, g, b) = Nx.create Nx.float64 [| 3 |] [| r; g; b |]

(* Colours *)

(* A colour drawn from one of these kinds, each covered: free in [0, 1]³ or [-1,
   1]³, two channels tied for the greatest or the least, a grey and black,
   either zero. *)
type kind = Free | Signed | Top_tie | Low_tie | Grey | Black

let kind_name = function
  | Free -> "free"
  | Signed -> "signed"
  | Top_tie -> "top tie"
  | Low_tie -> "low tie"
  | Grey -> "grey"
  | Black -> "black"

let permute k (a, b, c) =
  match k with
  | 0 -> (a, b, c)
  | 1 -> (a, c, b)
  | 2 -> (b, a, c)
  | 3 -> (b, c, a)
  | 4 -> (c, a, b)
  | _ -> (c, b, a)

let colour =
  let open Gen in
  let+ kind = of_list [ Free; Signed; Top_tie; Low_tie; Grey; Black ]
  and+ a = float_range 0. 1.
  and+ b = float_range 0. 1.
  and+ c = float_range 0. 1.
  and+ p = int_range 0 5
  and+ neg = bool in
  let hi = Float.max a b and lo = Float.min a b in
  let x =
    match kind with
    | Free -> (a, b, c)
    | Signed -> ((2. *. a) -. 1., (2. *. b) -. 1., (2. *. c) -. 1.)
    | Top_tie -> permute p (hi, hi, lo)
    | Low_tie -> permute p (hi, lo, lo)
    | Grey -> if neg then (-.a, -.a, -.a) else (a, a, a)
    | Black -> if neg then (-0., -0., -0.) else (0., 0., 0.)
  in
  (kind, x)

let colour =
  Gen.with_pp
    (fun ppf (k, (r, g, b)) ->
      Format.fprintf ppf "%s (%h, %h, %h)" (kind_name k) r g b)
    colour

let covered kind =
  List.iter
    (fun k -> cover ~share:0.1 (kind_name k) (k = kind))
    [ Free; Signed; Top_tie; Low_tie; Grey; Black ]

let magnitude (r, g, b) =
  Float.max (Float.abs r) (Float.max (Float.abs g) (Float.abs b))

(* Luma and luminance *)

type fn = { f : 's 'd. (float, 's, 'd) Nx.t -> (float, 's, 'd) Nx.t }

let weights =
  [
    ("luma", { f = V.Colour.luma }, (0.299, 0.587, 0.114));
    ( "luminance",
      { f = V.Colour.luminance },
      (87098. /. 409605., 175762. /. 245763., 12673. /. 175545.) );
  ]

let grey_values =
  [ 0.; -0.; 1.; -1.; 0.5; 1e-30; 3.; 255.; 6e4; -6e4; 1e300; 4e-324 ]

let greys_exact =
  cases "a grey's luma and luminance are that grey"
    ~id:(fun (F (n, _)) -> n)
    all_floats
    (fun (F (_, dt)) ->
      let x = colours dt (List.map (fun v -> (v, v, v)) grey_values) in
      let g = floats (Nx.slice [ Nx.A; Nx.I 0 ] x) in
      let same a b = Float.equal a b || (a = 0. && b = 0.) in
      List.iter
        (fun (name, { f }, _) ->
          let y = floats (f x) in
          Array.iteri
            (fun i v ->
              if not (same v y.(i)) then
                failf "%s of grey %h is %h" name v y.(i))
            g)
        weights)

let greys_compiled =
  test "greys are exact compiled as eager" (fun () ->
      let x =
        Nx.cast Nx.float32
          (Nx.create Nx.float64 [| 1001; 1 |]
             (Array.init 1001 (fun i -> Float.of_int (i - 500) /. 37.)))
      in
      let x = Nx.broadcast_to [| 1001; 3 |] x |> Nx.copy in
      let want = floats (Nx.slice [ Nx.A; Nx.I 0 ] x) in
      List.iter
        (fun (name, { f }, _) ->
          equal ~msg:name (array float_exact) want
            (floats (Rune.jit' f x |> Nx.reshape [| 1001 |])))
        weights)

(* Where the form is not finite, the sum: its limit at infinities, and in
   float16 past the range of [r - g]. *)
let limits =
  test "at infinities and past the range, the sum" (fun () ->
      let inf = Float.infinity in
      let x =
        colours Nx.float64
          [
            (inf, inf, inf);
            (inf, 0.5, 0.2);
            (-.inf, -.inf, 0.);
            (inf, -.inf, 0.);
          ]
      in
      List.iter
        (fun (name, { f }, _) ->
          equal ~msg:name (array float_exact)
            [| inf; inf; -.inf; Float.nan |]
            (floats (f x)))
        weights;
      let x = colours Nx.float16 [ (6e4, -6e4, 0.) ] in
      let r = Nx.slice [ Nx.A; Nx.I 0 ] x and g = Nx.slice [ Nx.A; Nx.I 1 ] x in
      let sum = Nx.add (Nx.mul_s r 0.299) (Nx.mul_s g 0.587) in
      equal (array float_exact) (floats sum) (floats (V.Colour.luma x)))

(* Agreement with the weighted sum in float64. Each of the form's five roundings
   and the weights' own errs by at most [u] times a term no larger than [2 M],
   [M] the largest magnitude: under [8 u M] in all, with float64's reference
   adding [3 u M] at float64. *)
let close_to_sum =
  prop "luma and luminance are the weighted sum within 8 u M" ~count:300
    Gen.(pair colour (of_list wide))
    (fun ((kind, c), F (_, dt)) ->
      covered kind;
      let x = colours dt [ c ] in
      let r, g, b =
        match floats x with [| r; g; b |] -> (r, g, b) | _ -> assert false
      in
      let m = magnitude (r, g, b) in
      let u = eps dt /. 2. in
      List.iter
        (fun (name, { f }, (wr, wg, wb)) ->
          let want = (wr *. r) +. (wg *. g) +. (wb *. b) in
          let bound = (8. *. u *. m) +. (3. *. epsilon_float *. m) in
          let got = (floats (f x)).(0) in
          if Float.abs (got -. want) > bound then
            failf "%s of (%h, %h, %h) is %h, not %h within %h" name r g b got
              want bound)
        weights)

let derivative_weights =
  test "the derivative is the weights, at a grey of infinities too" (fun () ->
      List.iter
        (fun (name, { f }, (wr, wg, wb)) ->
          List.iter
            (fun c ->
              let g = Rune.grad' (fun x -> Nx.sum (f x)) (one c) in
              equal ~msg:name
                (array (float 1e-15))
                [| wr; wg; wb |] (Nx.to_array g))
            [
              (0.2, 0.7, 0.4); (Float.infinity, Float.infinity, Float.infinity);
            ])
        weights)

(* HSV *)

(* [near tol want got] holds where both are NaN, equal, or within [tol]. *)
let near tol want got =
  Array.iteri
    (fun i w ->
      let g = got.(i) in
      let ok =
        (Float.is_nan w && Float.is_nan g) || w = g || Float.abs (w -. g) <= tol
      in
      if not ok then failf "element %d is %h, not %h" i g w)
    want

let hsv c = floats (V.Colour.to_hsv (one c))

let named_hsv =
  cases "the hexagon's corners, greys and black"
    ~id:(fun (n, _, _) -> n)
    [
      ("red", (1., 0., 0.), [| 0.; 1.; 1. |]);
      ("yellow", (1., 1., 0.), [| 1. /. 6.; 1.; 1. |]);
      ("green", (0., 1., 0.), [| 1. /. 3.; 1.; 1. |]);
      ("cyan", (0., 1., 1.), [| 0.5; 1.; 1. |]);
      ("blue", (0., 0., 1.), [| 2. /. 3.; 1.; 1. |]);
      ("magenta", (1., 0., 1.), [| 5. /. 6.; 1.; 1. |]);
      ("grey", (0.4, 0.4, 0.4), [| 0.; 0.; 0.4 |]);
      ("black", (0., 0., 0.), [| 0.; 0.; 0. |]);
      ("below black", (0., -0.5, -0.2), [| 0.9; Float.infinity; 0. |]);
      ( "infinite",
        (Float.infinity, 0.5, 0.2),
        [| Float.nan; Float.nan; Float.infinity |] );
    ]
    (fun (_, c, want) -> near 1e-15 want (hsv c))

let hue_range =
  prop "the hue lies in [0, 1)" ~count:300
    Gen.(pair colour (of_list wide))
    (fun ((kind, c), F (_, dt)) ->
      covered kind;
      let h = (floats (V.Colour.to_hsv (colours dt [ c ]))).(0) in
      if not (h >= 0. && h < 1.) then failf "hue %h" h)

(* float32 (1, 0, 1e-9): the hue [-1e-9 / 6] wraps to [1 - 1.7e-10], which
   rounds to [1] and takes [1] off, keeping its derivative. *)
let hue_wrap =
  test "a hue that rounds to 1 wraps to 0 with its derivative" (fun () ->
      let x = Nx.create Nx.float32 [| 3 |] [| 1.; 0.; 1e-9 |] in
      let hue x = Nx.slice [ Nx.I 0 ] (V.Colour.to_hsv x) in
      equal float_exact 0. (Nx.item [] (hue x));
      equal
        (array (float 1e-7))
        [| 1e-9 /. 6.; 1. /. 6.; -1. /. 6. |]
        (floats (Rune.grad' hue x)))

(* The hue against central differences in float64, at ties of the greatest and
   of the least channel away from red, where differences straddle the wrap, and
   at free colours away from greys and red. *)
let hue_differences =
  let tied =
    [
      (0.7, 0.7, 0.2);
      (0.2, 0.7, 0.7);
      (0.7, 0.2, 0.7);
      (0.3, 0.8, 0.3);
      (0.3, 0.3, 0.8);
      (0.9, 0.4, 0.9);
      (-0.2, -0.2, -0.6);
    ]
  in
  prop "the hue's derivative is its central difference" ~count:200
    ~examples:tied
    Gen.(triple (float_range 0. 1.) (float_range 0. 1.) (float_range 0. 1.))
    (fun c ->
      let h = (hsv c).(0) in
      let r, g, b = c in
      let v = Float.max r (Float.max g b) and m = Float.min r (Float.min g b) in
      assume (v -. m > 0.05);
      assume (h > 0.01 && h < 0.99);
      let d = 1e-9 in
      let at k s =
        let r, g, b = c in
        let c =
          match k with
          | 0 -> (r +. s, g, b)
          | 1 -> (r, g +. s, b)
          | _ -> (r, g, b +. s)
        in
        (hsv c).(0)
      in
      let fd = Array.init 3 (fun k -> (at k d -. at k (-.d)) /. (2. *. d)) in
      let got =
        Rune.grad' (fun x -> Nx.slice [ Nx.I 0 ] (V.Colour.to_hsv x)) (one c)
      in
      equal (array (float 1e-6)) fd (Nx.to_array got))

(* The round trip within [16 ε] of the largest channel's magnitude, every colour
   with [v ≠ 0] or grey. *)
let round_trip =
  prop "of_hsv inverts to_hsv within 16 ε" ~count:400
    Gen.(pair colour (of_list wide))
    (fun ((kind, c), F (_, dt)) ->
      covered kind;
      let x = colours dt [ c ] in
      let xs = floats x in
      let v = Array.fold_left Float.max Float.neg_infinity xs in
      let m = Array.fold_left Float.min Float.infinity xs in
      assume (v <> 0. || v = m);
      let y = floats (V.Colour.of_hsv (V.Colour.to_hsv x)) in
      let bound = 16. *. eps dt *. magnitude (xs.(0), xs.(1), xs.(2)) in
      Array.iteri
        (fun i want ->
          if Float.abs (y.(i) -. want) > bound then
            failf "channel %d of (%h, %h, %h) comes back %h" i xs.(0) xs.(1)
              xs.(2) y.(i))
        xs)

let identity = [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |]
let third = Array.make 9 (1. /. 3.)

(* [on_edge h] is whether [6 h], rounded, lies on a sextant's edge: [fmod (6 h)
   6], as [of_hsv] computes it, is an integer. *)
let on_edge h =
  let t = Float.rem (6. *. h) 6. in
  Float.is_integer t

let jacobian c =
  Rune.jacfwd' (fun x -> V.Colour.of_hsv (V.Colour.to_hsv x)) (one c)

(* The round trip's Jacobian is the identity, at ties too, and [(1/3)·ones] at a
   grey, in float64, but where the hue rounds onto a sextant's edge. Free
   colours keep [v - m] above [1e-3], since the hue's derivative grows as [1 /
   (v - m)] and its rounding with it. *)
let round_trip_jacobian =
  prop "the round trip's derivative is the identity" ~count:300 colour
    (fun (kind, ((r, g, b) as c)) ->
      covered kind;
      let v = Float.max r (Float.max g b) and m = Float.min r (Float.min g b) in
      let grey = v = m in
      assume (grey || (v <> 0. && v -. m > 1e-3));
      let h = Nx.item [ 0 ] (V.Colour.to_hsv (one c)) in
      if grey || not (on_edge h) then
        equal
          (array (float 1e-10))
          (if grey then third else identity)
          (Nx.to_array (jacobian c)))

(* Colours [d] ulps, 1 to 8, from a tie of the greatest or the least channel.
   Their hue lies a little off a sextant's edge [e], toward the home of the
   larger of the two near channels, red at [6 h = 0], green [2], blue [4]; [e]
   is the odd integer between the two homes for the greatest, and the third
   channel's home for the least. Where the hue's rounding keeps it on that side,
   the round trip's derivative is the identity; where it rounds onto [e] or
   across, it is not, as [of_hsv] states. [v - m] is at least [0.05], so each
   derivative is under [20] and rounds to within [1e-12]. *)
let near_ties =
  let gen =
    let open Gen in
    let+ hi = float_range 0.2 1.
    and+ lo = float_range 0. 0.15
    and+ d = int_range 1 8
    and+ up = bool
    and+ top = bool
    and+ k = int_range 0 2
    and+ swap = bool in
    let rec step n x =
      if n = 0 then x
      else step (n - 1) (if up then Float.succ x else Float.pred x)
    in
    let i, j =
      if swap then ((k + 2) mod 3, (k + 1) mod 3)
      else ((k + 1) mod 3, (k + 2) mod 3)
    in
    let c = Array.make 3 0. in
    let near = if top then hi else lo in
    c.(k) <- (if top then lo else hi);
    c.(i) <- near;
    c.(j) <- step d near;
    (top, c, i, j)
  in
  let home n = 2. *. Float.of_int n in
  (* [around a] is [a] read on the circle of [6], in [(-3, 3]]. *)
  let around a =
    let a = Float.rem a 6. in
    if a > 3. then a -. 6. else if a <= -3. then a +. 6. else a
  in
  prop "colours ulps from a tie keep the identity off their edge" ~count:400 gen
    (fun (top, c, i, j) ->
      cover ~share:0.3 "greatest" top;
      cover ~share:0.3 "least" (not top);
      let k = 3 - i - j in
      let e =
        if top then home i +. (around (home j -. home i) /. 2.) else home k
      in
      let larger = if c.(i) > c.(j) then i else j in
      let toward = around (home larger -. e) in
      let rgb = (c.(0), c.(1), c.(2)) in
      let h = Nx.item [ 0 ] (V.Colour.to_hsv (one rgb)) in
      let off = around ((6. *. h) -. e) in
      let kept = off <> 0. && Float.sign_bit off = Float.sign_bit toward in
      cover ~share:0.2 "kept its side" kept;
      if kept then
        equal (array (float 1e-12)) identity (Nx.to_array (jacobian rgb)))

(* [slope k] is [c]'s slope in [k] off its kinks: [c] rises on [(0, 1)] and
   falls on [(3, 4)]. *)
let slope k =
  let k = Float.rem (k +. 12.) 6. in
  if k > 0. && k < 1. then 1. else if k > 3. && k < 4. then -1. else 0.

(* Hues an ulp either side of each sextant's edge, and as far from red's as
   [1e-17]: the derivative in the hue of channel [n] is [-6 v s] times [c]'s
   slope on the side the hue lies on, and the mean of both sides where [6 h]
   rounds to an integer. *)
let sextant_edges =
  let edges = List.init 7 (fun j -> Float.of_int j /. 6.) in
  let hues =
    [ 1e-17; -1e-17 ]
    @ List.concat_map (fun e -> [ Float.pred e; Float.succ e ]) edges
  in
  cases "the hue's derivative is the side's, or the mean on an edge"
    ~id:(Printf.sprintf "%h") hues (fun h ->
      let s = 0.6 and v = 0.8 in
      let t = Float.rem (6. *. h) 6. in
      let t = if t < 0. then t +. 6. else t in
      let want =
        Array.map
          (fun n ->
            let k = n +. Float.round t in
            let d =
              if Float.is_integer t then
                (slope (k -. 0.5) +. slope (k +. 0.5)) /. 2.
              else slope (if t < Float.round t then k -. 0.5 else k +. 0.5)
            in
            -6. *. v *. s *. d)
          [| 5.; 3.; 1. |]
      in
      let rgb h =
        V.Colour.of_hsv
          (Nx.stack [ h; Nx.scalar Nx.float64 s; Nx.scalar Nx.float64 v ])
      in
      equal
        (array (float 1e-12))
        want
        (Nx.to_array (Rune.jacfwd' rgb (Nx.scalar Nx.float64 h))))

(* Hues on a grid of 64ths shift by whole turns exactly. *)
let whole_turns =
  prop "a hue and the hue a whole number of turns away agree"
    Gen.(
      quad (int_range 0 63) (int_range (-3) 3) (float_range 0. 1.)
        (float_range 0. 1.))
    (fun (j, k, s, v) ->
      let at h = floats (V.Colour.of_hsv (one (h, s, v))) in
      let h = Float.of_int j /. 64. in
      equal (array float_exact) (at h) (at (h +. Float.of_int k)))

(* At a grey the hue and the saturation are [0], with derivative [0]. *)
let grey_derivative =
  cases "a grey's hue and saturation have derivative 0"
    ~id:(fun (v, (name, _)) -> Printf.sprintf "%s %g" name v)
    (List.concat_map
       (fun v -> [ (v, ("hue", 0)); (v, ("saturation", 1)) ])
       [ 0.4; 0.; -0.3; 1. ])
    (fun (v, (_, k)) ->
      let g =
        Rune.grad'
          (fun x -> Nx.slice [ Nx.I k ] (V.Colour.to_hsv x))
          (one (v, v, v))
      in
      equal (array (float 0x1p-1074)) [| 0.; 0.; 0. |] (Nx.to_array g))

(* Every sign pattern of infinite channels, beside a finite one or not: hue and
   saturation NaN, value the greatest channel. *)
let infinities =
  let parts =
    [ ("finite", 0.3); ("+inf", Float.infinity); ("-inf", Float.neg_infinity) ]
  in
  let rows =
    List.concat_map
      (fun a ->
        List.concat_map (fun b -> List.map (fun c -> (a, b, c)) parts) parts)
      parts
    |> List.filter (fun ((_, a), (_, b), (_, c)) ->
        not (Float.is_finite a && Float.is_finite b && Float.is_finite c))
  in
  cases "a colour with an infinite channel has NaN hue and saturation"
    ~id:(fun ((a, _), (b, _), (c, _)) -> String.concat " " [ a; b; c ])
    rows
    (fun ((_, r), (_, g), (_, b)) ->
      let r = if Float.is_finite r then 0.2 else r
      and b = if Float.is_finite b then 0.5 else b in
      equal (array float_exact)
        [| Float.nan; Float.nan; Float.max r (Float.max g b) |]
        (hsv (r, g, b)))

let nan_in =
  let nan = Float.nan and inf = Float.infinity in
  cases "a NaN gives NaN in every channel"
    ~id:(fun (n, _, _) -> n)
    [
      ("to_hsv red", V.Colour.to_hsv, (nan, 0.5, 0.2));
      ("to_hsv green", V.Colour.to_hsv, (0.5, nan, 0.2));
      ("to_hsv blue", V.Colour.to_hsv, (0.5, 0.2, nan));
      ("of_hsv hue", V.Colour.of_hsv, (nan, 0.5, 0.5));
      ("of_hsv saturation", V.Colour.of_hsv, (0.3, nan, 0.5));
      ("of_hsv value", V.Colour.of_hsv, (0.3, 0.5, nan));
      ("of_hsv infinite hue", V.Colour.of_hsv, (inf, 0.5, 0.5));
      ("of_hsv infinite saturation", V.Colour.of_hsv, (0.3, inf, 0.5));
      ("of_hsv infinite value", V.Colour.of_hsv, (0.3, 0.5, -.inf));
      ("linear_to_lab red", V.Colour.linear_to_lab, (nan, 0.5, 0.2));
      ("linear_to_lab blue", V.Colour.linear_to_lab, (0.5, 0.2, nan));
      ("lab_to_linear a", V.Colour.lab_to_linear, (50., nan, 0.));
      ("lab_to_linear b", V.Colour.lab_to_linear, (50., 0., nan));
    ]
    (fun (_, f, c) ->
      equal (array float_exact) [| nan; nan; nan |] (floats (f (one c))))

(* Against scikit-image and torchvision, from golden/colour.npz
   (contrib/mynd/test/gen/colour.py). Both compute the same hexcone with other
   roundings: within 2e-15, about [9 u] on values in [0, 1], the hue read around
   the circle. *)
let golden = lazy (Nx_io.load_npz "golden/colour.npz")

let entry key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack Nx.float64 p
  | None -> failf "golden/colour.npz has no %s" key

let circular_tol tol want got =
  Array.iteri
    (fun i w ->
      let d = Float.abs (got.(i) -. w) in
      let d = if i mod 3 = 0 then Float.min d (1. -. d) else d in
      if d > tol then failf "element %d is %h, not %h" i got.(i) w)
    want

let circular = circular_tol 2e-15

let libraries =
  cases "agrees with scikit-image and torchvision"
    ~id:(fun (n, _, _, _) -> n)
    [
      ("to_hsv, scikit-image", V.Colour.to_hsv, "hsv.x", "hsv.skimage");
      ("to_hsv, torchvision", V.Colour.to_hsv, "hsv.x", "hsv.torchvision");
      ("of_hsv, scikit-image", V.Colour.of_hsv, "rgb.y", "rgb.skimage");
      ("of_hsv, torchvision", V.Colour.of_hsv, "rgb.y", "rgb.torchvision");
    ]
    (fun (name, f, input, want) ->
      let got = floats (f (entry input)) in
      let want = Nx.to_array (entry want) in
      if String.starts_with ~prefix:"rgb_to" name then circular want got
      else equal (array (float 2e-15)) want got)

(* Lab *)

let white_black =
  cases "white is (100, 0, 0) and black (0, 0, 0)"
    ~id:(fun (F (n, _)) -> n)
    wide
    (fun (F (_, dt)) ->
      let x = colours dt [ (1., 1., 1.); (0., 0., 0.) ] in
      let lab = [| 100.; 0.; 0.; 0.; 0.; 0. |] in
      equal (array float_exact) lab (floats (V.Colour.linear_to_lab x));
      equal (array float_exact)
        [| 1.; 1.; 1.; 0.; 0.; 0. |]
        (floats
           (V.Colour.lab_to_linear
              (colours dt [ (100., 0., 0.); (0., 0., 0.) ]))))

let grey_axis =
  prop "a grey has a* and b* 0"
    Gen.(pair (float_range (-1.) 2.) (of_list wide))
    (fun (v, F (_, dt)) ->
      let lab = floats (V.Colour.linear_to_lab (colours dt [ (v, v, v) ])) in
      equal (array float_exact) [| 0.; 0. |] [| lab.(1) +. 0.; lab.(2) +. 0. |];
      let back =
        floats (V.Colour.lab_to_linear (colours dt [ (lab.(0), 0., 0.) ]))
      in
      equal (array float_exact) [| back.(0) |] [| back.(1) |];
      equal (array float_exact) [| back.(0) |] [| back.(2) |])

(* Against the exact values: [a*]'s [500 (f X - f Y)] multiplies each [f]'s few
   roundings by [500], so float64 is within [1e-12]. *)
let lab_exact =
  cases "Lab is CSS Color 4's rationals rounded"
    ~id:(fun (n, _, _, _) -> n)
    [
      ("linear_to_lab", V.Colour.linear_to_lab, "lab.x", "lab.y");
      ("lab_to_linear", V.Colour.lab_to_linear, "linear.y", "linear.x");
    ]
    (fun (_, f, input, want) ->
      equal
        (array (float 1e-12))
        (Nx.to_array (entry want))
        (floats (f (entry input))))

(* Within [64 ε] of the colour's largest channel or [1]: [M] and its inverse
   rounded into the dtype multiply to the identity within about [10 ε], and each
   step rounds a few times more. *)
let lab_round_trip =
  prop "lab_to_linear inverts linear_to_lab" ~count:300
    Gen.(
      pair colour
        (of_list [ F ("float32", Nx.float32); F ("float64", Nx.float64) ]))
    (fun ((kind, c), F (_, dt)) ->
      covered kind;
      let x = colours dt [ c ] in
      let xs = floats x in
      let y = floats (V.Colour.lab_to_linear (V.Colour.linear_to_lab x)) in
      let bound = 64. *. eps dt *. Float.max 1. (magnitude c) in
      Array.iteri
        (fun i want ->
          if Float.abs (y.(i) -. want) > bound then
            failf "channel %d of (%h, %h, %h) comes back %h" i xs.(0) xs.(1)
              xs.(2) y.(i))
        xs)

(* Near black, within [64 ε] of the colour's largest channel, and [64 η] of the
   least subnormal [η] where the colour is subnormal: [L*], [a*], [b*] and the
   inverse take their linear pieces without adding and taking back [16 / 116],
   so a colour down to [10⁻⁸] keeps its relative accuracy. *)
let lab_near_black =
  prop "Lab's round trip is relative near black" ~count:300
    Gen.(
      triple colour (float_range 0. 8.)
        (of_list [ F ("float32", Nx.float32); F ("float64", Nx.float64) ]))
    (fun ((kind, (r, g, b)), e, F (_, dt)) ->
      covered kind;
      let k = 10. ** -.e in
      let x = colours dt [ (k *. r, k *. g, k *. b) ] in
      let xs = floats x in
      let m = magnitude (xs.(0), xs.(1), xs.(2)) in
      cover ~share:0.3 "below ε" (m < 216. /. 24389.);
      let f = Nx.Dtype.float_format dt in
      let eta = f.min_normal *. f.epsilon in
      let y = floats (V.Colour.lab_to_linear (V.Colour.linear_to_lab x)) in
      Array.iteri
        (fun i want ->
          if Float.abs (y.(i) -. want) > (64. *. eps dt *. m) +. (64. *. eta)
          then
            failf "channel %d of (%h, %h, %h) comes back %h" i xs.(0) xs.(1)
              xs.(2) y.(i))
        xs)

(* [L*] of a grey [v] is [κ v] below [ε] and [116 v^⅓ - 16] above, with slope
   [κ] at [ε] from both sides. *)
let lab_knee =
  test "L* is differentiable across ε" (fun () ->
      let epsilon = 216. /. 24389. and kappa = 24389. /. 27. in
      List.iter
        (fun v ->
          let x = Nx.create Nx.float64 [| 1 |] [| v |] in
          let l x =
            Nx.slice [ Nx.A; Nx.I 0 ]
              (V.Colour.linear_to_lab
                 (Nx.broadcast_to [| 1; 3 |] (Nx.reshape [| 1; 1 |] x)))
          in
          equal
            (float_rel ~rel:1e-6 ~abs:0.)
            kappa
            (Nx.item [ 0 ] (Rune.grad' (fun x -> Nx.sum (l x)) x)))
        [ epsilon *. (1. -. 1e-9); epsilon; epsilon *. (1. +. 1e-9) ])

let lab_differences =
  prop "Lab's derivatives are their central differences" ~count:100
    Gen.(triple (float_range 0. 1.) (float_range 0. 1.) (float_range 0. 1.))
    (fun c ->
      let d = 1e-7 in
      List.iter
        (fun (name, f) ->
          let x = one c in
          let j = Nx.to_array (Rune.jacfwd' f x) in
          let at k s =
            let e = Array.init 3 (fun i -> if i = k then s else 0.) in
            floats (f (Nx.add x (Nx.create Nx.float64 [| 3 |] e)))
          in
          let fd =
            Array.init 9 (fun i ->
                let row = i / 3 and col = i mod 3 in
                ((at col d).(row) -. (at col (-.d)).(row)) /. (2. *. d))
          in
          equal ~msg:name (array (float_rel ~rel:1e-5 ~abs:1e-5)) fd j)
        [
          ("linear_to_lab", V.Colour.linear_to_lab);
          ("lab_to_linear", fun x -> V.Colour.lab_to_linear (Nx.mul_s x 50.));
        ])

(* Compiled. A compiler may fuse a product into a sum, which rounds once less:
   float32 within [8 ε] of each function's scale, [1] or Lab's [100]. *)
let compiled =
  let read tol want got = equal (array (float tol)) want got in
  cases "compiled is eager to rounding"
    ~id:(fun (n, _, _, _, _) -> n)
    [
      ("luma", V.Colour.luma, "hsv.x", 1., read);
      ("luminance", V.Colour.luminance, "hsv.x", 1., read);
      ("to_hsv", V.Colour.to_hsv, "hsv.x", 1., circular_tol);
      ("of_hsv", V.Colour.of_hsv, "rgb.y", 1., read);
      ("linear_to_lab", V.Colour.linear_to_lab, "lab.x", 100., read);
      ("lab_to_linear", V.Colour.lab_to_linear, "linear.y", 1., read);
    ]
    (fun (_, f, input, scale, check) ->
      let x = Nx.cast Nx.float32 (entry input) in
      check
        (8. *. eps Nx.float32 *. scale)
        (floats (f x))
        (floats (Rune.jit' f x)))

(* Shapes *)

let shapes =
  test "any rank, a last axis of 3" (fun () ->
      equal (array int) [| 1 |] (Nx.shape (V.Colour.luma (one (0.1, 0.2, 0.3))));
      let x = Nx.zeros Nx.float32 [| 2; 4; 5; 3 |] in
      equal (array int) [| 2; 4; 5; 1 |] (Nx.shape (V.Colour.luminance x));
      equal (array int) [| 2; 4; 5; 3 |] (Nx.shape (V.Colour.to_hsv x));
      equal (array int) [| 0; 3 |]
        (Nx.shape (V.Colour.linear_to_lab (Nx.zeros Nx.float64 [| 0; 3 |]))))

let refusals =
  let fs =
    [
      ("Mynd.Colour.luma", V.Colour.luma);
      ("Mynd.Colour.luminance", V.Colour.luminance);
      ("Mynd.Colour.to_hsv", V.Colour.to_hsv);
      ("Mynd.Colour.of_hsv", V.Colour.of_hsv);
      ("Mynd.Colour.linear_to_lab", V.Colour.linear_to_lab);
      ("Mynd.Colour.lab_to_linear", V.Colour.lab_to_linear);
    ]
  in
  cases "a last axis other than 3 raises"
    ~id:(fun ((n, _), s) ->
      Printf.sprintf "%s [%s]" n
        (String.concat "; " (List.map string_of_int (Array.to_list s))))
    (List.concat_map
       (fun f ->
         List.map (fun s -> (f, s)) [ [| 4; 1 |]; [| 2 |]; [| 3; 4 |]; [||] ])
       fs)
    (fun ((name, f), s) ->
      raises_match (Exn.invalid_arg ~sub:name) (fun () ->
          f (Nx.zeros Nx.float32 s)))

let () =
  exit
    (run "mynd.colour"
       [
         greys_exact;
         greys_compiled;
         limits;
         close_to_sum;
         derivative_weights;
         named_hsv;
         hue_range;
         hue_wrap;
         hue_differences;
         round_trip;
         round_trip_jacobian;
         near_ties;
         sextant_edges;
         grey_derivative;
         infinities;
         whole_turns;
         nan_in;
         libraries;
         white_black;
         grey_axis;
         lab_exact;
         lab_round_trip;
         lab_near_black;
         lab_knee;
         lab_differences;
         compiled;
         shapes;
         refusals;
       ])
