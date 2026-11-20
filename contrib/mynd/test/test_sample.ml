(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
module V = Mynd

(* torch's grid_sample, from golden/sample.npz
   (contrib/mynd/test/gen/sample.py) *)
let golden = lazy (Nx_io.load_npz "golden/sample.npz")

let entry key =
  match Nx_io.Archive.find key (Lazy.force golden) with
  | Some p -> Nx.unpack Nx.float64 p
  | None -> failf "golden/sample.npz has no %s" key

let floats x = Nx.to_array (Nx.cast Nx.float64 x)
let same_shape want got = equal (array int) (Nx.shape want) (Nx.shape got)

let close tol want got =
  same_shape want got;
  equal (array (float tol)) (floats want) (floats got)

(* NaN equals NaN here: bit-for-bit comparison of float arrays. *)
let bits x =
  Array.map
    (fun v -> if Float.is_nan v then 0x7ff8L else Int64.bits_of_float v)
    (floats x)

let same want got =
  same_shape want got;
  equal (array int64) (bits want) (bits got)

type flt = F : string * (float, _) Nx.dtype -> flt

let narrow =
  [
    F ("float16", Nx.float16);
    F ("bfloat16", Nx.bfloat16);
    F ("float32", Nx.float32);
  ]

let kernels : (string * V.Sample.kernel) list =
  [
    ("box", `Box);
    ("triangle", `Triangle);
    ("keys -0.5", `Cubic (-0.5));
    ("keys -0.75", `Cubic (-0.75));
    ("lanczos2", `Lanczos 2);
    ("lanczos3", `Lanczos 3);
  ]

let half_width : V.Sample.kernel -> float = function
  | `Box -> 0.5
  | `Triangle -> 1.
  | `Cubic _ -> 2.
  | `Lanczos n -> Float.of_int n

let extensions : (string * V.extension) list =
  [
    ("zero", `Zero);
    ("replicate", `Replicate);
    ("symmetric whole", `Symmetric_whole);
    ("symmetric half", `Symmetric_half);
    ("periodic", `Periodic);
  ]

let edges : (string * V.Sample.edge) list =
  List.map (fun (n, e) -> (n, (e :> V.Sample.edge))) extensions
  @ [ ("renormalise", `Renormalise) ]

(* Against torch's grid_sample in float64 *)

let torch_rows =
  [ ("bilinear", `Triangle); ("bicubic", `Cubic (-0.75)) ]
  |> List.concat_map (fun (m, k) ->
      [
        (m, k, "zeros", ("zero", `Zero));
        (m, k, "border", ("replicate", `Replicate));
        (m, k, "reflection", ("symmetric half", `Symmetric_half));
      ])

(* Positions on multiples of 1/16, where torch's normalised grid holds them
   exactly; the centres among them check the derivative from the right. *)
let torch =
  cases "grid_sample's values and derivatives"
    ~id:(fun (m, _, pad, _) -> m ^ " " ^ pad)
    torch_rows
    (fun (m, k, pad, (_, edge)) ->
      let key s = Printf.sprintf "%s.%s.%s" m pad s in
      let x = entry "x" and p = entry "p" and r = entry "r" in
      close 1e-13 (entry (key "y")) (V.Sample.at ~edge k x p);
      let loss x p = Nx.sum (Nx.mul (V.Sample.at ~edge k x p) r) in
      let gx = Rune.grad' (fun x -> loss x p) x in
      let gp = Rune.grad' (fun p -> loss x p) p in
      close 1e-13 (entry (key "gx")) gx;
      (* torch clamps or reflects the coordinate under border and reflection,
         and takes its derivative from the right there: at a centre outside the
         image or at its outermost ones, its derivative is the left's or 0 where
         this one is the right's. Those points are left out. *)
      let pts = floats p in
      let outer t n = Float.is_integer (t -. 0.5) && (t < 1. || t > n -. 1.) in
      let edge_centre i =
        pad <> "zeros" && (outer pts.(2 * i) 8. || outer pts.((2 * i) + 1) 4.)
      in
      let keep a =
        List.filteri (fun j _ -> not (edge_centre (j / 2))) (Array.to_list a)
      in
      equal
        (list (float 1e-12))
        (keep (floats (entry (key "gp"))))
        (keep (floats gp)))

(* The same values in float16 and bfloat16, the positions exact there, to their
   rounding: values up to about 4, 16 taps' terms. *)
let torch_narrow =
  cases "grid_sample's values in float16 and bfloat16"
    ~id:(fun (n, _) -> n)
    [ ("float16", 0.02); ("bfloat16", 0.15) ]
    (fun (n, tol) ->
      let (F (_, dt)) = List.find (fun (F (m, _)) -> m = n) narrow in
      let x = Nx.cast dt (entry "x") and p = Nx.cast dt (entry "p") in
      List.iter
        (fun (m, k) ->
          close tol (entry (m ^ ".zeros.y")) (V.Sample.at ~edge:`Zero k x p))
        [ ("bilinear", `Triangle); ("bicubic", `Cubic (-0.75)) ])

(* Laws *)

let gen_position ~h ~w =
  Gen.(
    let+ u =
      frequency
        [
          (3, float_range (-9.) (Float.of_int w +. 9.));
          ( 1,
            map (fun k -> Float.of_int k /. 2.) (int_range (-12) ((2 * w) + 12))
          );
        ]
    and+ v =
      frequency
        [
          (3, float_range (-9.) (Float.of_int h +. 9.));
          ( 1,
            map (fun k -> Float.of_int k /. 2.) (int_range (-12) ((2 * h) + 12))
          );
        ]
    in
    (u, v))

let image ~h ~w ~c seed =
  let st = Random.State.make [| seed |] in
  Nx.init Nx.float64 [| h; w; c |] (fun _ -> Random.State.float st 2. -. 1.)

let at p = Nx.create Nx.float64 [| 1; 2 |] [| fst p; snd p |]

let nearest_is_box =
  prop "nearest is sample under Box at finite positions" ~count:300
    Gen.(pair (int_range 0 4) (gen_position ~h:3 ~w:4))
    (fun (e, p) ->
      let _, edge = List.nth extensions e in
      let x = image ~h:3 ~w:4 ~c:2 1 in
      cover "outside" (fst p < 0. || fst p >= 4. || snd p < 0. || snd p >= 3.);
      cover "on a cell edge" (Float.is_integer (fst p));
      same (V.Sample.at ~edge `Box x (at p)) (V.Sample.nearest ~edge x (at p)))

(* In the narrow floats too, where positions round coarsely near cell edges: a
   quarter of the positions are [-2⁻ᵏ], just left of the image. *)
let nearest_is_box_narrow =
  prop "nearest is sample under Box in the narrow floats" ~count:300
    Gen.(
      triple (int_range 0 2) (int_range 0 4)
        (frequency
           [
             (3, gen_position ~h:3 ~w:4);
             (1, map (fun k -> (-.Float.ldexp 1. (-k), 0.5)) (int_range 1 30));
           ]))
    (fun (d, e, (u, v)) ->
      let (F (_, dt)) = List.nth narrow d in
      let _, edge = List.nth extensions e in
      let x = Nx.cast dt (image ~h:3 ~w:4 ~c:2 1) in
      let p = Nx.cast dt (at (u, v)) in
      cover "just left of the image" (u < 0. && u > -1e-3);
      same (V.Sample.at ~edge `Box x p) (V.Sample.nearest ~edge x p))

(* A tap's argument is the exact difference rounded once, for [u] in [(-1, 0)]
   too, where [u + 1] rounds. Two columns read periodically, the left one 0 and
   the right 1, at the first row's centre: the result is the weight of the tap
   at column [-1], whose argument is [u + ½] rounded once into the dtype,
   divided by the sum of the two taps' weights, [1 - |τ|] each. The other tap is
   column [-2] below [u = -½], else column [0]. Each operation rounds once. *)
let argument_rounds_once =
  prop "an argument is the exact difference rounded once" ~count:300
    Gen.(pair (int_range 0 2) (float_range (-1.) 0.))
    (fun (d, u) ->
      let (F (_, dt)) = List.nth narrow d in
      let r = Nx.Dtype.of_float dt in
      let u = r u in
      assume (u > -1. && u < 0.);
      cover "u + 1 rounds" (r (u +. 1.) <> u +. 1.);
      let x = Nx.create dt [| 1; 2; 1 |] [| 0.; 1. |] in
      let p = Nx.create dt [| 1; 2 |] [| u; 0.5 |] in
      let k tau = r (1. -. Float.abs tau) in
      let read = k (r (u +. 0.5)) in
      let other = k (r (if u < -0.5 then u +. 1.5 else u -. 0.5)) in
      equal float_exact
        (r (read /. r (read +. other)))
        (Nx.item [ 0; 0 ]
           (Nx.cast Nx.float64 (V.Sample.at ~edge:`Periodic `Triangle x p))))

(* [extended e (lo, hi) x] is [x] extended under [e] by [lo] before and [hi]
   after its rows and its columns. *)
let extended e pad x = Nx.extend ~axis:0 e pad (Nx.extend ~axis:1 e pad x)

(* An edge rule reads what zero edges read of the image extended under it. *)
let edge_is_extend =
  prop "an edge rule reads as the image extended under it" ~count:300
    Gen.(triple (int_range 0 4) (int_range 0 5) (gen_position ~h:3 ~w:4))
    (fun (e, k, p) ->
      let _, edge = List.nth extensions e and _, k = List.nth kernels k in
      let pad = 20 in
      let x = image ~h:3 ~w:4 ~c:2 2 in
      (* On a grid of 1/1024, so that the shift is exact. *)
      let q x = Float.round (x *. 1024.) /. 1024. in
      let p = (q (fst p), q (snd p)) in
      let shifted = (fst p +. Float.of_int pad, snd p +. Float.of_int pad) in
      close 1e-12
        (V.Sample.at ~edge:`Zero k
           (extended (edge :> Nx.extension) (pad, pad) x)
           (at shifted))
        (V.Sample.at ~edge k x (at p)))

let interpolates =
  cases "an interpolating kernel reads the pixel at its centre" ~id:fst kernels
    (fun (_, k) ->
      let x = image ~h:3 ~w:5 ~c:2 3 in
      let p = Nx.reshape [| 15; 2 |] (V.Sample.centres Nx.float64 (3, 5)) in
      close 1e-15 (Nx.reshape [| 15; 2 |] x) (V.Sample.at ~edge:`Zero k x p))

let cell_edges =
  test "a cell edge reads the cell it starts" (fun () ->
      let x = Nx.create Nx.int32 [| 1; 4; 1 |] [| 10l; 11l; 12l; 13l |] in
      let p =
        Nx.create Nx.float64 [| 5; 2 |]
          [| 0.; 0.5; 1.; 0.5; 2.; 0.5; 2.999; 0.5; 4.; 0.5 |]
      in
      equal (array int32)
        [| 10l; 11l; 12l; 12l; 0l |]
        (Nx.to_array (V.Sample.nearest ~edge:`Zero x p));
      let x = Nx.cast Nx.float64 x in
      equal (array float_exact)
        [| 10.; 11.; 12.; 12.; 0. |]
        (floats (V.Sample.at ~edge:`Zero `Box x p)))

let non_finite =
  test "a non-finite position reads NaN, or zero by nearest" (fun () ->
      let x = Nx.add_s (image ~h:3 ~w:4 ~c:1 4) 5. in
      let p =
        Nx.create Nx.float64 [| 4; 2 |]
          [| Float.nan; 1.; 1.; infinity; neg_infinity; 1.; 1.5; 1.5 |]
      in
      List.iter
        (fun (_, edge) ->
          let s = floats (V.Sample.at ~edge `Triangle x p) in
          equal (array bool)
            [| true; true; true; false |]
            (Array.map Float.is_nan s))
        edges;
      List.iter
        (fun (_, edge) ->
          let n = floats (V.Sample.nearest ~edge x p) in
          equal (array float_exact) [| 0.; 0.; 0.; Nx.item [ 1; 1; 0 ] x |] n)
        extensions;
      let m = Nx.create Nx.bool [| 1; 1; 1 |] [| true |] in
      equal (array bool)
        [| false; false; false; true |]
        (Nx.to_array
           (V.Sample.nearest ~edge:`Replicate m
              (Nx.create Nx.float64 [| 4; 2 |]
                 [| Float.nan; 0.; 0.; infinity; neg_infinity; 0.; 0.5; 0.5 |]))))

let far_positions =
  test "positions far away fold, past 2^62 as at 2^62" (fun () ->
      let x = Nx.create Nx.float64 [| 1; 3; 1 |] [| 1.; 2.; 3. |] in
      let p a =
        Nx.create Nx.float64
          [| Array.length a; 2 |]
          (Array.concat (List.map (fun u -> [| u; 0.5 |]) (Array.to_list a)))
      in
      let read edge a = floats (V.Sample.nearest ~edge x (p a)) in
      let far = [| 1e30; -1e30; 3e6 +. 0.5; -3e6 -. 0.5 |] in
      equal (array float_exact) [| 3.; 1.; 3.; 1. |] (read `Replicate far);
      equal (array float_exact) [| 0.; 0.; 0.; 0. |] (read `Zero far);
      (* 2^62 is 1 modulo 3; 3e6 is a multiple of 3 and of 6. *)
      equal (array float_exact) [| 2.; 1.; 3. |]
        (read `Periodic [| 1e30; 3e6 +. 0.5; -3e6 -. 0.5 |]);
      equal (array float_exact) [| 1.; 1. |]
        (read `Symmetric_half [| 3e6 +. 0.5; -3e6 -. 0.5 |]);
      equal (array float_exact) (read `Periodic far)
        (floats (V.Sample.at ~edge:`Periodic `Box x (p far))))

let labels_keep_values =
  test "nearest moves labels and masks unchanged" (fun () ->
      let labels =
        Nx.create Nx.int64 [| 2; 2; 1 |] [| -1L; 7L; Int64.max_int; 3L |]
      in
      let p =
        Nx.create Nx.float32 [| 2; 2; 2 |]
          [| 0.5; 0.5; 1.5; 0.5; 0.5; 1.5; 1.7; 1.2 |]
      in
      equal (array int64)
        [| -1L; 7L; Int64.max_int; 3L |]
        (Nx.to_array (V.Sample.nearest ~edge:`Zero labels p));
      let mask =
        Nx.create Nx.bool [| 2; 2; 1 |] [| true; false; false; true |]
      in
      equal (array bool)
        [| true; false; false; true |]
        (Nx.to_array (V.Sample.nearest ~edge:`Zero mask p)))

(* The derivative in the positions against a central difference, away from the
   kernel's kinks, and the derivative in the image, which is linear. *)
let derivative =
  prop "derivatives against differences" ~count:150
    Gen.(triple (int_range 0 5) (int_range 1 5) (gen_position ~h:3 ~w:4))
    (fun (e, k, (u, v)) ->
      let _, edge = List.nth edges e and _, k = List.nth kernels k in
      let d = 1e-6 in
      let h = half_width k in
      let near t = Float.abs (t -. Float.round t) < 4. *. d in
      let near_kink t =
        near (t -. 0.5) || near (t -. 0.5 -. h) || near (t -. 0.5 +. h)
      in
      assume (not (near_kink u || near_kink v));
      let x = image ~h:3 ~w:4 ~c:2 5 in
      let f p = Nx.sum (V.Sample.at ~edge k x p) in
      let fd du dv =
        (Nx.item [] (f (at (u +. du, v +. dv)))
        -. Nx.item [] (f (at (u -. du, v -. dv))))
        /. (2. *. d)
      in
      let g = Nx.to_array (Rune.grad' f (at (u, v))) in
      equal (float_rel ~rel:1e-5 ~abs:1e-6) (fd d 0.) g.(0);
      equal (float_rel ~rel:1e-5 ~abs:1e-6) (fd 0. d) g.(1);
      (* Linear in the image: the derivative at each element is the sample of
         the image that is 1 there and 0 elsewhere. *)
      let gx =
        Nx.to_array
          (Rune.grad' (fun x -> Nx.sum (V.Sample.at ~edge k x (at (u, v)))) x)
      in
      let unit i =
        Nx.init Nx.float64 (Nx.shape x) (fun _ -> 0.)
        |> Nx.reshape [| 24 |] |> Nx.to_array
        |> Array.mapi (fun j _ -> if j = i then 1. else 0.)
        |> Nx.create Nx.float64 (Nx.shape x)
      in
      equal
        (array (float 1e-15))
        (Array.init 24 (fun i ->
             Nx.item [] (Nx.sum (V.Sample.at ~edge k (unit i) (at (u, v))))))
        gx)

let compiled =
  cases "compiled is eager" ~id:fst
    [
      ("triangle", `Triangle); ("keys", `Cubic (-0.5)); ("lanczos3", `Lanczos 3);
    ]
    (fun (_, k) ->
      let x = entry "x" and p = entry "p" in
      same
        (V.Sample.at ~edge:`Symmetric_whole k x p)
        (Rune.jit' (fun p -> V.Sample.at ~edge:`Symmetric_whole k x p) p);
      let x32 = Nx.cast Nx.float32 x and p32 = Nx.cast Nx.float32 p in
      same
        (V.Sample.at ~edge:`Periodic k x32 p32)
        (Rune.jit' (fun p -> V.Sample.at ~edge:`Periodic k x32 p) p32);
      same
        (V.Sample.nearest ~edge:`Replicate x p)
        (Rune.jit' (fun p -> V.Sample.nearest ~edge:`Replicate x p) p))

(* Batches *)

let batches =
  test "images and positions pair batch axes" (fun () ->
      let x = entry "x" and p = entry "p" in
      let one i = Nx.slice [ Nx.I i ] in
      let y =
        V.Sample.at ~edge:`Zero `Triangle x (Nx.slice [ Nx.R (0, 1) ] p)
      in
      equal (array int) [| 2; 48; 3 |] (Nx.shape y);
      List.iter
        (fun i ->
          same (V.Sample.at ~edge:`Zero `Triangle (one i x) (one 0 p)) (one i y))
        [ 0; 1 ];
      let q = Nx.reshape [| 2; 6; 8; 2 |] p in
      equal (array int) [| 2; 6; 8; 3 |]
        (Nx.shape (V.Sample.at ~edge:`Zero `Triangle x q)))

(* A batch padded with zeros reads under [`Zero] as each image alone; a batch of
   masks padded with false reads by nearest as each mask alone. *)
let padded =
  prop "a zero-padded batch reads as its images" ~count:100
    Gen.(pair (int_range 0 5) (gen_position ~h:3 ~w:4))
    (fun (k, p) ->
      let _, k = List.nth kernels k in
      let x = image ~h:3 ~w:4 ~c:2 6 in
      let padded = Nx.pad [| (0, 2); (0, 3); (0, 0) |] 0. x in
      same
        (V.Sample.at ~edge:`Zero k x (at p))
        (V.Sample.at ~edge:`Zero k padded (at p));
      let m = Nx.greater_s x 0. in
      let mp = Nx.pad [| (0, 2); (0, 3); (0, 0) |] false m in
      equal (array bool)
        (Nx.to_array (V.Sample.nearest ~edge:`Zero m (at p)))
        (Nx.to_array (V.Sample.nearest ~edge:`Zero mp (at p))))

(* Positions *)

let centres =
  test "centres" (fun () ->
      equal (array float_exact)
        [| 0.5; 0.5; 1.5; 0.5; 2.5; 0.5; 0.5; 1.5; 1.5; 1.5; 2.5; 1.5 |]
        (floats (V.Sample.centres Nx.float32 (2, 3)));
      equal (array int) [| 0; 3; 2 |]
        (Nx.shape (V.Sample.centres Nx.float32 (0, 3))))

let warp =
  test "a warp and its inverse" (fun () ->
      let module T = V.Transform in
      let x = image ~h:6 ~w:7 ~c:1 7 in
      let c = V.Sample.centres Nx.float64 (6, 7) in
      (* A translation by whole pixels and back reads every pixel again. *)
      let s = T.translation (Nx.create Nx.float64 [| 2 |] [| 2.; 1. |]) in
      let fwd = V.Sample.at ~edge:`Zero `Triangle x (T.apply (Nx.inv s) c) in
      let back = V.Sample.at ~edge:`Zero `Triangle fwd (T.apply s c) in
      let inside = Nx.slice [ Nx.R (0, 5); Nx.R (0, 5) ] in
      close 1e-15 (inside x) (inside back);
      (* The derivative in a rotation's angle, against a central difference. *)
      let f a =
        Nx.sum (V.Sample.at ~edge:`Zero `Triangle x (T.apply (T.rotation a) c))
      in
      let a = 0.3 and d = 1e-6 in
      let fd =
        (Nx.item [] (f (Nx.scalar Nx.float64 (a +. d)))
        -. Nx.item [] (f (Nx.scalar Nx.float64 (a -. d))))
        /. (2. *. d)
      in
      equal
        (float_rel ~rel:1e-5 ~abs:1e-6)
        fd
        (Nx.item [] (Rune.grad' f (Nx.scalar Nx.float64 a))))

(* Weighted means *)

(* A read is a weighted mean, so a flat image reads its value wherever a pixel
   weighs: under every edge but [`Zero], whose zeros outside weigh too; under
   [`Renormalise] inside the image, where the pixel holding the position
   weighs. *)
let flat =
  prop "a flat image reads its value" ~count:300
    Gen.(
      quad (int_range 1 5) (int_range 0 5) (float_range (-4.) 4.)
        (gen_position ~h:3 ~w:4))
    (fun (e, k, c, (u, v)) ->
      let name, edge = List.nth edges e and _, k = List.nth kernels k in
      let inside = u >= 0. && u < 4. && v >= 0. && v < 3. in
      assume (name <> "renormalise" || inside);
      cover "outside" (not inside);
      let x = Nx.full Nx.float64 [| 3; 4; 1 |] c in
      let y = Nx.item [ 0; 0 ] (V.Sample.at ~edge k x (at (u, v))) in
      equal (float_rel ~rel:1e-14 ~abs:0x1p-1000) c y)

(* [value k t] is [k] at [t] by mynd.mli's formulas, at float64. *)
let value (k : V.Sample.kernel) t =
  let sinc x =
    if x = 0. then 1. else Float.sin (Float.pi *. x) /. (Float.pi *. x)
  in
  let a = Float.abs t in
  match k with
  | `Box -> 1.
  | `Triangle -> 1. -. a
  | `Cubic c ->
      if a <= 1. then ((c +. 2.) *. a *. a *. a) -. ((c +. 3.) *. a *. a) +. 1.
      else
        (c *. a *. a *. a) -. (5. *. c *. a *. a) +. (8. *. c *. a) -. (4. *. c)
  | `Lanczos n -> sinc t *. sinc (t /. Float.of_int n)

(* In float32 a flat read is within rounding of its value. One row read at its
   centre has the row weights [1] and [0], exact, so the read is [Σ wⱼ c] over
   the [2h] columns: with [wⱼ = kⱼ / K], [K] the sum of the [kⱼ] in order, the
   sum rounds within [γ₂ₕ₋₁ = (2h - 1) u] of each term, each quotient and each
   product within [u], and [K] within [γ₂ₕ₋₁], for [u = 2⁻²⁴]. So [|y - c| <= 4h
   u κ |c|] to first order, [κ = Σ |kⱼ| / |Σ kⱼ|]. A read that did not divide by
   [K] would be off by [K - 1] too, up to 18 ulps of [1] under Keys' cubic
   (V1c's measurement), past the bound. *)
let flat32 =
  prop "a flat float32 image reads its value within rounding" ~count:300
    Gen.(
      quad (int_range 1 4) (int_range 0 5) (float_range 0.5 1.)
        (float_range (-6.) 10.))
    (fun (e, k, c, u) ->
      let _, edge = List.nth extensions e and _, k = List.nth kernels k in
      let h = half_width k in
      let r32 x = Int32.float_of_bits (Int32.bits_of_float x) in
      let c = r32 c and u = r32 u in
      let lo = Float.to_int (Float.floor (u -. 0.5 -. h)) + 1 in
      let ks =
        List.init
          (Float.to_int (2. *. h))
          (fun d -> value k (u -. 0.5 -. Float.of_int (lo + d)))
      in
      let kappa =
        List.fold_left (fun s v -> s +. Float.abs v) 0. ks
        /. Float.abs (List.fold_left ( +. ) 0. ks)
      in
      let x = Nx.full Nx.float32 [| 1; 4; 1 |] c in
      let p = Nx.create Nx.float32 [| 1; 2 |] [| u; 0.5 |] in
      let y = Nx.item [ 0; 0 ] (V.Sample.at ~edge k x p) in
      let bound = 4. *. h *. 0x1p-24 *. kappa *. Float.abs c *. 1.01 in
      at_most ~than:bound float_exact (Float.abs (y -. c)))

(* Under [`Renormalise] the inside weights sum to [0] where every inside tap
   lies at a kernel's zero: at [u = -½] under [`Triangle], and at [u = -3/2] too
   under [`Lanczos 3]. There the read is [0], and its derivative nearby is
   finite. *)
let renormalise_zero_sum =
  cases "where the inside weights sum to 0, the read is 0"
    ~id:(fun (n, _, _) -> n)
    [
      ("triangle at -1/2", `Triangle, -0.5);
      ("lanczos3 at -1/2", `Lanczos 3, -0.5);
      ("lanczos3 at -3/2", `Lanczos 3, -1.5);
    ]
    (fun (_, k, u) ->
      let x = Nx.add_s (image ~h:3 ~w:4 ~c:2 8) 3. in
      let read u = V.Sample.at ~edge:`Renormalise k x (at (u, 1.5)) in
      equal (array float_exact) [| 0.; 0. |] (floats (read u));
      List.iter
        (fun d ->
          let g =
            Rune.grad'
              (fun p -> Nx.sum (V.Sample.at ~edge:`Renormalise k x p))
              (at (u +. d, 1.5))
          in
          equal (array bool) [| true; true |]
            (Array.map Float.is_finite (Nx.to_array g)))
        [ -1e-3; -1e-9; 0.; 1e-9; 1e-3 ])

let refusals =
  group "misuse"
    [
      test "an image needs three axes" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Sample.at") (fun () ->
              V.Sample.at ~edge:`Zero `Box
                (Nx.zeros Nx.float32 [| 4; 4 |])
                (Nx.zeros Nx.float32 [| 2 |])));
      test "positions end in 2" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Sample.nearest") (fun () ->
              V.Sample.nearest ~edge:`Zero
                (Nx.zeros Nx.float32 [| 4; 4; 1 |])
                (Nx.zeros Nx.float32 [| 3 |])));
      test "batches broadcast" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Sample.at") (fun () ->
              V.Sample.at ~edge:`Zero `Box
                (Nx.zeros Nx.float32 [| 2; 4; 4; 1 |])
                (Nx.zeros Nx.float32 [| 3; 5; 2 |])));
      test "an empty image under a folding edge" (fun () ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Sample.at") (fun () ->
              V.Sample.at ~edge:`Replicate `Box
                (Nx.zeros Nx.float32 [| 0; 4; 1 |])
                (Nx.zeros Nx.float32 [| 2 |])));
      cases "an invalid kernel" ~id:fst
        [ ("keys of nan", `Cubic Float.nan); ("lanczos of 0", `Lanczos 0) ]
        (fun (_, k) ->
          raises_match (Exn.invalid_arg ~sub:"Mynd.Sample.at") (fun () ->
              V.Sample.at ~edge:`Zero k
                (Nx.zeros Nx.float32 [| 4; 4; 1 |])
                (Nx.zeros Nx.float32 [| 2 |])));
    ]

let () =
  exit
    (run "mynd.sample"
       [
         torch;
         torch_narrow;
         nearest_is_box;
         nearest_is_box_narrow;
         argument_rounds_once;
         edge_is_extend;
         interpolates;
         cell_edges;
         non_finite;
         far_positions;
         labels_keep_values;
         derivative;
         compiled;
         batches;
         padded;
         centres;
         warp;
         flat;
         flat32;
         renormalise_zero_sum;
         refusals;
       ])
