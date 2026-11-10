(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* {!Nx.layer_norm} and {!Nx.rms_norm} over rune: a compiled call is within
   the bound rune.mli's Numerics table states of a reference of twice
   float64's digits, at every float dtype, over strided views, with special
   values eager's; a lane of equal elements, [-0.] among them, is [+0.]
   compiled too; gradients are the definition's, against finite differences
   and in closed form, and compiled gradients the eager ones. *)

open Windtrap

let f64 = Nx.float64

(* Pairs [hi + lo] of twice float64's digits. *)
let two_sum a b =
  let s = a +. b in
  let v = s -. a in
  (s, (a -. (s -. v)) +. (b -. v))

let fast_two_sum a b =
  let s = a +. b in
  (s, b -. (s -. a))

let dd_add (ah, al) (bh, bl) =
  let s, e = two_sum ah bh in
  fast_two_sum s (e +. al +. bl)

let dd_mul (ah, al) (bh, bl) =
  let p = ah *. bh in
  fast_two_sum p (Float.fma ah bh (-.p) +. (ah *. bl) +. (al *. bh))

let dd_mul_float (hi, lo) b =
  let p = hi *. b in
  if not (Float.is_finite p) then (p, 0.)
  else fast_two_sum p (Float.fma hi b (-.p) +. (lo *. b))

let dd_div (ah, al) (bh, bl) =
  let q = ah /. bh in
  fast_two_sum q ((Float.fma (-.q) bh ah +. al -. (q *. bl)) /. bh)

let dd_sqrt (ah, al) =
  if ah = 0. then (0., 0.)
  else
    let r = Float.sqrt ah in
    fast_two_sum r ((Float.fma (-.r) r ah +. al) /. (2. *. r))

(* A lane's mean and variance, in float64. *)
let stats xs =
  let n = Float.of_int (Array.length xs) in
  let mean f = Array.fold_left (fun a x -> a +. f x) 0. xs /. n in
  let mu = mean Fun.id in
  (mu, mean (fun x -> (x -. mu) *. (x -. mu)))

(* [xs] times the power of two that brings its greatest magnitude into
   [[1, 2)]: at [eps = 0], a lane's normalisation is that of the scaled
   lane, whose squares do not underflow. *)
let unit_scale xs =
  let m = Array.fold_left (fun a x -> Float.max a (Float.abs x)) 0. xs in
  if m = 0. || not (Float.is_finite m) then xs
  else
    let _, e = Float.frexp m in
    Array.map (fun x -> Float.ldexp x (1 - e)) xs

(* A lane by its definition, [(x - μ) / sqrt (V + eps)], with the exact
   mean, variance, mean absolute deviation and mean magnitude. *)
let lane_ln eps xs =
  let xs = if eps = 0. then unit_scale xs else xs in
  let n = (Float.of_int (Array.length xs), 0.) in
  let sum = Array.fold_left (fun a x -> dd_add a (x, 0.)) (0., 0.) xs in
  let mu = dd_div sum n in
  let ds = Array.map (fun x -> dd_add (x, 0.) (-.fst mu, -.snd mu)) xs in
  let sq = Array.fold_left (fun a d -> dd_add a (dd_mul d d)) (0., 0.) ds in
  let v = dd_div sq n in
  let root = dd_sqrt (dd_add v (eps, 0.)) in
  let y = Array.map (fun d -> dd_div d root) ds in
  let mean f = Array.fold_left (fun a x -> a +. f x) 0. xs /. fst n in
  (y, fst v, mean (fun x -> Float.abs (x -. fst mu)), mean Float.abs)

(* A lane by [rms_norm]'s definition, [x / sqrt (Σ x² / n + eps)]. *)
let lane_rms eps xs =
  let xs = if eps = 0. then unit_scale xs else xs in
  let n = (Float.of_int (Array.length xs), 0.) in
  let square a x = dd_add a (dd_mul (x, 0.) (x, 0.)) in
  let sq = Array.fold_left square (0., 0.) xs in
  let root = dd_sqrt (dd_add (dd_div sq n) (eps, 0.)) in
  Array.map (fun x -> dd_div (x, 0.) root) xs

(* rune.mli's row for [layer_norm], an element of value [y] of a lane of
   [n] terms: [(F / 2s + 11u/2) |y| + n u A / √s], [F] [var]'s row. *)
let ln_bound u n eps (v, dev, mag) y =
  let k = Float.of_int n in
  let b = (k +. 1.) *. u *. mag in
  let f =
    ((k +. 4.) *. u *. v)
    +. (2. *. k *. u *. b *. Float.sqrt v)
    +. (3. *. (k +. 2.) *. u *. b *. b)
  in
  let s = v +. eps in
  (((f /. (2. *. s)) +. (5.5 *. u)) *. Float.abs y)
  +. (k *. u *. dev /. Float.sqrt s)

type dtype =
  | Dt : (float, 's) Nx.dtype * float * float * float * float -> dtype

(* Each float dtype with its computing format's unit roundoff, its own
   ([0] where it is that format), its least subnormal, and the scale of a
   small lane, whose elements are then subnormal. *)
let dtypes =
  [
    Dt (Nx.float64, 0x1p-53, 0., 0x1p-1074, 0x1p-1060);
    Dt (Nx.float32, 0x1p-24, 0., 0x1p-149, 0x1p-140);
    Dt (Nx.float16, 0x1p-24, 0x1p-11, 0x1p-24, 0x1p-20);
    Dt (Nx.bfloat16, 0x1p-24, 0x1p-8, 0x1p-133, 0x1p-128);
  ]

type lane = Drawn | Constant | Zeros | Nan | Inf | Small

let lane_kind =
  Gen.frequency
    [
      (6, Gen.constant Drawn);
      (2, Gen.constant Constant);
      (1, Gen.constant Zeros);
      (1, Gen.constant Nan);
      (1, Gen.constant Inf);
      (2, Gen.constant Small);
    ]

(* Lanes about a mean far from their spread, constant, of zeros of both
   signs, or holding a NaN or an infinity; read through a strided view
   half the time. *)
let case =
  let open Gen in
  let* dt =
    of_list ~pp:(fun ppf (Dt (d, _, _, _, _)) -> Nx.pp_dtype ppf d) dtypes
  in
  let* r = int_range 1 3 in
  let* k = of_list ~pp:Format.pp_print_int [ 1; 2; 16; 17; 768; 1025 ] in
  let* kinds = array ~size:(constant r) lane_kind in
  let* mu = of_list [ 0.; 64.; -1000.; 1e4 ] in
  let* sigma = of_list [ 1.; 1e-3; 100. ] in
  let* zs = array ~size:(constant (r * k)) (float_range (-3.) 3.) in
  let* signs = array ~size:(constant (r * k)) bool in
  let* eps = of_list [ 1e-5; 0.5; 0. ] in
  (* Small lanes hold subnormals, which [eps = 0.] normalises. *)
  let eps = if Array.mem Small kinds then 0. else eps in
  let* affine = bool in
  let* strided = bool in
  let+ rms = bool in
  (* Deviations of [2^-10] spreads at least, far from underflow. *)
  let grid z = Float.round (z *. 1024.) /. 1024. in
  let xs =
    Array.mapi
      (fun i z ->
        match kinds.(i / k) with
        | Drawn -> mu +. (sigma *. grid z)
        | Constant -> mu +. sigma
        | Zeros -> if signs.(i) then 0. else -0.
        | Nan -> if i mod k = 0 then Float.nan else mu
        | Inf -> if i mod k = 0 then Float.infinity else mu
        | Small -> grid z)
      zs
  in
  (dt, [| r; k |], kinds, xs, eps, affine, strided, rms)

let gains k = Array.init k (fun j -> [| 1.5; -0.5; 2.; 1. |].(j mod 4))
let offsets k = Array.init k (fun j -> [| 0.25; -1.; 0.; 3. |].(j mod 4))

let same a b =
  (Float.is_nan a && Float.is_nan b)
  || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

(* Compiled, within the bound of rune.mli's row of the definition, the
   affine's [|g|] times it plus [u |g y|] and [u |g y + b|], then a narrow
   dtype's rounding; a lane of no spread the affine of [+0.]; a lane
   holding a NaN or an infinity eager's. *)
let law_jit
    (Dt (dt, u, ud, tiny, small), s, kinds, xs, eps, affine, strided, rms) =
  let r = s.(0) and k = s.(1) in
  cover "a narrow dtype" (ud > 0.);
  cover "an affine" affine;
  cover "a strided view" strided;
  cover "rms_norm" rms;
  Array.iter
    (fun kind ->
      cover "a constant lane" (kind = Constant);
      cover "a lane of zeros" (kind = Zeros);
      cover "a lane holding a NaN" (kind = Nan);
      cover "a lane holding an infinity" (kind = Inf);
      cover "a small lane" (kind = Small))
    kinds;
  let at i x = if kinds.(i / k) = Small then x *. small else x in
  let xs = Array.mapi at xs in
  let x = Nx.cast dt (Nx.create f64 s xs) in
  let x =
    if not strided then x
    else
      let wide = Nx.concatenate ~axis:1 [ x; x ] in
      let wide = Nx.reshape [| r; 2; k |] wide in
      Nx.reshape [| r; k |] (Nx.slice [ A; I 0; A ] wide)
  in
  let xs = Nx.to_array (Nx.cast f64 x) in
  let vec v = Nx.cast dt (Nx.create f64 [| k |] v) in
  let g = gains k and b = offsets k in
  let f x =
    match (rms, affine) with
    | true, true -> Nx.rms_norm ~eps ~scale:(vec g) x
    | true, false -> Nx.rms_norm ~eps x
    | false, true -> Nx.layer_norm ~eps ~scale:(vec g) ~shift:(vec b) x
    | false, false -> Nx.layer_norm ~eps x
  in
  let read y = Nx.to_array (Nx.cast f64 y) in
  let eager = read (f x) and compiled = read (Rune.jit' f x) in
  for l = 0 to r - 1 do
    let lx = Array.sub xs (l * k) k in
    let special = Array.exists (fun x -> not (Float.is_finite x)) lx in
    let want =
      if special || rms then [||]
      else
        let y, v, dev, mag = lane_ln eps lx in
        Array.mapi
          (fun j y ->
            let gj = if affine then g.(j) else 1. in
            let bj = if affine then b.(j) else 0. in
            (y, v, dev, mag, gj, bj))
          y
    in
    let rms_want = if special || not rms then [||] else lane_rms eps lx in
    for j = 0 to k - 1 do
      let i = (l * k) + j in
      let c = compiled.(i) in
      if special then begin
        if not (same c eager.(i)) then
          failf "at %d of a special lane: compiled %h, eager %h" i c
            eager.(i)
      end
      else if rms then begin
        let gj = if affine then g.(j) else 1. in
        let kk = Float.of_int (k + if affine then 10 else 8) /. 2. in
        let rel = kk *. u /. (1. -. (4. *. kk *. u)) in
        let zh, zl = dd_mul_float rms_want.(j) gj in
        let sub = (Float.abs gj +. 1.) *. tiny in
        let tol = (rel *. Float.abs zh) +. (ud *. Float.abs c) +. sub in
        if Float.abs (c -. zh -. zl) > tol then
          failf "rms_norm at %d: compiled %h, by definition %h" i c zh
      end
      else
        let y, v, dev, mag, gj, bj = want.(j) in
        if v = 0. then begin
          let z = if affine then (0. *. gj) +. bj else 0. in
          if not (same c z) then
            failf "at %d of a lane of no spread: compiled %h, not %h" i c z
        end
        else
          let zh, zl = dd_add (dd_mul_float y gj) (bj, 0.) in
          let tol =
            (Float.abs gj *. ln_bound u k eps (v, dev, mag) (fst y))
            +. (u *. (Float.abs (gj *. fst y) +. Float.abs zh))
            +. (ud *. Float.abs c)
            +. ((Float.abs gj +. 1.) *. tiny)
          in
          if Float.abs (c -. zh -. zl) > tol then
            failf "layer_norm at %d: compiled %h, by definition %h" i c zh
    done
  done

(* A lane of equal elements, [-0.] among them, is [+0.] compiled, as
   eagerly, at [eps = 0.] too. *)
let test_constant () =
  List.iter
    (fun (v, n) ->
      List.iter
        (fun eps ->
          let x = Nx.full Nx.float32 [| 2; n |] v in
          let y = Nx.to_array (Rune.jit' (fun x -> Nx.layer_norm ~eps x) x) in
          Array.iteri
            (fun i y ->
              if Int64.bits_of_float y <> 0L then
                failf "%d of %h, eps %g: %h at %d" n v eps y i)
            y)
        [ 1e-5; 0. ])
    [
      (1000.1, 1024); (1000.1, 768); (0.1, 4096); (-3.3, 5000); (-0., 3);
      (0., 1);
    ]

(* Compiled, a lane whose squares overflow, or underflow at [eps = 0.], is
   normalised at a power of two, beside one that keeps its scale. *)
let test_overflow () =
  let x = Nx.create Nx.float32 [| 2; 2 |] [| 1e20; -1e20; 3.; 5. |] in
  let ln = Nx.to_array (Rune.jit' (fun x -> Nx.layer_norm x) x) in
  let rms = Nx.to_array (Rune.jit' (fun x -> Nx.rms_norm x) x) in
  let y = 1. /. Float.sqrt (1. +. 1e-5) in
  equal ~msg:"layer_norm" (array (float 1e-6)) [| 1.; -1.; -.y; y |] ln;
  let tiny = Nx.create Nx.float32 [| 2 |] [| 0x1p-140; -0x1p-140 |] in
  equal ~msg:"layer_norm of subnormals" (array float_exact) [| 1.; -1. |]
    (Nx.to_array (Rune.jit' (fun x -> Nx.layer_norm ~eps:0. x) tiny));
  equal ~msg:"rms_norm of subnormals" (array float_exact) [| 1.; -1. |]
    (Nx.to_array (Rune.jit' (fun x -> Nx.rms_norm ~eps:0. x) tiny));
  let r = Float.sqrt 17. in
  equal ~msg:"rms_norm" (array (float 1e-6))
    [| 1.; -1.; 3. /. r; 5. /. r |]
    rms

let x0 =
  Nx.create f64 [| 3; 5 |]
    (Array.init 15 (fun i -> 100. +. Float.sin (Float.of_int i *. 1.7)))

let w0 =
  Nx.create f64 [| 3; 5 |] (Array.init 15 (fun i -> Float.cos (Float.of_int i)))
let g0 = Nx.create f64 [| 5 |] [| 1.5; -0.5; 2.; 1.; 0.25 |]
let b0 = Nx.create f64 [| 5 |] [| 0.25; -1.; 0.; 3.; 1. |]

let ln_loss ?axes x g b =
  Nx.sum (Nx.mul w0 (Nx.layer_norm ?axes ~scale:g ~shift:b x))
let rms_loss x g = Nx.sum (Nx.mul w0 (Nx.rms_norm ~scale:g x))

let ok = result unit string
let t = Nx.Ptree.tensor

let test_check_grads () =
  let check msg f p = equal ~msg ok (Ok ()) (Rune.check_grads t f p) in
  check "layer_norm x" (fun x -> ln_loss x g0 b0) x0;
  check "layer_norm along 0" (fun x -> ln_loss ~axes:[ 0 ] x g0 b0) x0;
  check "layer_norm scale" (fun g -> ln_loss x0 g b0) g0;
  check "layer_norm shift" (fun b -> ln_loss x0 g0 b) b0;
  check "rms_norm x" (fun x -> rms_loss x g0) x0;
  check "rms_norm scale" (fun g -> rms_loss x0 g) g0

(* With [h = scale w] and [σ = sqrt (V + eps)], the gradient of [Σ w y] at
   [x_i] is [(h_i - mean h) / σ - (x_i - μ) mean (h (x - μ)) / σ³]. *)
let test_closed_form () =
  let eps = 1e-5 in
  let xs = Nx.to_array x0 and ws = Nx.to_array w0 and gs = Nx.to_array g0 in
  let got = Nx.to_array (Rune.grad' (fun x -> ln_loss x g0 b0) x0) in
  for l = 0 to 2 do
    let lane = Array.sub xs (l * 5) 5 in
    let mu, v = stats lane in
    let sd = Float.sqrt (v +. eps) in
    let h = Array.init 5 (fun j -> gs.(j) *. ws.((l * 5) + j)) in
    let mean f = (f 0 +. f 1 +. f 2 +. f 3 +. f 4) /. 5. in
    let mh = mean (fun j -> h.(j)) in
    let mhd = mean (fun j -> h.(j) *. (lane.(j) -. mu)) in
    for j = 0 to 4 do
      let i = (l * 5) + j in
      let want =
        ((h.(j) -. mh) /. sd) -. ((lane.(j) -. mu) *. mhd /. (sd *. sd *. sd))
      in
      equal ~msg:(Printf.sprintf "at %d" i) (float 1e-9) want got.(i)
    done
  done

(* Compiled, the gradients are the eager ones. *)
let test_jit_grads () =
  List.iter
    (fun (name, loss) ->
      equal ~msg:name (array (float 1e-12))
        (Nx.to_array (Rune.grad' loss x0))
        (Nx.to_array (Rune.grad' (Rune.jit' loss) x0)))
    [
      ("layer_norm", fun x -> ln_loss x g0 b0);
      ("layer_norm along 0", fun x -> ln_loss ~axes:[ 0 ] x g0 b0);
      ("rms_norm", fun x -> rms_loss x g0);
    ]

let () =
  exit
    (run "rune norms"
       [
         prop "compiled, within rune.mli's bound of the definition" case
           law_jit;
         test "a lane of equal elements is zero compiled" test_constant;
         test "a lane whose squares overflow, compiled" test_overflow;
         test "gradients against finite differences" test_check_grads;
         test "layer_norm's gradient in closed form" test_closed_form;
         test "compiled gradients are the eager ones" test_jit_grads;
       ])
