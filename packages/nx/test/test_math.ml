(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The float functions nx composes from the kernels' kinds, and the rules of
   their section: each within its stated ulps of the correctly rounded value
   (golden/special/, written by gen/special.py from mpmath) at float32 and
   float64, over drawn float32 values against the C library's float64 result,
   which lies far inside a float32 ulp, at the special values nx.mli states, and
   a narrower float computed at float32 and rounded once. roundeven is exact:
   the float64 arithmetic's ties to even. An ulp is counted as nx.mli counts it,
   in floats from the exact value rounded to the dtype (Nx_support.ulps). *)

open Windtrap
module D = Nx_array.Dtype

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f
let float_pp ppf v = Format.fprintf ppf "%h" v
let values f dt xs = Nx.to_array (f (Nx.create dt [| Array.length xs |] xs))

let same a b =
  (Float.is_nan a && Float.is_nan b)
  || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let pp_floats ppf xs =
  Format.pp_print_list
    ~pp_sep:(fun ppf () -> Format.pp_print_string ppf " ")
    float_pp ppf (Array.to_list xs)

(* [v] holds [r], the exact value, to [k] ulps at [dt], float32 or float64:
   where [r] rounds to an infinity or to zero, [v] is that infinity or zero, of
   [r]'s sign, as nx.mli states; a NaN for a NaN. *)
let held (type s) (dt : (float, s) D.t) k r v =
  let rounded =
    match dt with
    | D.Float32 -> Int32.float_of_bits (Int32.bits_of_float r)
    | _ -> r
  in
  if Float.is_nan rounded || rounded = 0. || Float.abs rounded = Float.infinity
  then same rounded v
  else Nx_support.ulps dt r v <= k

(* A float dtype and the bound that holds there, in ulps. *)
type at = At : (float, 's) D.t * float -> at

let at_id (At (dt, k)) = Printf.sprintf "%s, %g ulps" (D.name dt) k
let f32_f64 = [ D.Any D.Float32; D.Any D.Float64 ]
let dtype_id (D.Any dt) = D.name dt

(* Goldens *)

(* [check_golden ?keep name dt k f] holds [f], over the golden's rows at [dt]
   whose arguments [keep] keeps, by column, as [held] holds a value. It fails
   listing the worst rows that miss. *)
let check_golden (type s) ?(keep = fun _ -> true) name (dt : (float, s) D.t) k
    (f : (float, s, Nx.host) Nx.t array -> (float, s, Nx.host) Nx.t) =
  let rows =
    List.filter
      (fun (r : Nx_support.row) -> keep r.args)
      (Nx_support.golden name dt)
  in
  let n = List.length rows in
  let arity = Array.length (List.hd rows).args in
  let column i =
    Nx.create dt [| n |]
      (Array.of_list (List.map (fun r -> r.Nx_support.args.(i)) rows))
  in
  let got = Nx.to_array (f (Array.init arity column)) in
  let miss i (r : Nx_support.row) =
    let v = got.(i) in
    if held dt k r.value v then None
    else Some (Nx_support.ulps dt r.value v, r, v)
  in
  match List.filter_map Fun.id (List.mapi miss rows) with
  | [] -> ()
  | misses ->
      let misses = List.sort (fun (a, _, _) (b, _, _) -> compare b a) misses in
      let pp ppf (d, (r : Nx_support.row), v) =
        Format.fprintf ppf "at (%a): %h, not %h (%g ulps)" pp_floats r.args v
          r.value d
      in
      failf "%s: %d of %d rows past %g ulps:@\n%a" name (List.length misses) n k
        (Format.pp_print_list pp)
        (List.filteri (fun i _ -> i < 8) misses)

let golden1 ?keep name
    (f : 's. (float, 's, Nx.host) Nx.t -> (float, 's, Nx.host) Nx.t)
    (At (dt, k)) =
  check_golden ?keep name dt k (fun a -> f a.(0))

let golden2 name
    (f :
      's.
      (float, 's, Nx.host) Nx.t ->
      (float, 's, Nx.host) Nx.t ->
      (float, 's, Nx.host) Nx.t) (At (dt, k)) =
  check_golden name dt k (fun a -> f a.(0) a.(1))

(* Drawn float32 values *)

(* The float32 [k] floats from [x], toward [+inf] from a positive [x]. *)
let step32 x k = Int32.float_of_bits (Int32.add (Int32.bits_of_float x) k)

(* [n] float32 values from bits: any, or within 8 floats of one of [near],
   either sign. *)
let float32s near =
  let open Gen in
  let close =
    let+ c = of_list near and+ d = int_range (-8) 8 and+ neg = bool in
    let v = step32 c (Int32.of_int d) in
    if neg then -.v else v
  in
  let one = frequency [ (1, map Int32.float_of_bits int32); (1, close) ] in
  with_pp pp_floats (array ~size:(constant 64) one)

(* Each drawn [xs] has a value within 8 floats of every point of [near]. *)
let covers_near near xs =
  List.iter
    (fun c ->
      cover
        (Printf.sprintf "near %h" c)
        (Array.exists
           (fun x ->
             Float.is_finite x
             && Nx_support.ulps D.Float32 c (Float.abs x) <= 8.)
           xs))
    near

(* [f] at float32 within [k] ulps of [reference], the C library's float64
   function, at each of [xs]. *)
let law_float32 name f reference k xs =
  let got = values f Nx.float32 xs in
  Array.iteri
    (fun i x ->
      let r = reference x and v = got.(i) in
      if not (held D.Float32 k r v) then
        failf "%s %h is %h, the exact %h rounded to float32 %g ulps away" name x
          v r
          (Nx_support.ulps D.Float32 r v))
    xs

(* Narrow floats *)

let narrow_dtypes =
  List.filter (fun (D.Any dt) -> D.is D.Float dt && D.bits dt < 32) D.all

(* Every code of [dt], a float narrower than float32. *)
let codes (type s) (dt : (float, s) D.t) : (float, s, Nx.host) Nx.t =
  let every =
    match D.bits dt with
    | 4 -> Nx.bitcast dt (Nx.arange Nx.uint4 0 16 1)
    | 8 -> Nx.bitcast dt (Nx.arange Nx.uint8 0 256 1)
    | _ -> Nx.bitcast dt (Nx.arange Nx.uint16 0 65536 1)
  in
  Nx.place Nx.Host.on every

(* At a float narrower than float32, [f x] is [f] at float32 rounded once
   (nx.mli's Arithmetic section), at every code. *)
let law_narrow (f : 's. (float, 's, Nx.host) Nx.t -> (float, 's, Nx.host) Nx.t)
    (D.Any dt) =
  match D.kind dt with
  | D.Float ->
      let x = codes dt in
      let expected = Nx.to_array (Nx.cast dt (f (Nx.cast Nx.float32 x))) in
      let got = Nx.to_array (f x) and xs = Nx.to_array x in
      Array.iteri
        (fun i e ->
          if not (same e got.(i)) then
            failf "at %h: %h, not %h" xs.(i) got.(i) e)
        expected
  | _ -> ()

let narrow (f : 's. (float, 's, Nx.host) Nx.t -> (float, 's, Nx.host) Nx.t) =
  cases "at a narrow float is float32's, rounded once" ~id:dtype_id
    narrow_dtypes (law_narrow f)

(* The section's rules *)

(* The special values nx.mli lists from C99 Annex F, F.10, at float32 and
   float64: exact, a zero with its sign. *)
let test_annex_f (D.Any dt) =
  match D.kind dt with
  | D.Float ->
      let one f x = (values f dt [| x |]).(0) in
      let pow a b = Nx.item [] (Nx.pow (Nx.scalar dt a) (Nx.scalar dt b)) in
      let rows =
        [
          ("exp (-inf)", one Nx.exp neg_infinity, 0.);
          ("log 0.", one Nx.log 0., neg_infinity);
          ("log (-0.)", one Nx.log (-0.), neg_infinity);
          ("sin inf", one Nx.sin infinity, nan);
          ("atan inf", one Nx.atan infinity, D.of_float dt (Float.pi /. 2.));
          ("exp 710.", one Nx.exp 710., infinity);
          ("pow (-2.) (-1101.)", pow (-2.) (-1101.), -0.);
          ("sin (-0.)", one Nx.sin (-0.), -0.);
          ("pow nan 0.", pow nan 0., 1.);
          ("pow 1. nan", pow 1. nan, 1.);
        ]
      in
      List.iter (fun (msg, got, want) -> equal ~msg float_exact want got) rows
  | _ -> ()

(* A format without infinities or NaN stores an infinite or NaN result as a cast
   does: float8_e4m3fn saturates, and float4_e2m1fn holds NaN as +0. *)
let no_infinities () =
  let e4m3 = Nx.float8_e4m3fn and e2m1 = Nx.float4_e2m1fn in
  equal (array float_exact) [| -448. |] (values Nx.log e4m3 [| 0. |]);
  equal (array float_exact) [| 448. |] (values Nx.atanh e4m3 [| 1. |]);
  equal (array float_exact) [| 448. |] (values Nx.exp e4m3 [| 100. |]);
  equal (array float_exact) [| -6.; 0. |] (values Nx.log e2m1 [| 0.; -1. |])

(* rsqrt *)

(* rsqrt's exact value at [x], [e = 1/√x], from its result [v = e (1 + d)]. Then
   [v² x - 1 = 2d + d²], which a square split exactly by fma and one fma give
   far below an ulp. [x] and [v] are first scaled by [2^-2k] and [2^k], which
   keeps [v² x] and puts [x] in [\[1/2, 2)], where [v²] cannot overflow. The
   result, [v / (1 + d)] rounded once, is the exact value rounded but within
   about [2^-100] of a midpoint. *)
let rsqrt_exact x v =
  let _, ex = Float.frexp x in
  let k = if ex >= 0 then ex / 2 else (ex - 1) / 2 in
  let x = Float.ldexp x (-2 * k) and v = Float.ldexp v k in
  let p = v *. v in
  let pe = Float.fma v v (-.p) in
  let t = Float.fma p x (-1.) +. (pe *. x) in
  let d = t /. (1. +. Float.sqrt (1. +. t)) in
  Float.ldexp (v /. (1. +. d)) (-k)

let law_rsqrt (type s) (dt : (float, s) D.t) (x : (float, s, Nx.host) Nx.t) =
  let xs = Nx.to_array x and vs = Nx.to_array (Nx.rsqrt x) in
  Array.iteri
    (fun i x ->
      let v = vs.(i) in
      if x > 0. && Float.is_finite x then begin
        let d = Nx_support.ulps dt (rsqrt_exact x v) v in
        if d > 2. then
          failf "at %a: %a, %g ulps from 1/√x" float_pp x float_pp v d
      end
      else if not (same (1. /. Float.sqrt x) v) then
        failf "at %a: %a, not %a" float_pp x float_pp v float_pp
          (1. /. Float.sqrt x))
    xs

(* A host value of [dt] and shape [[n]] over drawn bytes. *)
let drawn (type s) (dt : (float, s) D.t) n : (float, s, Nx.host) Nx.t Gen.t =
  let open Gen in
  let+ data = string_of ~size:(constant (D.bytes dt n)) char in
  Nx.Repr.of_array Nx.Host.v
    (Nx_array.v dt
       (Nx_array.Layout.contiguous [| n |])
       (Rig.Buffer.of_string data))

let samples = 64

let rsqrt_ends (type s) (dt : (float, s) D.t) =
  let least = match dt with D.Float32 -> 0x1p-149 | _ -> 0x1p-1074 in
  let most =
    match dt with D.Float32 -> 0x1.fffffep127 | _ -> Float.max_float
  in
  law_rsqrt dt
    (Nx.create dt [| 7 |] [| least; 3. *. least; 0x1p-126; 0.5; 2.; 3.; most |])

(* Annex F's values of rsqrt, asinh, acosh and atanh, at float64 and float32. *)
let specials (D.Any dt) =
  match D.kind dt with
  | D.Float ->
      equal ~msg:"rsqrt" (array float_exact)
        [| infinity; neg_infinity; nan |]
        (values Nx.rsqrt dt [| 0.; -0.; -1. |]);
      equal ~msg:"asinh" (array float_exact)
        [| -0.; 0.; infinity; neg_infinity; nan |]
        (values Nx.asinh dt [| -0.; 0.; infinity; neg_infinity; nan |]);
      equal ~msg:"acosh" (array float_exact)
        [| 0.; nan; nan; infinity; nan |]
        (values Nx.acosh dt [| 1.; 0.5; neg_infinity; infinity; nan |]);
      equal ~msg:"atanh" (array float_exact)
        [| infinity; neg_infinity; nan; nan; -0.; 0.; nan; nan; nan |]
        (values Nx.atanh dt
           [| 1.; -1.; 1.5; -2.; -0.; 0.; infinity; neg_infinity; nan |])
  | _ -> ()

(* hypot *)

let hypot_specials (D.Any dt) =
  match D.kind dt with
  | D.Float ->
      let v a b = Nx.item [] (Nx.hypot (Nx.scalar dt a) (Nx.scalar dt b)) in
      equal float_exact infinity (v nan infinity);
      equal float_exact infinity (v neg_infinity nan);
      equal float_exact 0. (v 0. (-0.));
      equal float_exact 0. (v (-0.) (-0.));
      equal float_exact 3. (v (-3.) (-0.));
      equal float_exact infinity (v 2. neg_infinity);
      equal float_exact nan (v nan 1.);
      equal float_exact 5. (v 3. 4.);
      (* Each square overflows float32. *)
      equal float_exact 0x5p100 (v 0x3p100 0x4p100)
  | _ -> ()

let law_hypot32 (x, y) =
  let xs = Nx.to_array x and ys = Nx.to_array y in
  let got = Nx.to_array (Nx.hypot x y) in
  Array.iteri
    (fun i x ->
      let r = Float.hypot x ys.(i) and v = got.(i) in
      if not (held D.Float32 3. r v) then
        failf "hypot %h %h is %h, the exact %h rounded %g ulps away" x ys.(i) v
          r
          (Nx_support.ulps D.Float32 r v))
    xs

(* log *)

(* [e^-3] rounded to float32: its exact log lies 0.13 ulp below [-3.]. *)
let e_minus_3 = 0x3d4bed86l

(* Bit patterns of positive finite float32: subnormals, values near [1.], the
   neighbours of [e_minus_3], values from [2^28], where asinh changes form, and
   any. *)
let positive_bits =
  Gen.(
    frequency
      [
        (1, int32_range 1l 0x007f_ffffl);
        (1, int32_range 0x3f7f_0000l 0x3f81_0000l);
        (1, int32_range (Int32.sub e_minus_3 4l) (Int32.add e_minus_3 4l));
        (1, int32_range 0x4d80_0000l 0x7f7f_ffffl);
        (2, int32_range 1l 0x7f7f_ffffl);
      ])

let float32_values bits =
  let open Gen in
  let+ bits = array ~size:(constant 64) bits in
  Array.map Int32.float_of_bits bits

let covers xs =
  let has p = Array.exists p xs in
  cover "a subnormal" (has (fun x -> Float.abs x < 0x1p-126));
  cover "near 1" (has (fun x -> Float.abs (Float.abs x -. 1.) < 0x1p-7));
  cover "e^-3 rounded" (has (fun x -> x = Int32.float_of_bits e_minus_3));
  cover "from 2^28" (has (fun x -> Float.abs x >= 0x1p28))

(* [v] rounded to the nearest float32. *)
let round32 v = Int32.float_of_bits (Int32.bits_of_float v)

(* The float32 next above [x], a float32. *)
let succ32 x =
  if x = 0. then 0x1p-149 else if x > 0. then step32 x 1l else step32 x (-1l)

let pred32 x = -.succ32 (-.x)

(* The greatest float32 at most [v], and the least at least [v]. *)
let below v =
  let f = round32 v in
  if f > v then pred32 f else f

let above v =
  let f = round32 v in
  if f < v then succ32 f else f

(* The C library's float64 [log] and [asinh] are taken within [2^-48] of the
   exact value, relatively: far below a float32 ulp, [2^-23] relatively. *)
let reference f x =
  let y = f x in
  (y, Float.abs y *. 0x1p-48)

(* Each result, of [x] and of [x] reversed, a view, holds [ok], given the
   reference [y] within [e] of the exact value. *)
let results_within ~name f libm ~ok xs =
  let x = Nx.create Nx.float32 [| Array.length xs |] xs in
  let got = Nx.to_array (f x) in
  let reversed = Nx.to_array (f (Nx.flip x)) in
  let n = Array.length xs in
  Array.iteri
    (fun i x ->
      let y, e = reference libm x in
      let check r =
        if not (ok y e r) then
          failf "%s %h is %h, the C library's float64 %h" name x r y
      in
      check got.(i);
      check reversed.(n - 1 - i))
    xs

(* The two floats around the exact value, or that value where it is a float:
   only [log 1.], whose reference is exactly [0.]. *)
let law_log xs =
  covers xs;
  (* [log 1.] is [+0.], Annex F's. *)
  let ok y e r =
    if y = 0. then same 0. r else below (y -. e) <= r && r <= above (y +. e)
  in
  results_within ~name:"log" Nx.log Stdlib.log ~ok xs

(* [1.], [e_minus_3], the least subnormal and the greatest float. *)
let edge_examples =
  [
    Array.map Int32.float_of_bits
      [| 0x3f80_0000l; e_minus_3; 1l; 0x7f7f_ffffl |];
  ]

(* log10 *)

(* [10^n] at each [n] whose power the dtype holds exactly, and Annex F's values:
   [-inf] at either zero, NaN below it and at NaN, [inf] at [inf]. *)
let log10_values () =
  let at (type s) (dt : (float, s) D.t) most =
    let n = most + 1 in
    let powers =
      Array.init n (fun k -> float_of_string (Printf.sprintf "1e%d" k))
    in
    equal ~msg:(D.name dt) (array float_exact)
      (Array.init n Float.of_int)
      (values Nx.log10 dt powers)
  in
  at Nx.float64 22;
  at Nx.float32 10;
  at Nx.float16 4;
  at Nx.bfloat16 3;
  at Nx.float8_e4m3fn 1;
  at Nx.float8_e5m2 1;
  at Nx.float4_e2m1fn 0;
  List.iter
    (fun (D.Any dt) ->
      match D.kind dt with
      | D.Float ->
          equal ~msg:(D.name dt) (array float_exact)
            [| neg_infinity; neg_infinity; nan; nan; nan; infinity |]
            (values Nx.log10 dt [| 0.; -0.; -1.; neg_infinity; nan; infinity |])
      | _ -> ())
    f32_f64

(* At float32, log10 of every float16 and bfloat16, each a float32, within 1 ulp
   of the C library's float64 result rounded: with the narrow law, a narrow
   float is within one ulp of its exact value rounded. *)
let log10_wide_codes () =
  List.iter
    (fun (D.Any dt) ->
      match D.kind dt with
      | D.Float when D.bits dt = 16 ->
          let xs = Nx.to_array (Nx.cast Nx.float32 (codes dt)) in
          law_float32 "log10" Nx.log10 Stdlib.log10 1. xs
      | _ -> ())
    narrow_dtypes

(* sinc *)

(* [π] less its float64 head. *)
let pi_tail = 0x1.1a62633145c07p-53

(* [sin (πx) / (πx)] within about an ulp of float64, far inside float32's. [sin
   (πx)] is [±sin (πr)] for [r = x - k], [k] the nearest integer, exact; [πr]
   and [πx] are each a head and a tail by fma, and the quotient is corrected by
   its residual. *)
let sinc_ref x =
  if x = 0. then 1.
  else if not (Float.is_finite x) then Float.nan
  else if Float.is_integer x then 0.
  else
    let k = Float.round x in
    let times a =
      let h = Float.pi *. a in
      (h, Float.fma Float.pi a (-.h) +. (pi_tail *. a))
    in
    let sh, sl = times (x -. k) and dh, dl = times x in
    let s = Float.sin sh +. (sl *. Float.cos sh) in
    let q = s /. dh in
    let q = q +. ((Float.fma (-.q) dh s -. (q *. dl)) /. dh) in
    if Float.rem k 2. = 0. then q else -.q

(* nx.mli's 10 ulps, derived from sinc's two arms in ulps of the exact value
   rounded. Away from 0 it is [±sin (π r) / (π x)], [r = x - k]: the roundings
   of [π], of [π r], of [π x] and of the quotient are each a relative [u] (that
   of [π] enters as [u (π r cot π r - 1)] and that of [π r] as [u π r cot π r],
   each at most [u] since [π r cot π r] lies in [\[0, 1\]]), and sin's 2 ulps of
   its value rounded are at most [5u] of the exact sine: [9u], under 9 ulps of
   the exact value and 9.5 of it rounded. Below [1/2] the polynomial in [(π
   x)²], Horner by fma with coefficients rounded to the compute dtype, is within
   about 3. *)
let sinc_ulps = 10.

(* sinc is exactly zero at the integers but [0], a zero whose sign nx.mli leaves
   open: "is 1 at zeros, 0 at the other integers" holds those. *)
let off_zeros x = x = 0. || not (Float.is_integer x)

let law_sinc32 xs =
  let has p = Array.exists p xs in
  cover "below 1/2, not 0" (has (fun x -> x <> 0. && Float.abs x < 0.5));
  cover "from 1/2 to 8" (has (fun x -> Float.abs x >= 0.5 && Float.abs x <= 8.));
  cover "near an integer"
    (has (fun x -> x <> Float.round x && Float.abs (x -. Float.round x) < 1e-3));
  law_float32 "sinc" Nx.sinc sinc_ref sinc_ulps
    (Array.of_list (List.filter off_zeros (Array.to_list xs)))

(* Float32 values in [-8, 8], near [0], near the integers, where the zeros lie,
   and from bits. *)
let sinc_values =
  let open Gen in
  let near_int =
    let+ n = int_range (-6) 6 and+ e = int_range 1 23 and+ s = bool in
    Float.of_int n +. Float.ldexp (if s then 1. else -1.) (-e)
  in
  let one =
    frequency
      [
        (3, float_range (-8.) 8.);
        (1, map (fun e -> Float.ldexp 1. (-e)) (int_range 1 149));
        (2, near_int);
        (1, map Int32.float_of_bits int32);
      ]
  in
  with_pp pp_floats
    (map (Array.map round32) (array ~size:(constant samples) one))

(* In every float format: [1] at both zeros, NaN at NaN and the infinities, and
   [0] at the integers but [0], up to the largest finite value. *)
let sinc_specials () =
  let check (type s) (dt : (float, s) D.t) =
    let f = D.float_format dt in
    let small = Array.init 32 (fun i -> Float.of_int (i + 1)) in
    let large = [| 1000.; 12345.; 0x1p23 -. 1.; 0x1p52 +. 2.; f.max_finite |] in
    let ints = Array.append small large in
    let ints = Array.append ints (Array.map Float.neg ints) in
    Array.iteri
      (fun i v ->
        let x = D.of_float dt ints.(i) in
        if Float.is_integer x && x <> 0. && v <> 0. then
          failf "%s: sinc %h is %h, not 0" (D.name dt) x v)
      (values Nx.sinc dt ints);
    equal (array float_exact) [| 1.; 1. |] (values Nx.sinc dt [| 0.; -0. |]);
    if f.infinities then
      equal (array float_exact) [| nan; nan; nan |]
        (values Nx.sinc dt [| nan; infinity; neg_infinity |])
  in
  List.iter
    (fun (D.Any dt) -> match D.kind dt with D.Float -> check dt | _ -> ())
    D.all

(* asinh, acosh, atanh *)

(* Within 2 ulps of the exact value rounded to float32, which lies within [e] of
   [y]. *)
let law_asinh xs =
  covers xs;
  let ok y e r = held D.Float32 2. (y -. e) r || held D.Float32 2. (y +. e) r in
  results_within ~name:"asinh" Nx.asinh Float.asinh ~ok xs

let signed bits =
  let open Gen in
  let+ b = bits and+ negative = bool in
  if negative then Int32.logor b Int32.min_int else b

(* The C library's acosh, NaN below 1 as C states: musl's computes numbers at
   some x below -2, -0x1.c2c2c2p+13 among them. *)
let acosh x = if x < 1. then Float.nan else Float.acosh x

(* Where the forms of nx.ml's asinh, acosh and atanh change: [2^-28], below
   which asinh and atanh are [x]; [1/2], [1] and [2]; [2^28], past which a
   square would overflow float32. *)
let asinh_near = [ 0x1p-28; 0.5; 1.; 0x1p28 ]
let acosh_near = [ 1.; 2.; 0x1p28 ]
let atanh_near = [ 0x1p-28; 0.5; 1. ]

let law_acosh xs =
  covers_near acosh_near xs;
  law_float32 "acosh" Nx.acosh acosh 3. xs

let law_atanh xs =
  covers_near atanh_near xs;
  law_float32 "atanh" Nx.atanh Float.atanh 3. xs

(* [f (-x)] is [-(f x)] bit for bit, NaN as a class, at float32 and float64,
   over [xs]. *)
let law_odd (f : 's. (float, 's, Nx.host) Nx.t -> (float, 's, Nx.host) Nx.t)
    near xs =
  covers_near near xs;
  let check (type s) (dt : (float, s) D.t) =
    let x = Nx.create dt [| Array.length xs |] (Array.map (D.of_float dt) xs) in
    let a = Nx.to_array (Nx.neg (f x)) and b = Nx.to_array (f (Nx.neg x)) in
    Array.iteri
      (fun i v ->
        if not (same v b.(i)) then
          failf "%s at %h: -f x is %h, f (-x) %h" (D.name dt) xs.(i) v b.(i))
      a
  in
  check Nx.float32;
  check Nx.float64

(* roundeven *)

(* Ties to even by the float64 arithmetic: below [2^52], [(|x| + 2^52) - 2^52]
   is [|x|] rounded to an integer, ties to even, and the sign is put back; from
   [2^52] every float is an integer. *)
let even x =
  let a = Float.abs x in
  if a < 0x1p52 then Float.copy_sign (a +. 0x1p52 -. 0x1p52) x else x

(* Float32 bit patterns to round: drawn ones, every pattern from [first] (a
   signed int32) on, [count] of them, or the representatives of every class that
   rounding to an integer tells apart. *)
type patterns =
  | Drawn of int32 array
  | Every of { first : int; count : int }
  | Representatives

let pp_patterns ppf = function
  | Drawn a -> pp_floats ppf (Array.map Int32.float_of_bits a)
  | Every { first; count } ->
      Format.fprintf ppf "every pattern from 0x%08lx, %d of them"
        (Int32.of_int first) count
  | Representatives -> Format.pp_print_string ppf "the classes' representatives"

(* roundeven of the float32s of bits [b], the [i]th [bit i], against [even],
   failing at the first that differs. *)
let check_bits (b : (int32, Nx.int32_elt, Nx.host) Nx.t) bit =
  let got =
    Bigarray.array1_of_genarray
      (Nx.to_bigarray Bigarray.int32
         (Nx.bitcast Nx.int32 (Nx.roundeven (Nx.bitcast Nx.float32 b))))
  in
  for i = 0 to Bigarray.Array1.dim got - 1 do
    let x = Int32.float_of_bits (bit i) in
    let g = Int32.float_of_bits (Bigarray.Array1.unsafe_get got i) in
    if not (same (even x) g) then failf "at %h: %h, not %h" x g (even x)
  done

let check_array a =
  check_bits (Nx.create Nx.int32 [| Array.length a |] a) (Array.get a)

(* Rounding to an integer reads, in a fraction, the half's bit, whether the bits
   below it are zero, the integer part's parity and its carry out. For every
   place [j] the half may take: the half alone (a tie), with the least bit
   (above it), the bits below it alone (below it), none and every bit, under
   integer parts of none, the lowest bit (odd), every bit (the carry) and every
   bit but the lowest; at every exponent and sign. *)
let representatives () =
  let all = (1 lsl 23) - 1 in
  let ms =
    List.concat_map
      (fun j ->
        let half = 1 lsl j and lower = (1 lsl j) - 1 in
        let above = all land lnot ((half lsl 1) - 1) in
        let lows = [ half; half lor 1; lower; 0; half lor lower ] in
        let highs =
          List.map
            (fun h -> h land all)
            [ 0; half lsl 1; above; above land lnot (half lsl 1) ]
        in
        List.concat_map (fun h -> List.map (fun l -> h lor l) lows) highs)
      (List.init 23 Fun.id)
  in
  let patterns =
    List.concat_map
      (fun e ->
        List.concat_map
          (fun m -> [ (e lsl 23) lor m; (e lsl 23) lor m lor (1 lsl 31) ])
          ms)
      (List.init 256 Fun.id)
  in
  Array.of_list (List.map Int32.of_int patterns)

let law_roundeven32 p =
  let classes xs =
    let has f = Array.exists (fun b -> f (Int32.float_of_bits b)) xs in
    let integer x = Float.of_int (truncate (Float.abs x)) in
    let tie x =
      Float.abs x >= 0.5
      && Float.abs x < 0x1p23
      && Float.abs x -. integer x = 0.5
    in
    let odd x = truncate (Float.abs x) land 1 = 1 in
    cover "a tie of an even integer part" (has (fun x -> tie x && not (odd x)));
    cover "a tie of an odd integer part" (has (fun x -> tie x && odd x));
    cover "a float next to a tie"
      (has (fun x ->
           Float.is_finite x
           && (not (tie x))
           && (tie (step32 x 1l) || tie (step32 x (-1l)))));
    cover "a carry into the next exponent"
      (has (fun x ->
           Float.is_finite x
           && Float.abs x >= 0.5
           && snd (Float.frexp (even x)) > snd (Float.frexp x)));
    cover "an integer from 2^23"
      (has (fun x -> Float.is_finite x && Float.abs x >= 0x1p23));
    cover "below 1/2" (has (fun x -> x <> 0. && Float.abs x < 0.5));
    cover "a subnormal" (has (fun x -> x <> 0. && Float.abs x < 0x1p-126));
    cover "NaN or an infinity" (has (fun x -> not (Float.is_finite x)))
  in
  match p with
  | Drawn xs ->
      classes xs;
      check_array xs
  | Every { first; count } ->
      classes [||];
      (* [2^24] patterns, 64 MiB, at a time. *)
      let chunk = 1 lsl 24 in
      let rec go start left =
        if left > 0 then begin
          let n = Int.min chunk left in
          check_bits
            (Nx.arange Nx.int32 start (start + n) 1)
            (fun i -> Int32.of_int (start + i));
          go (start + n) (left - n)
        end
      in
      go first count
  | Representatives ->
      classes [||];
      check_array (representatives ())

(* Batches of float32 patterns: an exponent mostly from those where a float
   rounds to neither zero nor itself, and a fraction placed against its half's
   bit: a tie, a float from one, the carry, or any. *)
let drawn_patterns =
  let open Gen in
  let pattern =
    let* sign = bool in
    let* e =
      frequency
        [
          (6, int_range 126 149);
          (1, constant 0);
          (1, constant 255);
          (1, int_range 150 158);
          (1, int_range 100 125);
          (1, int_range 0 255);
        ]
    in
    let all = (1 lsl 23) - 1 in
    (* The half's bit, for an exponent from 2^-1 to 2^22. *)
    let half = 1 lsl Int.max 0 (Int.min 22 (149 - e)) in
    let+ above = int_range 0 all
    and+ low = int_range 0 all
    and+ shape = int_range 0 6 in
    let high = above land lnot ((half lsl 1) - 1) in
    let m =
      match shape with
      | 0 | 1 -> high lor half
      | 2 -> high lor half lor 1
      | 3 -> high lor (half - 1)
      | 4 -> all land lnot (half - 1)
      | _ -> low
    in
    Int32.of_int ((e lsl 23) lor m lor if sign then 1 lsl 31 else 0)
  in
  with_pp pp_patterns
    (map (fun a -> Drawn a) (array ~size:(constant 256) pattern))

let roundeven32_examples =
  [
    Representatives;
    (* [1/2, 2) of each sign, where a tie rounds to 0 or 2 and [1.5] carries. *)
    Every { first = 0x3f00_0000; count = 1 lsl 24 };
    Every { first = 0x3f00_0000 - (1 lsl 31); count = 1 lsl 24 };
    (* [2^22, 2^24) of each sign: every other float a tie below [2^23], every
       float an integer above. *)
    Every { first = 0x4a80_0000; count = 1 lsl 24 };
    Every { first = 0x4a80_0000 - (1 lsl 31); count = 1 lsl 24 };
  ]

(* Float64s to round: drawn bits, or the halves, quarters and integers up to
   [2^53] and the ends, each with its neighbours. Each runs with its
   negations. *)
type doubles = Doubles of float array | Structured

let pp_doubles ppf = function
  | Doubles a -> pp_floats ppf a
  | Structured -> Format.pp_print_string ppf "halves, quarters and ends"

let structured () =
  let rng = Random.State.make [| 2026 |] in
  let below k = Int64.to_float (Random.State.int64 rng k) in
  let halves = List.init 65536 (fun _ -> below 0x10_0000_0000_0000L +. 0.5) in
  let quarters =
    List.init 65536 (fun _ -> ((2. *. below 0x8_0000_0000_0000L) +. 1.) /. 4.)
  in
  let small = List.init 8001 (fun i -> Float.of_int (i - 4000) /. 2.) in
  let ends =
    [
      0.;
      0.49999999999999994;
      0.5;
      0x1p51 +. 0.5;
      0x1p52 -. 0.5;
      0x1p52 -. 1.5;
      0x1p52;
      0x1p52 +. 1.;
      0x1p53;
      0x1p-1074;
      Float.max_float;
      infinity;
      nan;
    ]
  in
  let around x = [ Float.pred x; x; Float.succ x ] in
  List.concat_map around (halves @ quarters @ small @ ends)

let law_roundeven64 d =
  let xs =
    match d with Doubles a -> Array.to_list a | Structured -> structured ()
  in
  let xs = Array.of_list (xs @ List.map Float.neg xs) in
  let got = values Nx.roundeven Nx.float64 xs in
  Array.iteri
    (fun i x ->
      if not (same (even x) got.(i)) then
        failf "at %h: %h, not %h" x got.(i) (even x))
    xs

let drawn_doubles =
  Gen.with_pp pp_doubles
    (Gen.map
       (fun a -> Doubles (Array.map Int64.float_of_bits a))
       (Gen.array ~size:(Gen.constant 512) Gen.int64))

(* nx.mli's examples, an integer dtype its identity, and float32's greatest
   float below a half, which rounds to [0.]. *)
let roundeven_others () =
  let ints = Nx.create Nx.int32 [| 3 |] [| -7l; 0l; 9l |] in
  equal (array int32) [| -7l; 0l; 9l |] (Nx.to_array (Nx.roundeven ints));
  equal (array float_exact)
    [| 2.; 4.; -0.; -2.; 3.; 0. |]
    (values Nx.roundeven Nx.float32
       [| 2.5; 3.5; -0.5; -2.5; 2.5000002; 0x1.fffffep-2 |])

let math =
  [
    group "the section"
      [
        cases "Annex F's values nx.mli lists are exact" ~id:dtype_id f32_f64
          test_annex_f;
        test "a format without infinities or NaN stores as a cast" no_infinities;
      ];
    group "rsqrt"
      [
        prop "float64 within 2 ulps" (drawn D.Float64 samples)
          (law_rsqrt D.Float64);
        prop "float32 within 2 ulps" (drawn D.Float32 samples)
          (law_rsqrt D.Float32);
        test "within 2 ulps at the ends of the range" (fun () ->
            rsqrt_ends D.Float64;
            rsqrt_ends D.Float32);
        cases "Annex F's values, and asinh's, acosh's and atanh's" ~id:dtype_id
          f32_f64 specials;
        narrow (fun x -> Nx.rsqrt x);
      ];
    group "hypot"
      [
        cases "within 3 ulps of the golden" ~id:at_id
          [ At (D.Float32, 3.); At (D.Float64, 3.) ]
          (golden2 "hypot" (fun x y -> Nx.hypot x y));
        prop "float32 within 3 ulps"
          (Gen.pair (drawn D.Float32 samples) (drawn D.Float32 samples))
          law_hypot32;
        cases
          "is infinite beside a NaN, exact at zeros, with no square overflowing"
          ~id:dtype_id f32_f64 hypot_specials;
        narrow (fun x -> Nx.hypot x (Nx.flip x));
      ];
    group "log"
      [
        prop ~examples:edge_examples
          "float32 is one of the two floats around the exact value"
          (float32_values positive_bits)
          law_log;
      ];
    group "log10"
      [
        cases "within its ulps of the golden" ~id:at_id
          [ At (D.Float32, 1.); At (D.Float64, 4.) ]
          (golden1 "log10" (fun x -> Nx.log10 x));
        test "is n at each 10^n the dtype holds, with Annex F's values"
          log10_values;
        test "float32 within 1 ulp at every float16 and bfloat16"
          log10_wide_codes;
        narrow (fun x -> Nx.log10 x);
      ];
    group "sinc"
      [
        cases "within 10 ulps of the golden" ~id:at_id
          [ At (D.Float32, sinc_ulps); At (D.Float64, sinc_ulps) ]
          (golden1 ~keep:(fun a -> off_zeros a.(0)) "sinc" (fun x -> Nx.sinc x));
        prop "float32 within 10 ulps" sinc_values law_sinc32;
        test "is 1 at zeros, 0 at the other integers, NaN at NaN and infinities"
          sinc_specials;
        narrow (fun x -> Nx.sinc x);
      ];
    group "asinh"
      [
        cases "within its ulps of the golden" ~id:at_id
          [ At (D.Float32, 2.); At (D.Float64, 3.) ]
          (golden1 "asinh" (fun x -> Nx.asinh x));
        prop ~examples:edge_examples "float32 within 2 ulps"
          (float32_values (signed positive_bits))
          law_asinh;
        prop "is odd, bit for bit" (float32s asinh_near)
          (law_odd (fun x -> Nx.asinh x) asinh_near);
        narrow (fun x -> Nx.asinh x);
      ];
    group "acosh"
      [
        cases "within 3 ulps of the golden" ~id:at_id
          [ At (D.Float32, 3.); At (D.Float64, 3.) ]
          (golden1 "acosh" (fun x -> Nx.acosh x));
        prop "float32 within 3 ulps"
          ~examples:[ [| -0x1.c2c2c2p+13; -0x1.161636p+23 |] ]
          (float32s acosh_near) law_acosh;
        narrow (fun x -> Nx.acosh x);
      ];
    group "atanh"
      [
        cases "within 3 ulps of the golden" ~id:at_id
          [ At (D.Float32, 3.); At (D.Float64, 3.) ]
          (golden1 "atanh" (fun x -> Nx.atanh x));
        prop "float32 within 3 ulps" (float32s atanh_near) law_atanh;
        prop "is odd, bit for bit" (float32s atanh_near)
          (law_odd (fun x -> Nx.atanh x) atanh_near);
        narrow (fun x -> Nx.atanh x);
      ];
    group "roundeven"
      [
        prop ~examples:roundeven32_examples
          "float32 is the float64 arithmetic's ties to even" drawn_patterns
          law_roundeven32;
        prop ~examples:[ Structured ]
          "float64 is its own arithmetic's ties to even" drawn_doubles
          law_roundeven64;
        test "is nx.mli's examples, and an integer itself" roundeven_others;
        narrow (fun x -> Nx.roundeven x);
      ];
    group "dtypes"
      [
        cases "a function of floats refuses another dtype"
          ~id:(fun (n, _) -> n)
          [
            ("Nx.rsqrt", fun () -> ignore (Nx.rsqrt (Nx.zeros Nx.int32 [| 1 |])));
            ( "Nx.hypot",
              fun () ->
                let z = Nx.zeros Nx.complex64 [| 1 |] in
                ignore (Nx.hypot z z) );
            ("Nx.asinh", fun () -> ignore (Nx.asinh (Nx.zeros Nx.bool [| 1 |])));
            ("Nx.acosh", fun () -> ignore (Nx.acosh (Nx.zeros Nx.uint8 [| 1 |])));
            ("Nx.atanh", fun () -> ignore (Nx.atanh (Nx.zeros Nx.int64 [| 1 |])));
            ( "Nx.square",
              fun () -> ignore (Nx.square (Nx.zeros Nx.bool [| 1 |])) );
            ( "Nx.roundeven",
              fun () -> ignore (Nx.roundeven (Nx.zeros Nx.complex64 [| 1 |])) );
          ]
          (fun (by, f) -> invalid ~by f);
      ];
  ]

let () = exit (run "nx math" math)
