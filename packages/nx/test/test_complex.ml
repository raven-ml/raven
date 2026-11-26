(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Complex numbers, in nx.mli's order: the parts, against their bits computed in
   OCaml; magnitude and angle as nx.mli defines them, hypot and atan2 of the
   parts in the wider format; conjugate; the complex functions at the values C99
   defines, within nx.mli's normwise bounds. Values are drawn from bytes, into
   every float witness. *)

open Windtrap
open Nx_support
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let invalid ~by f = raises_match (Exn.invalid_arg ~sub:(by ^ ": ")) f

let bits_equal a b =
  (Float.is_nan a && Float.is_nan b)
  || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let floats =
  Testable.make
    ~pp:(Format.pp_print_list (fun ppf v -> Format.fprintf ppf "%h" v))
    ~equal:(fun a b ->
      List.length a = List.length b && List.for_all2 bits_equal a b)

let equal_floats a b = equal floats (Array.to_list a) (Array.to_list b)

let drawn (type s) (dt : (Complex.t, s) D.t) shape :
    (Complex.t, s, Nx.host) Nx.t Gen.t =
  let open Gen in
  let n = Array.fold_left ( * ) 1 shape in
  let+ data = string_of ~size:(constant (D.bytes dt n)) char in
  Nx.Repr.of_array Nx.Host.v
    (A.v dt
       (L.contiguous (Iarray.of_array shape))
       (Rig.Buffer.of_string (if data = "" then "\000" else data)))

let shape =
  Gen.array ~size:(Gen.int_range 0 2)
    (Gen.frequency
       [ (1, Gen.constant 0); (1, Gen.constant 1); (4, Gen.int_range 2 5) ])

type z = Z : (Complex.t, 's) D.t * (Complex.t, 's, Nx.host) Nx.t -> z

let pp_z ppf (Z (dt, x)) =
  Format.fprintf ppf "%a %a" D.pp dt pp_ints (Nx.shape x)

let complexes =
  Gen.with_pp pp_z
    (let open Gen in
     let* complex128 = bool in
     let* s = shape in
     if complex128 then
       let+ x = drawn D.Complex128 s in
       Z (D.Complex128, x)
     else
       let+ x = drawn D.Complex64 s in
       Z (D.Complex64, x))

type f = F : (float, 's) D.t -> f

let witnesses =
  List.filter_map
    (fun (D.Any dt) ->
      match D.kind dt with D.Float -> Some (F dt) | _ -> None)
    D.all

let witness = Gen.of_list ~pp:(fun ppf (F dt) -> D.pp ppf dt) witnesses
let res z = Array.map (fun c -> c.Complex.re) (Nx.to_array z)
let ims z = Array.map (fun c -> c.Complex.im) (Nx.to_array z)

let values dt l =
  Nx.create dt
    [| List.length l |]
    (Array.of_list (List.map (fun (re, im) -> { Complex.re; im }) l))

(* An example of each complex dtype with no element. *)
let empty =
  [
    Z (D.Complex64, Nx.zeros D.Complex64 [| 0 |]);
    Z (D.Complex128, Nx.zeros D.Complex128 [| 2; 0 |]);
  ]

(* Parts *)

let law_real (Z (_, z), F f) =
  cover "no element" (Nx.numel z = 0);
  equal_floats (Array.map (D.of_float f) (res z)) (Nx.to_array (Nx.real f z));
  equal_floats (Array.map (D.of_float f) (ims z)) (Nx.to_array (Nx.imag f z))

(* A complex value's bits: its parts' bits read as integers of their width. *)
let bit_pattern (type s) (z : (Complex.t, s, Nx.host) Nx.t) =
  match Nx.dtype z with
  | D.Complex64 ->
      Array.map Int64.of_int32
        (Nx.to_array (Nx.bitcast Nx.int32 (Nx.bitcast Nx.float32 z)))
  | D.Complex128 -> Nx.to_array (Nx.bitcast Nx.int64 (Nx.bitcast Nx.float64 z))

let law_round_trip (Z (dt, z)) =
  let back (type p) (f : (float, p) D.t) =
    Nx.complex dt ~re:(Nx.real f z) ~im:(Nx.imag f z)
  in
  let y =
    match dt with
    | D.Complex64 -> back D.Float32
    | D.Complex128 -> back D.Float64
  in
  equal (array int64) (bit_pattern z) (bit_pattern y)

let test_complex_broadcast () =
  let re = Nx.create Nx.float16 [| 2; 1 |] [| 1.; -0. |] in
  let im = Nx.create Nx.float16 [| 3 |] [| infinity; nan; 0.5 |] in
  let z = Nx.complex Nx.complex128 ~re ~im in
  equal (array int) [| 2; 3 |] (Nx.shape z);
  equal_floats [| 1.; 1.; 1.; -0.; -0.; -0. |] (res z);
  equal_floats [| infinity; nan; 0.5; infinity; nan; 0.5 |] (ims z);
  invalid ~by:"Nx.complex" (fun () ->
      Nx.complex Nx.complex64
        ~re:(Nx.zeros Nx.float32 [| 2 |])
        ~im:(Nx.zeros Nx.float32 [| 3 |]))

(* Magnitude and angle *)

(* [f w z] is [g] of [z]'s parts computed in the wider of [w] and the parts'
   format, then stored in [w], bit for bit, NaN as a class. *)
let law_polar
    (f :
      'r 's.
      (float, 'r) D.t ->
      (Complex.t, 's, Nx.host) Nx.t ->
      (float, 'r, Nx.host) Nx.t)
    (g :
      'p.
      (float, 'p, Nx.host) Nx.t ->
      (float, 'p, Nx.host) Nx.t ->
      (float, 'p, Nx.host) Nx.t) (Z (dt, z), F w) =
  let parts (type p) (p : (float, p) D.t) =
    Nx.to_array (Nx.cast w (g (Nx.real p z) (Nx.imag p z)))
  in
  let expected =
    match (dt, D.bits w) with
    | D.Complex128, _ | D.Complex64, 64 -> parts D.Float64
    | D.Complex64, _ -> parts D.Float32
  in
  equal_floats expected (Nx.to_array (f w z))

(* [6 + 8i] of either dtype, whose modulus [10] every format stores, [1e300 +
   4e300i], whose parts are infinite at float32, and [3e38 + 3e38i], whose
   modulus overflows float32. *)
let polar_examples =
  List.concat_map
    (fun w ->
      [
        (Z (D.Complex64, values D.Complex64 [ (6., 8.); (3e38, 3e38) ]), w);
        (Z (D.Complex128, values D.Complex128 [ (6., 8.); (1e300, 4e300) ]), w);
      ])
    witnesses

let test_branch_cut () =
  let z =
    values Nx.complex64
      [
        (-1., 0.);
        (-1., -0.);
        (0., 0.);
        (3., 4.);
        (-0., 0.);
        (-0., -0.);
        (0., -0.);
      ]
  in
  equal_floats
    [| Float.pi; -.Float.pi; 0.; Float.atan2 4. 3.; Float.pi; -.Float.pi; -0. |]
    (Nx.to_array (Nx.angle Nx.float64 z));
  equal_floats
    [| 1.; 1.; 0.; 5.; 0.; 0.; 0. |]
    (Nx.to_array (Nx.magnitude Nx.float64 z))

(* Conjugate *)

let law_conjugate (Z (_, z)) =
  let y = Nx.conjugate z in
  equal_floats (res z) (res y);
  equal_floats (Array.map Float.neg (ims z)) (ims y)

let conjugate_examples =
  [
    Z (D.Complex64, values D.Complex64 [ (1., 0.); (2., -0.); (nan, 0.) ]);
    Z (D.Complex128, values D.Complex128 [ (1., 0.); (-0., -0.) ]);
  ]

let test_conjugate_real () =
  let x = Nx.create Nx.int8 [| 2 |] [| 1; -1 |] in
  satisfies ~claim:"x itself" pass (fun y -> y == x) (Nx.conjugate x)

(* Functions *)

(* A complex value's elements as pairs of part bits, so that values compare bit
   for bit, any NaN for a NaN. *)
let parts_of (type s) (z : (Complex.t, s, Nx.host) Nx.t) =
  Array.to_list (Array.map (fun c -> [ c.Complex.re; c.im ]) (Nx.to_array z))
  |> List.concat

let law_rsqrt (Z (_, z)) =
  equal floats (parts_of (Nx.recip (Nx.sqrt z))) (parts_of (Nx.rsqrt z))

(* On the positive real axis, rsqrt's real part is the float rsqrt's. *)
let law_rsqrt_real xs =
  let x = Nx.create Nx.float64 [| Array.length xs |] xs in
  let z = Nx.complex Nx.complex128 ~re:x ~im:(Nx.zeros_like x) in
  equal_floats (Nx.to_array (Nx.rsqrt x)) (res (Nx.rsqrt z))

let positive =
  Gen.with_pp
    (fun ppf xs ->
      Format.pp_print_list
        ~pp_sep:(fun ppf () -> Format.pp_print_string ppf " ")
        (fun ppf v -> Format.fprintf ppf "%h" v)
        ppf (Array.to_list xs))
    (Gen.array ~size:(Gen.constant 16)
       (Gen.map
          (fun b -> Float.abs (Int64.float_of_bits b))
          (Gen.such_that
             (fun b ->
               let x = Float.abs (Int64.float_of_bits b) in
               Float.is_finite x && x > 0.)
             Gen.int64)))

let test_abs_refuses () =
  let z = Nx.create Nx.complex64 [| 1 |] [| { Complex.re = 3.; im = 4. } |] in
  invalid ~by:"Nx.abs" (fun () -> Nx.abs z)

(* C99's values at complex128, each within nx.mli's [k u |f|] of the exact value
   [f], [u = 2^-53]: [exp (iπ) = -1] (k 4), [log (-4 + 0i) = log 4 + iπ] (4),
   [asinh 2i = log (2 + √3) + iπ/2] (5); and exactly [sqrt (-4 ± 0i) = ±2i],
   [sign (inf + 2i) = 1 + 0i] and [rsqrt 0 = NaN + NaN i]. The exact values' own
   roundings, an ulp of each part, are added. *)
let test_c99_values () =
  let u = 0x1p-53 in
  let near ~msg k (want : Complex.t) (got : Complex.t) =
    let err = Float.hypot (gap got.re want.re) (gap got.im want.im) in
    let bound =
      (k *. u *. Complex.norm want) +. (2. *. u *. Complex.norm want)
    in
    at_most ~msg float_exact ~than:bound err
  in
  let one f re im = (Nx.to_array (f (values Nx.complex128 [ (re, im) ]))).(0) in
  near ~msg:"exp (i pi)" 4. { re = -1.; im = 0. } (one Nx.exp 0. Float.pi);
  near ~msg:"log (-4 + 0i)" 4.
    { re = Float.log 4.; im = Float.pi }
    (one Nx.log (-4.) 0.);
  near ~msg:"asinh 2i" 5.
    { re = Float.log (2. +. Float.sqrt 3.); im = Float.pi /. 2. }
    (one Nx.asinh 0. 2.);
  let parts f re im = parts_of (f (values Nx.complex128 [ (re, im) ])) in
  equal ~msg:"sqrt (-4 + 0i)" floats [ 0.; 2. ] (parts Nx.sqrt (-4.) 0.);
  equal ~msg:"sqrt (-4 - 0i)" floats [ 0.; -2. ] (parts Nx.sqrt (-4.) (-0.));
  equal ~msg:"sign (inf + 2i)" floats [ 1.; 0. ] (parts Nx.sign infinity 2.);
  equal ~msg:"rsqrt 0" floats [ Float.nan; Float.nan ]
    (parts_of (Nx.rsqrt (values Nx.complex64 [ (0., 0.) ])))

(* nx_kinds_complex.h's pow takes a NaN imaginary part of [w log z] as the NaN
   of positive sign, so that where the real part is [-inf] the result is [+0 +
   i0], whatever sign the target gives a NaN it computes. Each pair reaches that
   case: an infinite part of [w] times a zero part of [log z], infinite products
   of opposite signs summed, or a NaN part of [z] beside an infinite one.
   nx.cuda's suite holds every kind to these bits. *)
let inf = Float.infinity
let nan = Float.nan
let neg_nan = Float.neg Float.nan

let vanishing_powers =
  [
    ("(0 + 0i)^(inf + 0i)", (0., 0.), (inf, 0.));
    ("(0 - 0i)^(inf + 0i)", (0., -0.), (inf, 0.));
    ("(0.5 + 0i)^(inf + i)", (0.5, 0.), (inf, 1.));
    ("(0 + i)^(0 + inf i)", (0., 1.), (0., inf));
    ("(inf + i)^(-inf + 0i)", (inf, 1.), (Float.neg_infinity, 0.));
    ("(0.5 + 0.5i)^(inf + inf i)", (0.5, 0.5), (inf, inf));
    ("(inf + NaN i)^(-2 + 0i)", (inf, nan), (-2., 0.));
    ("(inf - NaN i)^(-2 + 0i)", (inf, neg_nan), (-2., 0.));
    ("(NaN + inf i)^(-1 + 0i)", (nan, inf), (-1., 0.));
    ("(-NaN + inf i)^(-1 + 0i)", (neg_nan, inf), (-1., 0.));
  ]

(* One element of [dt] whose parts have the bits of [re] and [im] rounded to the
   parts' format, a NaN's sign kept. *)
let element (type s) (dt : (Complex.t, s) D.t) (re, im) :
    (Complex.t, s, Nx.host) Nx.t =
  match dt with
  | D.Complex64 ->
      let part v =
        Nx.bitcast Nx.float32
          (Nx.create Nx.int32 [| 1 |] [| Int32.bits_of_float v |])
      in
      Nx.complex dt ~re:(part re) ~im:(part im)
  | D.Complex128 ->
      let part v =
        Nx.bitcast Nx.float64
          (Nx.create Nx.int64 [| 1 |] [| Int64.bits_of_float v |])
      in
      Nx.complex dt ~re:(part re) ~im:(part im)

(* The parts at complex64, then at complex128. *)
let test_vanishing_power (_, z, w) =
  let at (type s) (dt : (Complex.t, s) D.t) =
    parts_of (Nx.pow (element dt z) (element dt w))
  in
  equal floats [ 0.; 0.; 0.; 0. ] (at D.Complex64 @ at D.Complex128)

let () =
  exit
    (run "nx complex"
       [
         group "real and imag"
           [
             prop "are the parts, stored in the witness"
               ~examples:
                 (List.concat_map
                    (fun z -> List.map (fun w -> (z, w)) witnesses)
                    empty)
               (Gen.pair complexes witness)
               law_real;
           ];
         group "magnitude"
           [
             prop ~examples:polar_examples
               "is hypot of the parts in the wider format, stored in dt"
               (Gen.pair complexes witness)
               (law_polar
                  (fun w z -> Nx.magnitude w z)
                  (fun a b -> Nx.hypot a b));
           ];
         group "angle"
           [
             prop ~examples:polar_examples
               "is atan2 of the parts in the wider format, stored in dt"
               (Gen.pair complexes witness)
               (law_polar
                  (fun w z -> Nx.angle w z)
                  (fun re im -> Nx.atan2 im re));
             test "takes its branch cut's side from a zero's sign"
               test_branch_cut;
           ];
         group "complex"
           [
             prop "of the parts is the value, bit for bit" complexes
               law_round_trip;
             test "broadcasts and keeps every part" test_complex_broadcast;
           ];
         group "conjugate"
           [
             prop ~examples:conjugate_examples "negates the imaginary parts"
               complexes law_conjugate;
             test "a value that is not complex is itself" test_conjugate_real;
           ];
         group "functions"
           [
             prop "rsqrt is recip of sqrt" complexes law_rsqrt;
             prop "rsqrt on the positive real axis is the float rsqrt" positive
               law_rsqrt_real;
             test "abs refuses a complex value, naming itself" test_abs_refuses;
             test "C99's values, within nx.mli's bounds" test_c99_values;
             cases "pow is +0 + i0 where w log z is -inf + i NaN"
               ~id:(fun (name, _, _) -> name)
               vanishing_powers test_vanishing_power;
           ];
       ])
