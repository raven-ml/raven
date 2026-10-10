(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Complex parts: each function against its parts computed in OCaml, over
   complex values drawn from bytes, into every float witness. *)

open Windtrap
module A = Nx_array
module D = Nx_array.Dtype
module L = Nx_array.Layout

let invalid ~by f = raises_match (Exn.invalid_arg ~substring:(by ^ ": ")) f

let pp_ints ppf a =
  Format.fprintf ppf "[%s]"
    (String.concat "; " (Array.to_list (Array.map string_of_int a)))

let bits_equal a b =
  (Float.is_nan a && Float.is_nan b)
  || Int64.equal (Int64.bits_of_float a) (Int64.bits_of_float b)

let floats =
  Testable.make
    ~pp:(Format.pp_print_list (fun ppf v -> Format.fprintf ppf "%h" v))
    ~equal:(fun a b -> List.length a = List.length b && List.for_all2 bits_equal a b)

let equal_floats a b = equal floats (Array.to_list a) (Array.to_list b)

let drawn (type s) (dt : (Complex.t, s) D.t) shape :
    (Complex.t, s, Nx.host) Nx.t Gen.t =
  let open Gen in
  let n = Array.fold_left ( * ) 1 shape in
  let+ data = string_of ~size:(constant (D.bytes dt n)) char in
  Nx.Repr.of_array Nx.Host.v
    (A.v dt (L.contiguous shape)
       (Rig.Buffer.of_string (if data = "" then "\000" else data)))

let shape =
  Gen.array ~size:(Gen.int_range 0 2)
    (Gen.frequency [ (1, Gen.constant 0); (1, Gen.constant 1); (4, Gen.int_range 2 5) ])

type z = Z : (Complex.t, 's) D.t * (Complex.t, 's, Nx.host) Nx.t -> z

let complexes =
  Gen.with_pp
    (fun ppf (Z (dt, x)) -> Format.fprintf ppf "%a %a" D.pp dt pp_ints (Nx.shape x))
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
    (fun (D.Any dt) -> match D.kind dt with D.Float -> Some (F dt) | _ -> None)
    D.all

let witness =
  Gen.of_list ~pp:(fun ppf (F dt) -> D.pp ppf dt) witnesses

let res z = Array.map (fun c -> c.Complex.re) (Nx.to_array z)
let ims z = Array.map (fun c -> c.Complex.im) (Nx.to_array z)

(* Parts *)

let law_real (Z (_, z), F f) =
  cover "no element" (Nx.numel z = 0);
  equal_floats (Array.map (D.of_float f) (res z)) (Nx.to_array (Nx.real f z));
  equal_floats (Array.map (D.of_float f) (ims z)) (Nx.to_array (Nx.imag f z))

(* A complex value's bits: its parts' bits read as integers of their width. *)
let bit_pattern (type s) (z : (Complex.t, s, Nx.host) Nx.t) =
  match Nx.dtype z with
  | D.Complex64 ->
      Array.map Int64.of_int32 (Nx.to_array (Nx.bitcast Nx.int32 (Nx.bitcast Nx.float32 z)))
  | D.Complex128 -> Nx.to_array (Nx.bitcast Nx.int64 (Nx.bitcast Nx.float64 z))

let law_round_trip (Z (dt, z)) =
  let back (type p) (f : (float, p) D.t) =
    Nx.complex dt ~re:(Nx.real f z) ~im:(Nx.imag f z)
  in
  let y = match dt with D.Complex64 -> back D.Float32 | D.Complex128 -> back D.Float64 in
  equal (array int64) (bit_pattern z) (bit_pattern y)

let test_complex_broadcast () =
  let re = Nx.create Nx.float16 [| 2; 1 |] [| 1.; -0. |] in
  let im = Nx.create Nx.float16 [| 3 |] [| infinity; nan; 0.5 |] in
  let z = Nx.complex Nx.complex128 ~re ~im in
  equal (array int) [| 2; 3 |] (Nx.shape z);
  equal_floats [| 1.; 1.; 1.; -0.; -0.; -0. |] (res z);
  equal_floats [| infinity; nan; 0.5; infinity; nan; 0.5 |] (ims z);
  invalid ~by:"Nx.complex" (fun () ->
      Nx.complex Nx.complex64 ~re:(Nx.zeros Nx.float32 [| 2 |])
        ~im:(Nx.zeros Nx.float32 [| 3 |]))

(* Magnitude and angle *)

let ulp (type s) (dt : (float, s) D.t) r =
  let r = Float.abs (D.of_float dt r) in
  match dt with
  | D.Float32 ->
      let b = Int32.bits_of_float r in
      Int32.float_of_bits (Int32.succ b) -. r
  | _ -> Float.succ r -. r

let close dt k r v =
  let r' = D.of_float dt r in
  if Float.is_nan r then Float.is_nan v
  else if (not (Float.is_finite r')) || r = 0. then bits_equal r' v
  else Float.abs (v -. r) <= k *. ulp dt r

let law_polar fname
    (f : 'r 's. (float, 'r) D.t -> (Complex.t, 's, Nx.host) Nx.t -> (float, 'r, Nx.host) Nx.t)
    libm k (Z (_, z), F w) =
  match w with
  | D.Float32 | D.Float64 ->
      let got = Nx.to_array (f w z) in
      Array.iteri
        (fun i c ->
          let r = libm c.Complex.re c.im in
          if not (close w k r got.(i)) then
            failf "%s at %h%+hi: %h, the C library's %h" fname c.re c.im
              got.(i) r)
        (Nx.to_array z)
  | _ -> ()

let polar =
  [
    prop "magnitude is hypot of the parts, within 3 ulps"
      (Gen.pair complexes witness)
      (law_polar "magnitude" (fun w z -> Nx.magnitude w z) Float.hypot 3.);
    prop "angle is atan2 of the parts, within 3 ulps"
      (Gen.pair complexes witness)
      (law_polar "angle" (fun w z -> Nx.angle w z) (fun re im -> Float.atan2 im re) 3.);
    test "the branch cut and zero" (fun () ->
        let z =
          Nx.create Nx.complex64 [| 4 |]
            [|
              { Complex.re = -1.; im = 0. };
              { re = -1.; im = -0. };
              { re = 0.; im = 0. };
              { re = 3.; im = 4. };
            |]
        in
        equal_floats
          [| Float.pi; -.Float.pi; 0.; Float.atan2 4. 3. |]
          (Nx.to_array (Nx.angle Nx.float64 z));
        equal_floats [| 1.; 1.; 0.; 5. |] (Nx.to_array (Nx.magnitude Nx.float64 z)));
    test "a large modulus does not overflow" (fun () ->
        let z = Nx.create Nx.complex64 [| 1 |] [| { Complex.re = 3e38; im = 3e38 } |] in
        equal_floats [| D.of_float D.Float32 (Float.hypot 3e38 3e38) |]
          [| (Nx.to_array (Nx.magnitude Nx.float32 z)).(0) |]);
  ]

(* Conjugate *)

let law_conjugate (Z (_, z)) =
  let y = Nx.conjugate z in
  equal_floats (res z) (res y);
  equal_floats (Array.map Float.neg (ims z)) (ims y)

let test_conjugate_real () =
  let x = Nx.create Nx.int8 [| 2 |] [| 1; -1 |] in
  equal bool true (Nx.conjugate x == x)

let () =
  exit
    (run "nx complex"
       [
         group "parts"
           [
             prop "real and imag are the parts, stored in the witness"
               (Gen.pair complexes witness) law_real;
             prop "complex of the parts is the value, bit for bit" complexes
               law_round_trip;
             test "complex broadcasts and keeps every part"
               test_complex_broadcast;
           ];
         group "polar" polar;
         group "conjugate"
           [
             prop "conjugate negates the imaginary parts" complexes law_conjugate;
             test "a value that is not complex is itself" test_conjugate_real;
           ];
       ])
