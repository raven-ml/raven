(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk

(* ───── Testables ───── *)

let dtype = testable ~pp:Dtype.pp ~equal:Dtype.equal ()

let bound =
  let pp fmt = function
    | `Bool b -> Format.fprintf fmt "`Bool %b" b
    | `SInt n -> Format.fprintf fmt "`SInt %Ld" n
    | `UInt n -> Format.fprintf fmt "`UInt %Ld" n
    | `Float f -> Format.fprintf fmt "`Float %g" f
  in
  let equal a b =
    match (a, b) with
    | `Bool a, `Bool b -> a = b
    | `SInt a, `SInt b -> Int64.equal a b
    | `UInt a, `UInt b -> Int64.equal a b
    | `Float a, `Float b ->
        (Float.is_nan a && Float.is_nan b) || Float.equal a b
    | _ -> false
  in
  testable ~pp ~equal ()

let int_pair =
  let pp fmt (a, b) = Format.fprintf fmt "(%d, %d)" a b in
  testable ~pp ~equal:( = ) ()

(* Dtypes that participate in promotion (excludes Void and Index). *)
let promotable_dtypes =
  Dtype.
    [
      bool;
      int8;
      int16;
      int32;
      int64;
      uint8;
      uint16;
      uint32;
      uint64;
      float16;
      bfloat16;
      float32;
      float64;
      fp8e4m3;
      fp8e5m2;
    ]

let promotable_dtype =
  let gen = Gen.oneofl promotable_dtypes in
  testable ~pp:Dtype.pp ~equal:Dtype.equal ~gen ()

(* Integer dtypes suitable for truncate_int (excludes Index). *)
let int_dtypes = Dtype.[ bool; int8; int16; int32; uint8; uint16; uint32 ]

let int_dtype =
  let gen = Gen.oneofl int_dtypes in
  testable ~pp:Dtype.pp ~equal:Dtype.equal ~gen ()

(* ───── Type Promotion ───── *)

let test_promo_lattice () =
  (* Adjacent pairs: verify key edges in the promotion graph. *)
  equal dtype Dtype.int8 (Dtype.least_upper_dtype [ Dtype.bool; Dtype.int8 ]);
  equal dtype Dtype.int16 (Dtype.least_upper_dtype [ Dtype.int8; Dtype.uint8 ]);
  equal dtype Dtype.int32
    (Dtype.least_upper_dtype [ Dtype.int16; Dtype.uint16 ]);
  equal dtype Dtype.int64
    (Dtype.least_upper_dtype [ Dtype.int32; Dtype.uint32 ]);
  (* Cross-category: int through float. *)
  equal dtype Dtype.float16
    (Dtype.least_upper_dtype [ Dtype.float16; Dtype.int64 ]);
  (* FP8 siblings meet at float16. *)
  equal dtype Dtype.float16
    (Dtype.least_upper_dtype [ Dtype.fp8e4m3; Dtype.fp8e5m2 ]);
  (* Float16 and bfloat16 are incomparable; they meet at float32. *)
  equal dtype Dtype.float32
    (Dtype.least_upper_dtype [ Dtype.float16; Dtype.bfloat16 ])

let test_promo_strips_vec () =
  let vec4 = Dtype.vec Dtype.int8 4 in
  equal dtype Dtype.int16 (Dtype.least_upper_dtype [ vec4; Dtype.uint8 ])

let test_promo_errors () =
  raises_invalid_arg "least_upper_dtype requires at least one dtype" (fun () ->
      Dtype.least_upper_dtype []);
  raises_invalid_arg "Index does not participate in dtype promotion" (fun () ->
      Dtype.least_upper_dtype [ Dtype.index ])

(* ───── Lossless Cast ───── *)

let test_lossless_widening () =
  is_true (Dtype.can_lossless_cast Dtype.int8 Dtype.int16);
  is_true (Dtype.can_lossless_cast Dtype.int16 Dtype.int32);
  is_true (Dtype.can_lossless_cast Dtype.uint8 Dtype.uint16);
  is_true (Dtype.can_lossless_cast Dtype.float16 Dtype.float32);
  is_true (Dtype.can_lossless_cast Dtype.float32 Dtype.float64);
  is_true (Dtype.can_lossless_cast Dtype.fp8e4m3 Dtype.float16);
  is_true (Dtype.can_lossless_cast Dtype.fp8e5m2 Dtype.float16)

let test_lossless_narrowing_fails () =
  is_false (Dtype.can_lossless_cast Dtype.int32 Dtype.int16);
  is_false (Dtype.can_lossless_cast Dtype.float64 Dtype.float32);
  is_false (Dtype.can_lossless_cast Dtype.float16 Dtype.fp8e4m3)

let test_lossless_cross_sign () =
  (* uint8 fits in int16 (wider signed). *)
  is_true (Dtype.can_lossless_cast Dtype.uint8 Dtype.int16);
  (* int8 doesn't fit in uint8 (loses negatives). *)
  is_false (Dtype.can_lossless_cast Dtype.int8 Dtype.uint8);
  is_false (Dtype.can_lossless_cast Dtype.int16 Dtype.uint16)

let test_lossless_to_index () =
  is_true (Dtype.can_lossless_cast Dtype.int32 Dtype.index);
  is_true (Dtype.can_lossless_cast Dtype.uint64 Dtype.index);
  is_false (Dtype.can_lossless_cast Dtype.float32 Dtype.index)

(* ───── Sum Accumulator ───── *)

let test_sum_acc () =
  (* Unsigned widens to at least uint32. *)
  equal dtype Dtype.uint32 (Dtype.sum_acc_dtype Dtype.uint8);
  equal dtype Dtype.uint32 (Dtype.sum_acc_dtype Dtype.uint32);
  equal dtype Dtype.uint64 (Dtype.sum_acc_dtype Dtype.uint64);
  (* Signed widens to at least int32. *)
  equal dtype Dtype.int32 (Dtype.sum_acc_dtype Dtype.int8);
  equal dtype Dtype.int64 (Dtype.sum_acc_dtype Dtype.int64);
  (* Bool accumulates as int32. *)
  equal dtype Dtype.int32 (Dtype.sum_acc_dtype Dtype.bool);
  (* Floats widen to at least float32. *)
  equal dtype Dtype.float32 (Dtype.sum_acc_dtype Dtype.float16);
  equal dtype Dtype.float64 (Dtype.sum_acc_dtype Dtype.float64);
  (* Index rejected. *)
  raises_invalid_arg "sum_acc_dtype does not accept index dtype" (fun () ->
      Dtype.sum_acc_dtype Dtype.index)

(* ───── FP16 Conversion ───── *)

let test_fp16_boundaries () =
  let eq = equal (float 0.0) in
  (* Exact representable values. *)
  eq 1.0 (Dtype.float_to_fp16 1.0);
  eq (-1.0) (Dtype.float_to_fp16 (-1.0));
  eq 0.0 (Dtype.float_to_fp16 0.0);
  eq (-0.0) (Dtype.float_to_fp16 (-0.0));
  (* Max representable. *)
  eq 65504.0 (Dtype.float_to_fp16 65504.0);
  (* Overflow to infinity. *)
  eq infinity (Dtype.float_to_fp16 65520.0);
  eq neg_infinity (Dtype.float_to_fp16 (-65520.0));
  (* Underflow to zero. *)
  eq 0.0 (Dtype.float_to_fp16 1e-8);
  (* Non-finite passthrough. *)
  eq infinity (Dtype.float_to_fp16 infinity);
  eq neg_infinity (Dtype.float_to_fp16 neg_infinity);
  is_true (Float.is_nan (Dtype.float_to_fp16 Float.nan))

let test_fp16_denormal () =
  (* Smallest positive fp16 denormal: 2^-24 ≈ 5.96e-8 *)
  let x = Float.ldexp 1.0 (-24) in
  equal (float 0.0) x (Dtype.float_to_fp16 x);
  (* Largest fp16 denormal: just below 2^-14. *)
  let x = Float.ldexp 1.0 (-14) -. Float.ldexp 1.0 (-24) in
  let r = Dtype.float_to_fp16 x in
  is_true ~msg:"denormal round-trips to finite" (Float.is_finite r);
  is_true ~msg:"denormal non-zero" (r > 0.0)

(* ───── BF16 Conversion ───── *)

let test_bf16_boundaries () =
  let eq = equal (float 0.0) in
  eq 1.0 (Dtype.float_to_bf16 1.0);
  eq 0.0 (Dtype.float_to_bf16 0.0);
  (* 128.0 = 1.0 × 2^7, exactly representable. *)
  eq 128.0 (Dtype.float_to_bf16 128.0);
  (* 1234.0 needs 10 mantissa bits, rounds to 1232.0 in bf16's 7. *)
  eq 1232.0 (Dtype.float_to_bf16 1234.0);
  (* Non-finite passthrough. *)
  eq infinity (Dtype.float_to_bf16 infinity);
  eq neg_infinity (Dtype.float_to_bf16 neg_infinity);
  is_true (Float.is_nan (Dtype.float_to_bf16 Float.nan))

(* ───── FP8 Conversions ───── *)

let test_fp8_boundaries () =
  let eq = equal (float 0.0) in
  (* Zero. *)
  equal int 0 (Dtype.float_to_fp8 Fp8e4m3 0.0);
  equal int 0 (Dtype.float_to_fp8 Fp8e5m2 0.0);
  eq 0.0 (Dtype.fp8_to_float Fp8e4m3 0);
  eq 0.0 (Dtype.fp8_to_float Fp8e5m2 0);
  (* E4m3 max normal: 448.0. *)
  eq 448.0 (Dtype.fp8_to_float Fp8e4m3 (Dtype.float_to_fp8 Fp8e4m3 448.0));
  (* E4m3 is saturating: infinity → NaN, above-max → maxnorm. *)
  is_true
    (Float.is_nan
       (Dtype.fp8_to_float Fp8e4m3 (Dtype.float_to_fp8 Fp8e4m3 infinity)));
  eq 448.0 (Dtype.fp8_to_float Fp8e4m3 (Dtype.float_to_fp8 Fp8e4m3 500.0));
  (* E5m2 max normal: 57344.0. *)
  eq 57344.0 (Dtype.fp8_to_float Fp8e5m2 (Dtype.float_to_fp8 Fp8e5m2 57344.0));
  (* E5m2 is IEEE-like: infinity → infinity, NaN → NaN. *)
  eq infinity (Dtype.fp8_to_float Fp8e5m2 (Dtype.float_to_fp8 Fp8e5m2 infinity));
  is_true
    (Float.is_nan
       (Dtype.fp8_to_float Fp8e5m2 (Dtype.float_to_fp8 Fp8e5m2 Float.nan)));
  (* Invalid scalar raises. *)
  raises_match
    (function Invalid_argument _ -> true | _ -> false)
    (fun () -> Dtype.float_to_fp8 Int8 1.0);
  raises_match
    (function Invalid_argument _ -> true | _ -> false)
    (fun () -> Dtype.fp8_to_float Int8 0)

(* ───── Integer Truncation ───── *)

let test_trunc_boundaries () =
  (* In-range identity. *)
  equal int 42 (Dtype.truncate_int Dtype.int8 42);
  equal int (-1) (Dtype.truncate_int Dtype.int8 (-1));
  (* Unsigned wrap. *)
  equal int 0 (Dtype.truncate_int Dtype.uint8 256);
  equal int 255 (Dtype.truncate_int Dtype.uint8 255);
  equal int 0 (Dtype.truncate_int Dtype.uint16 65536);
  (* Signed wrap with sign extension. *)
  equal int (-128) (Dtype.truncate_int Dtype.int8 128);
  equal int (-1) (Dtype.truncate_int Dtype.int8 255);
  equal int (-1) (Dtype.truncate_int Dtype.int16 65535);
  (* Bool: 0 → 0, nonzero → 1. *)
  equal int 0 (Dtype.truncate_int Dtype.bool 0);
  equal int 1 (Dtype.truncate_int Dtype.bool 1);
  equal int 1 (Dtype.truncate_int Dtype.bool 2);
  (* Invalid dtype. *)
  raises_match
    (function Invalid_argument _ -> true | _ -> false)
    (fun () -> Dtype.truncate_int Dtype.float32 1)

(* ───── Vec Operations ───── *)

let test_vec () =
  (* Basic vectorization. *)
  let v = Dtype.vec Dtype.int32 4 in
  equal dtype { scalar = Int32; count = 4 } v;
  (* Count=1 is identity. *)
  equal dtype Dtype.int32 (Dtype.vec Dtype.int32 1);
  (* Void ignores count. *)
  equal dtype Dtype.void (Dtype.vec Dtype.void 4);
  (* index.vec(0) for empty shape vectors. *)
  equal int 0 (Dtype.vec Dtype.index 0).count;
  (* scalar_of strips count. *)
  equal dtype Dtype.int32 (Dtype.scalar_of v);
  equal dtype Dtype.float64 (Dtype.scalar_of Dtype.float64)

let test_vec_errors () =
  raises_invalid_arg "only index dtype can use zero-length vectors" (fun () ->
      Dtype.vec Dtype.int32 0);
  raises_match
    (function Invalid_argument _ -> true | _ -> false)
    (fun () -> Dtype.vec (Dtype.vec Dtype.int32 4) 2);
  raises_match
    (function Invalid_argument _ -> true | _ -> false)
    (fun () -> Dtype.vec Dtype.int32 (-1))

(* ───── Bounds ───── *)

let test_bounds () =
  (* Spot-check representative types. *)
  equal bound (`Bool false) (Dtype.min Dtype.bool);
  equal bound (`Bool true) (Dtype.max Dtype.bool);
  equal bound (`SInt (-128L)) (Dtype.min Dtype.int8);
  equal bound (`SInt 127L) (Dtype.max Dtype.int8);
  equal bound (`UInt 0L) (Dtype.min Dtype.uint8);
  equal bound (`UInt 255L) (Dtype.max Dtype.uint8);
  equal bound (`SInt Int64.min_int) (Dtype.min Dtype.int64);
  equal bound (`SInt Int64.max_int) (Dtype.max Dtype.int64);
  equal bound (`UInt Int64.minus_one) (Dtype.max Dtype.uint64);
  equal bound (`Float neg_infinity) (Dtype.min Dtype.float32);
  equal bound (`Float infinity) (Dtype.max Dtype.float64);
  (* Vec inherits scalar bounds. *)
  equal bound (`SInt (-128L)) (Dtype.min (Dtype.vec Dtype.int8 4));
  (* Void rejected. *)
  raises_invalid_arg "void has no numeric bounds" (fun () ->
      Dtype.min Dtype.void)

(* ───── Float Info ───── *)

let test_finfo () =
  equal int_pair (5, 10) (Dtype.finfo Dtype.float16);
  equal int_pair (8, 7) (Dtype.finfo Dtype.bfloat16);
  equal int_pair (8, 23) (Dtype.finfo Dtype.float32);
  equal int_pair (11, 52) (Dtype.finfo Dtype.float64);
  equal int_pair (4, 3) (Dtype.finfo Dtype.fp8e4m3);
  equal int_pair (5, 2) (Dtype.finfo Dtype.fp8e5m2);
  raises_invalid_arg "finfo expects a floating-point dtype" (fun () ->
      Dtype.finfo Dtype.int32)

(* ───── Property Tests ───── *)

(* lub is commutative: lub [a; b] = lub [b; a]. *)
let prop_promo_commutative a b =
  Dtype.equal
    (Dtype.least_upper_dtype [ a; b ])
    (Dtype.least_upper_dtype [ b; a ])

(* lub [a; a] = scalar_of a. *)
let prop_promo_idempotent a =
  Dtype.equal (Dtype.least_upper_dtype [ a; a ]) (Dtype.scalar_of a)

(* Every type casts losslessly to itself. *)
let prop_lossless_reflexive a = Dtype.can_lossless_cast a a

(* sum_acc is idempotent: widening the accumulator again is a no-op. *)
let prop_sum_acc_idempotent a =
  Dtype.equal
    (Dtype.sum_acc_dtype (Dtype.sum_acc_dtype a))
    (Dtype.sum_acc_dtype a)

(* FP16 projection is idempotent. *)
let prop_fp16_idempotent x =
  let r = Dtype.float_to_fp16 x in
  (* NaN ≠ NaN by IEEE, so check with Float.is_nan. *)
  if Float.is_nan r then Float.is_nan (Dtype.float_to_fp16 r)
  else Float.equal r (Dtype.float_to_fp16 r)

(* BF16 projection is idempotent. *)
let prop_bf16_idempotent x =
  let r = Dtype.float_to_bf16 x in
  if Float.is_nan r then Float.is_nan (Dtype.float_to_bf16 r)
  else Float.equal r (Dtype.float_to_bf16 r)

(* FP8 encode/decode round-trip is idempotent for valid bytes. *)
let fp8_byte =
  let gen = Gen.int_range 0 255 in
  testable ~pp:Format.pp_print_int ~equal:Int.equal ~gen ()

let prop_fp8_byte_round_trip byte =
  List.for_all
    (fun s ->
      let f = Dtype.fp8_to_float s byte in
      let byte' = Dtype.float_to_fp8 s f in
      let f' = Dtype.fp8_to_float s byte' in
      (Float.is_nan f && Float.is_nan f') || Float.equal f f')
    [ Fp8e4m3; Fp8e5m2 ]

(* truncate_int is idempotent. *)
let prop_trunc_idempotent (dt, x) =
  let r = Dtype.truncate_int dt x in
  r = Dtype.truncate_int dt r

(* ───── Runner ───── *)

let () =
  run "Dtype"
    [
      group "Type Promotion"
        [
          test "lattice edges" test_promo_lattice;
          test "strips vectorization" test_promo_strips_vec;
          test "errors" test_promo_errors;
          prop "commutative" (pair promotable_dtype promotable_dtype)
            (fun (a, b) -> prop_promo_commutative a b);
          prop "idempotent" promotable_dtype prop_promo_idempotent;
        ];
      group "Lossless Cast"
        [
          test "widening" test_lossless_widening;
          test "narrowing fails" test_lossless_narrowing_fails;
          test "cross-sign" test_lossless_cross_sign;
          test "to index" test_lossless_to_index;
          prop "reflexive" promotable_dtype prop_lossless_reflexive;
        ];
      group "Sum Accumulator"
        [
          test "all categories" test_sum_acc;
          prop "idempotent" promotable_dtype prop_sum_acc_idempotent;
        ];
      group "FP16 Conversion"
        [
          test "boundaries" test_fp16_boundaries;
          test "denormal range" test_fp16_denormal;
          prop "idempotent" (float 0.0) prop_fp16_idempotent;
        ];
      group "BF16 Conversion"
        [
          test "boundaries" test_bf16_boundaries;
          prop "idempotent" (float 0.0) prop_bf16_idempotent;
        ];
      group "FP8 Conversion"
        [
          test "boundaries" test_fp8_boundaries;
          prop "byte round-trip stable" fp8_byte prop_fp8_byte_round_trip;
        ];
      group "Integer Truncation"
        [
          test "boundaries" test_trunc_boundaries;
          prop "idempotent" (pair int_dtype int) prop_trunc_idempotent;
        ];
      group "Vec" [ test "operations" test_vec; test "errors" test_vec_errors ];
      group "Bounds" [ test "spot checks" test_bounds ];
      group "Float Info" [ test "all types" test_finfo ];
    ]
