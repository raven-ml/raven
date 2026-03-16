(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Tolk_ir

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

let raises_invalid (f : unit -> _) =
  raises_match (function Invalid_argument _ -> true | _ -> false) f

(* Dtypes that participate in promotion (excludes Void and Index). *)
let promotable_dtypes =
  Dtype.
    [
      bool; int8; int16; int32; int64; uint8; uint16; uint32; uint64; float16;
      bfloat16; float32; float64; fp8e4m3; fp8e5m2;
    ]

let promotable_dtype =
  let gen = Gen.oneofl promotable_dtypes in
  testable ~pp:Dtype.pp ~equal:Dtype.equal ~gen ()

(* Integer dtypes suitable for truncate_int (excludes Index). *)
let int_dtypes = Dtype.[ bool; int8; int16; int32; uint8; uint16; uint32 ]

let int_dtype =
  let gen = Gen.oneofl int_dtypes in
  testable ~pp:Dtype.pp ~equal:Dtype.equal ~gen ()

let fp8_byte =
  let gen = Gen.int_range 0 255 in
  testable ~pp:Format.pp_print_int ~equal:Int.equal ~gen ()

let lub = Dtype.least_upper_dtype

let () =
  run "Dtype"
    [
      group "Type Promotion"
        [
          test "lattice edges" (fun () ->
            equal dtype Dtype.int8 (lub [ Dtype.bool; Dtype.int8 ]);
            equal dtype Dtype.int16 (lub [ Dtype.int8; Dtype.uint8 ]);
            equal dtype Dtype.int32 (lub [ Dtype.int16; Dtype.uint16 ]);
            equal dtype Dtype.int64 (lub [ Dtype.int32; Dtype.uint32 ]);
            (* Cross-category: int through float. *)
            equal dtype Dtype.float16 (lub [ Dtype.float16; Dtype.int64 ]);
            (* FP8 siblings meet at float16. *)
            equal dtype Dtype.float16 (lub [ Dtype.fp8e4m3; Dtype.fp8e5m2 ]);
            (* Float16 and bfloat16 are incomparable; they meet at float32. *)
            equal dtype Dtype.float32 (lub [ Dtype.float16; Dtype.bfloat16 ]));
          test "strips vectorization" (fun () ->
            let vec4 = Dtype.vec Dtype.int8 4 in
            equal dtype Dtype.int16 (lub [ vec4; Dtype.uint8 ]));
          test "errors" (fun () ->
            raises_invalid_arg "least_upper_dtype requires at least one dtype"
              (fun () -> lub []);
            raises_invalid_arg "Index does not participate in dtype promotion"
              (fun () -> lub [ Dtype.index ]));
          prop2 "commutative" promotable_dtype promotable_dtype (fun a b ->
            Dtype.equal (lub [ a; b ]) (lub [ b; a ]));
          prop "idempotent" promotable_dtype (fun a ->
            Dtype.equal (lub [ a; a ]) (Dtype.scalar_of a));
        ];
      group "Lossless Cast"
        [
          test "widening" (fun () ->
            is_true (Dtype.can_lossless_cast Dtype.int8 Dtype.int16);
            is_true (Dtype.can_lossless_cast Dtype.int16 Dtype.int32);
            is_true (Dtype.can_lossless_cast Dtype.uint8 Dtype.uint16);
            is_true (Dtype.can_lossless_cast Dtype.float16 Dtype.float32);
            is_true (Dtype.can_lossless_cast Dtype.float32 Dtype.float64);
            is_true (Dtype.can_lossless_cast Dtype.fp8e4m3 Dtype.float16);
            is_true (Dtype.can_lossless_cast Dtype.fp8e5m2 Dtype.float16));
          test "narrowing fails" (fun () ->
            is_false (Dtype.can_lossless_cast Dtype.int32 Dtype.int16);
            is_false (Dtype.can_lossless_cast Dtype.float64 Dtype.float32);
            is_false (Dtype.can_lossless_cast Dtype.float16 Dtype.fp8e4m3));
          test "cross-sign" (fun () ->
            (* uint8 fits in int16 (wider signed). *)
            is_true (Dtype.can_lossless_cast Dtype.uint8 Dtype.int16);
            (* int8 doesn't fit in uint8 (loses negatives). *)
            is_false (Dtype.can_lossless_cast Dtype.int8 Dtype.uint8);
            is_false (Dtype.can_lossless_cast Dtype.int16 Dtype.uint16));
          test "to index" (fun () ->
            is_true (Dtype.can_lossless_cast Dtype.int32 Dtype.index);
            is_true (Dtype.can_lossless_cast Dtype.uint64 Dtype.index);
            is_false (Dtype.can_lossless_cast Dtype.float32 Dtype.index));
          prop "reflexive" promotable_dtype (fun a ->
            Dtype.can_lossless_cast a a);
        ];
      group "Sum Accumulator"
        [
          test "all categories" (fun () ->
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
            raises_invalid_arg "sum_acc_dtype does not accept index dtype"
              (fun () -> Dtype.sum_acc_dtype Dtype.index));
          prop "idempotent" promotable_dtype (fun a ->
            Dtype.equal
              (Dtype.sum_acc_dtype (Dtype.sum_acc_dtype a))
              (Dtype.sum_acc_dtype a));
        ];
      group "FP16 Conversion"
        [
          test "boundaries" (fun () ->
            let eq = equal (float 0.0) in
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
            is_true (Float.is_nan (Dtype.float_to_fp16 Float.nan)));
          test "denormal range" (fun () ->
            (* Smallest positive fp16 denormal: 2^-24 *)
            let x = Float.ldexp 1.0 (-24) in
            equal (float 0.0) x (Dtype.float_to_fp16 x);
            (* Largest fp16 denormal: just below 2^-14. *)
            let x = Float.ldexp 1.0 (-14) -. Float.ldexp 1.0 (-24) in
            let r = Dtype.float_to_fp16 x in
            is_true ~msg:"denormal round-trips to finite" (Float.is_finite r);
            is_true ~msg:"denormal non-zero" (r > 0.0));
          prop "idempotent" (float 0.0) (fun x ->
            let r = Dtype.float_to_fp16 x in
            if Float.is_nan r then Float.is_nan (Dtype.float_to_fp16 r)
            else Float.equal r (Dtype.float_to_fp16 r));
        ];
      group "BF16 Conversion"
        [
          test "boundaries" (fun () ->
            let eq = equal (float 0.0) in
            eq 1.0 (Dtype.float_to_bf16 1.0);
            eq 0.0 (Dtype.float_to_bf16 0.0);
            (* 128.0 = 1.0 * 2^7, exactly representable. *)
            eq 128.0 (Dtype.float_to_bf16 128.0);
            (* 1234.0 needs 10 mantissa bits, rounds to 1232.0 in bf16's 7. *)
            eq 1232.0 (Dtype.float_to_bf16 1234.0);
            (* Non-finite passthrough. *)
            eq infinity (Dtype.float_to_bf16 infinity);
            eq neg_infinity (Dtype.float_to_bf16 neg_infinity);
            is_true (Float.is_nan (Dtype.float_to_bf16 Float.nan)));
          prop "idempotent" (float 0.0) (fun x ->
            let r = Dtype.float_to_bf16 x in
            if Float.is_nan r then Float.is_nan (Dtype.float_to_bf16 r)
            else Float.equal r (Dtype.float_to_bf16 r));
        ];
      group "FP8 Conversion"
        [
          test "boundaries" (fun () ->
            let eq = equal (float 0.0) in
            equal int 0 (Dtype.float_to_fp8 Fp8e4m3 0.0);
            equal int 0 (Dtype.float_to_fp8 Fp8e5m2 0.0);
            eq 0.0 (Dtype.fp8_to_float Fp8e4m3 0);
            eq 0.0 (Dtype.fp8_to_float Fp8e5m2 0);
            (* E4m3 max normal: 448.0. *)
            eq 448.0
              (Dtype.fp8_to_float Fp8e4m3
                 (Dtype.float_to_fp8 Fp8e4m3 448.0));
            (* E4m3 is saturating: infinity -> NaN, above-max -> maxnorm. *)
            is_true
              (Float.is_nan
                 (Dtype.fp8_to_float Fp8e4m3
                    (Dtype.float_to_fp8 Fp8e4m3 infinity)));
            eq 448.0
              (Dtype.fp8_to_float Fp8e4m3
                 (Dtype.float_to_fp8 Fp8e4m3 500.0));
            (* E5m2 max normal: 57344.0. *)
            eq 57344.0
              (Dtype.fp8_to_float Fp8e5m2
                 (Dtype.float_to_fp8 Fp8e5m2 57344.0));
            (* E5m2 is IEEE-like: infinity -> infinity, NaN -> NaN. *)
            eq infinity
              (Dtype.fp8_to_float Fp8e5m2
                 (Dtype.float_to_fp8 Fp8e5m2 infinity));
            is_true
              (Float.is_nan
                 (Dtype.fp8_to_float Fp8e5m2
                    (Dtype.float_to_fp8 Fp8e5m2 Float.nan)));
            raises_invalid (fun () -> Dtype.float_to_fp8 Int8 1.0);
            raises_invalid (fun () -> Dtype.fp8_to_float Int8 0));
          prop "byte round-trip stable" fp8_byte (fun byte ->
            List.for_all
              (fun s ->
                let f = Dtype.fp8_to_float s byte in
                let byte' = Dtype.float_to_fp8 s f in
                let f' = Dtype.fp8_to_float s byte' in
                (Float.is_nan f && Float.is_nan f') || Float.equal f f')
              [ Fp8e4m3; Fp8e5m2 ]);
        ];
      group "Integer Truncation"
        [
          test "boundaries" (fun () ->
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
            (* Bool: 0 -> 0, nonzero -> 1. *)
            equal int 0 (Dtype.truncate_int Dtype.bool 0);
            equal int 1 (Dtype.truncate_int Dtype.bool 1);
            equal int 1 (Dtype.truncate_int Dtype.bool 2);
            raises_invalid (fun () -> Dtype.truncate_int Dtype.float32 1));
          prop "idempotent" (pair int_dtype int) (fun (dt, x) ->
            let r = Dtype.truncate_int dt x in
            r = Dtype.truncate_int dt r);
        ];
      group "Vec"
        [
          test "operations" (fun () ->
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
            equal dtype Dtype.float64 (Dtype.scalar_of Dtype.float64));
          test "errors" (fun () ->
            raises_invalid_arg
              "only index dtype can use zero-length vectors" (fun () ->
                Dtype.vec Dtype.int32 0);
            raises_invalid (fun () -> Dtype.vec (Dtype.vec Dtype.int32 4) 2);
            raises_invalid (fun () -> Dtype.vec Dtype.int32 (-1)));
        ];
      group "Bounds"
        [
          test "spot checks" (fun () ->
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
            raises_invalid_arg "void has no numeric bounds" (fun () ->
              Dtype.min Dtype.void));
        ];
      group "Float Info"
        [
          test "all types" (fun () ->
            equal int_pair (5, 10) (Dtype.finfo Dtype.float16);
            equal int_pair (8, 7) (Dtype.finfo Dtype.bfloat16);
            equal int_pair (8, 23) (Dtype.finfo Dtype.float32);
            equal int_pair (11, 52) (Dtype.finfo Dtype.float64);
            equal int_pair (4, 3) (Dtype.finfo Dtype.fp8e4m3);
            equal int_pair (5, 2) (Dtype.finfo Dtype.fp8e5m2);
            raises_invalid_arg "finfo expects a floating-point dtype" (fun () ->
              Dtype.finfo Dtype.int32));
        ];
    ]
