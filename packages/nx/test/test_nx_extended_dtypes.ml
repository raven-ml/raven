(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for extended bigarray dtypes *)

open Windtrap
open Test_nx_support

(* ───── BFloat16 Tests ───── *)

let test_create_bfloat16 () =
  let t = Nx.create Nx_core.Dtype.bfloat16 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  check_t ~eps:0.01 "create bfloat16" [| 3 |] [| 1.0; 2.0; 3.0 |] t

let test_scalar_bfloat16 () =
  let t = Nx.scalar Nx_core.Dtype.bfloat16 42.5 in
  check_t ~eps:0.01 "scalar bfloat16" [||] [| 42.5 |] t

let test_zeros_bfloat16 () =
  let t = Nx.zeros Nx_core.Dtype.bfloat16 [| 2; 2 |] in
  check_t ~eps:0.01 "zeros bfloat16" [| 2; 2 |] [| 0.0; 0.0; 0.0; 0.0 |] t

let test_ones_bfloat16 () =
  let t = Nx.ones Nx_core.Dtype.bfloat16 [| 2; 2 |] in
  check_t ~eps:0.01 "ones bfloat16" [| 2; 2 |] [| 1.0; 1.0; 1.0; 1.0 |] t

let test_arange_bfloat16 () =
  let t = Nx.arange Nx_core.Dtype.bfloat16 0 5 1 in
  check_t ~eps:0.01 "arange bfloat16" [| 5 |] [| 0.0; 1.0; 2.0; 3.0; 4.0 |] t

(* ───── Bool Tests ───── *)

let test_create_bool () =
  let t = Nx.create Nx_core.Dtype.bool [| 4 |] [| false; true; false; true |] in
  check_t "create bool" [| 4 |] [| false; true; false; true |] t

let test_scalar_bool () =
  let t = Nx.scalar Nx_core.Dtype.bool true in
  check_t "scalar bool" [||] [| true |] t

let test_zeros_bool () =
  let t = Nx.zeros Nx_core.Dtype.bool [| 2; 2 |] in
  check_t "zeros bool" [| 2; 2 |] [| false; false; false; false |] t

let test_ones_bool () =
  let t = Nx.ones Nx_core.Dtype.bool [| 2; 2 |] in
  check_t "ones bool" [| 2; 2 |] [| true; true; true; true |] t

(* ───── Int4 Tests ───── *)

let test_create_int4 () =
  let t = Nx.create Nx_core.Dtype.int4 [| 4 |] [| -8; -1; 0; 7 |] in
  check_t "create int4" [| 4 |] [| -8; -1; 0; 7 |] t

let test_scalar_int4 () =
  let t = Nx.scalar Nx_core.Dtype.int4 5 in
  check_t "scalar int4" [||] [| 5 |] t

let test_zeros_int4 () =
  let t = Nx.zeros Nx_core.Dtype.int4 [| 2; 2 |] in
  check_t "zeros int4" [| 2; 2 |] [| 0; 0; 0; 0 |] t

let test_ones_int4 () =
  let t = Nx.ones Nx_core.Dtype.int4 [| 2; 2 |] in
  check_t "ones int4" [| 2; 2 |] [| 1; 1; 1; 1 |] t

let test_arange_int4 () =
  let t = Nx.arange Nx_core.Dtype.int4 (-3) 4 1 in
  check_t "arange int4" [| 7 |] [| -3; -2; -1; 0; 1; 2; 3 |] t

(* ───── UInt4 Tests ───── *)

let test_create_uint4 () =
  let t = Nx.create Nx_core.Dtype.uint4 [| 4 |] [| 0; 5; 10; 15 |] in
  check_t "create uint4" [| 4 |] [| 0; 5; 10; 15 |] t

let test_scalar_uint4 () =
  let t = Nx.scalar Nx_core.Dtype.uint4 12 in
  check_t "scalar uint4" [||] [| 12 |] t

let test_zeros_uint4 () =
  let t = Nx.zeros Nx_core.Dtype.uint4 [| 2; 2 |] in
  check_t "zeros uint4" [| 2; 2 |] [| 0; 0; 0; 0 |] t

let test_ones_uint4 () =
  let t = Nx.ones Nx_core.Dtype.uint4 [| 2; 2 |] in
  check_t "ones uint4" [| 2; 2 |] [| 1; 1; 1; 1 |] t

let test_arange_uint4 () =
  let t = Nx.arange Nx_core.Dtype.uint4 0 8 2 in
  check_t "arange uint4" [| 4 |] [| 0; 2; 4; 6 |] t

(* ───── UInt32 Tests ───── *)

let test_create_uint32 () =
  let t = Nx.create Nx_core.Dtype.uint32 [| 3 |] [| 0l; 1l; 42l |] in
  check_t "create uint32" [| 3 |] [| 0l; 1l; 42l |] t

let test_scalar_uint32 () =
  let t = Nx.scalar Nx_core.Dtype.uint32 7l in
  check_t "scalar uint32" [||] [| 7l |] t

let test_zeros_uint32 () =
  let t = Nx.zeros Nx_core.Dtype.uint32 [| 2; 2 |] in
  check_t "zeros uint32" [| 2; 2 |] [| 0l; 0l; 0l; 0l |] t

let test_ones_uint32 () =
  let t = Nx.ones Nx_core.Dtype.uint32 [| 2; 2 |] in
  check_t "ones uint32" [| 2; 2 |] [| 1l; 1l; 1l; 1l |] t

(* ───── UInt64 Tests ───── *)

let test_create_uint64 () =
  let t = Nx.create Nx_core.Dtype.uint64 [| 3 |] [| 0L; 1L; 42L |] in
  check_t "create uint64" [| 3 |] [| 0L; 1L; 42L |] t

let test_scalar_uint64 () =
  let t = Nx.scalar Nx_core.Dtype.uint64 7L in
  check_t "scalar uint64" [||] [| 7L |] t

let test_zeros_uint64 () =
  let t = Nx.zeros Nx_core.Dtype.uint64 [| 2; 2 |] in
  check_t "zeros uint64" [| 2; 2 |] [| 0L; 0L; 0L; 0L |] t

let test_ones_uint64 () =
  let t = Nx.ones Nx_core.Dtype.uint64 [| 2; 2 |] in
  check_t "ones uint64" [| 2; 2 |] [| 1L; 1L; 1L; 1L |] t

(* ───── Float8_e4m3 Tests ───── *)

let test_create_float8_e4m3 () =
  let t = Nx.create Nx_core.Dtype.float8_e4m3 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  check_t ~eps:0.1 "create float8_e4m3" [| 3 |] [| 1.0; 2.0; 3.0 |] t

let test_scalar_float8_e4m3 () =
  (* Test with a value that can be exactly represented in Float8_e4m3. With a
     3-bit mantissa, we can represent 1.000 through 1.111 in binary. For
     example: 11.0 = 1.011 × 2^3 is exactly representable. *)
  let t = Nx.scalar Nx_core.Dtype.float8_e4m3 11.0 in
  check_t ~eps:0.1 "scalar float8_e4m3" [||] [| 11.0 |] t

let test_zeros_float8_e4m3 () =
  let t = Nx.zeros Nx_core.Dtype.float8_e4m3 [| 2; 2 |] in
  check_t ~eps:0.1 "zeros float8_e4m3" [| 2; 2 |] [| 0.0; 0.0; 0.0; 0.0 |] t

let test_ones_float8_e4m3 () =
  let t = Nx.ones Nx_core.Dtype.float8_e4m3 [| 2; 2 |] in
  check_t ~eps:0.1 "ones float8_e4m3" [| 2; 2 |] [| 1.0; 1.0; 1.0; 1.0 |] t

(* ───── Float8_e5m2 Tests ───── *)

let test_create_float8_e5m2 () =
  let t = Nx.create Nx_core.Dtype.float8_e5m2 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  check_t ~eps:0.1 "create float8_e5m2" [| 3 |] [| 1.0; 2.0; 3.0 |] t

let test_scalar_float8_e5m2 () =
  let t = Nx.scalar Nx_core.Dtype.float8_e5m2 20.0 in
  check_t ~eps:0.1 "scalar float8_e5m2" [||] [| 20.0 |] t

let test_zeros_float8_e5m2 () =
  let t = Nx.zeros Nx_core.Dtype.float8_e5m2 [| 2; 2 |] in
  check_t ~eps:0.1 "zeros float8_e5m2" [| 2; 2 |] [| 0.0; 0.0; 0.0; 0.0 |] t

let test_ones_float8_e5m2 () =
  let t = Nx.ones Nx_core.Dtype.float8_e5m2 [| 2; 2 |] in
  check_t ~eps:0.1 "ones float8_e5m2" [| 2; 2 |] [| 1.0; 1.0; 1.0; 1.0 |] t

(* ───── Half-precision (float16 / bfloat16) numerics ───── *)

(* Quantize a float through a half dtype's cast kernel and read it back at
   float32: exercises op_cast in both directions, not just buffer set/get. *)
let via_cast (type b) (dt : (float, b) Nx.dtype) v =
  Nx.item [] (Nx.cast Nx.float32 (Nx.cast dt (Nx.scalar Nx.float32 v)))

(* Quantize through element storage (create writes half bits directly). *)
let via_store (type b) (dt : (float, b) Nx.dtype) v =
  Nx.item [] (Nx.scalar dt v)

let check_quantize name dt cases =
  List.iter
    (fun (v, expected) ->
      equal
        ~msg:(Printf.sprintf "%s cast %.17g" name v)
        (float 0.0) expected (via_cast dt v);
      equal
        ~msg:(Printf.sprintf "%s store %.17g" name v)
        (float 0.0) expected (via_store dt v))
    cases

(* bfloat16: 7 stored mantissa bits, step 2^-7 near 1.0. Round to nearest,
   ties to even mantissa. *)
let test_bfloat16_rne () =
  check_quantize "bfloat16" Nx_core.Dtype.bfloat16
    [
      (1.0, 1.0);
      (* exactly representable *)
      (1.0078125, 1.0078125);
      (* 1 + 2^-8: tie between 1.0 (even) and 1.0078125 (odd) -> 1.0 *)
      (1.00390625, 1.0);
      (* 1 + 3*2^-8: tie between 1.0078125 (odd) and 1.015625 (even) *)
      (1.01171875, 1.015625);
      (* above the tie: rounds up *)
      (1.004, 1.0078125);
      (* 257 ties between 256 (even) and 258 (odd) -> 256 *)
      (257.0, 256.0);
      (258.0, 258.0);
      (-257.0, -256.0);
      (0.0, 0.0);
    ]

(* float16: 10 stored mantissa bits, step 2^-10 near 1.0. *)
let test_float16_rne () =
  check_quantize "float16" Nx_core.Dtype.float16
    [
      (1.0, 1.0);
      (1.0009765625, 1.0009765625);
      (* 1 + 2^-11: tie between 1.0 (even) and 1 + 2^-10 (odd) -> 1.0 *)
      (1.00048828125, 1.0);
      (* 1 + 3*2^-11: tie between 1+2^-10 (odd) and 1+2^-9 (even) *)
      (1.00146484375, 1.001953125);
      (* 2049 ties between 2048 (even) and 2050 (odd) -> 2048 *)
      (2049.0, 2048.0);
      (2050.0, 2050.0);
      (-2049.0, -2048.0);
      (65504.0, 65504.0);
      (* largest finite float16 *)
    ]

(* float16 subnormals: min normal 2^-14, subnormal step 2^-24. Casting small
   float32 values must produce subnormals, not flush to zero. *)
let test_float16_subnormals () =
  check_quantize "float16 subnormal" Nx_core.Dtype.float16
    [
      (Float.ldexp 1.0 (-14), Float.ldexp 1.0 (-14));
      (* min normal *)
      (Float.ldexp 1.0 (-24), Float.ldexp 1.0 (-24));
      (* smallest subnormal survives *)
      (Float.ldexp 3.0 (-24), Float.ldexp 3.0 (-24));
      (* exact subnormal *)
      (Float.ldexp 1.0 (-25), 0.0);
      (* tie between 0 (even) and 2^-24 -> 0 *)
      (Float.ldexp 3.0 (-26), Float.ldexp 1.0 (-24));
      (* 1.5 * 2^-25 rounds to the nearest subnormal *)
      (Float.ldexp 1.0 (-26), 0.0);
      (* below the tie: underflows *)
    ]

let test_float16_overflow_to_inf () =
  equal ~msg:"float16 overflow" (float 0.0) Float.infinity
    (via_cast Nx_core.Dtype.float16 65536.0);
  equal ~msg:"float16 negative overflow" (float 0.0) Float.neg_infinity
    (via_cast Nx_core.Dtype.float16 (-65536.0))

let test_half_cast_roundtrip (type b) name (dt : (float, b) Nx.dtype) () =
  (* Values exactly representable at the half dtype survive
     f32 -> half -> f32 unchanged. *)
  let exact = [| 0.0; -0.5; 1.0; 1.5; -2.0; 0.25; 384.0; -0.09375 |] in
  let t32 = Nx.create Nx.float32 [| 8 |] exact in
  let back = Nx.cast Nx.float32 (Nx.cast dt t32) in
  check_data ~eps:0.0 (name ^ " exact roundtrip") exact back;
  (* Casting the quantized value again is a fixed point. *)
  let once = Nx.cast Nx.float32 (Nx.cast dt (Nx.scalar Nx.float32 0.1)) in
  let twice = Nx.cast Nx.float32 (Nx.cast dt once) in
  equal
    ~msg:(name ^ " quantization is idempotent")
    (float 0.0) (Nx.item [] once) (Nx.item [] twice);
  (* 0.1 is not representable: the roundtrip must move it, but only within
     one unit in the last place. *)
  let err = Float.abs (Nx.item [] once -. 0.1) in
  is_true ~msg:(name ^ " 0.1 is inexact") (err > 0.0);
  is_true ~msg:(name ^ " 0.1 error within 1 ulp") (err <= 0.1 /. 128.0)

let test_half_special_values (type b) name (dt : (float, b) Nx.dtype) () =
  let specials = [| Float.infinity; Float.neg_infinity; Float.nan |] in
  let t = Nx.cast dt (Nx.create Nx.float32 [| 3 |] specials) in
  let back = Nx.to_array (Nx.cast Nx.float32 t) in
  equal ~msg:(name ^ " +inf") (float 0.0) Float.infinity back.(0);
  equal ~msg:(name ^ " -inf") (float 0.0) Float.neg_infinity back.(1);
  is_true ~msg:(name ^ " nan") (Float.is_nan back.(2))

(* Binary ops compute wide and round the result back to the half dtype. *)
let test_half_binary_rounding () =
  (* bfloat16: 256 + 1 ties between 256 and 258 -> 256. *)
  let bf16 = Nx_core.Dtype.bfloat16 in
  let r =
    Nx.add (Nx.scalar bf16 256.0) (Nx.scalar bf16 1.0)
  in
  equal ~msg:"bfloat16 256+1" (float 0.0) 256.0 (Nx.item [] r);
  (* float16: 2048 + 1 ties between 2048 and 2050 -> 2048. *)
  let f16 = Nx_core.Dtype.float16 in
  let r =
    Nx.add (Nx.scalar f16 2048.0) (Nx.scalar f16 1.0)
  in
  equal ~msg:"float16 2048+1" (float 0.0) 2048.0 (Nx.item [] r);
  (* mul: (1 + 2^-7)^2 = 1 + 2^-6 + 2^-14 rounds to 1 + 2^-6 at bfloat16. *)
  let x = Nx.scalar bf16 1.0078125 in
  equal ~msg:"bfloat16 (1+2^-7)^2" (float 0.0) 1.015625
    (Nx.item [] (Nx.mul x x));
  (* mul: (1 + 2^-10)^2 rounds to 1 + 2^-9 at float16. *)
  let x = Nx.scalar f16 1.0009765625 in
  equal ~msg:"float16 (1+2^-10)^2" (float 0.0) 1.001953125
    (Nx.item [] (Nx.mul x x))

let test_half_binary_exact (type b) name (dt : (float, b) Nx.dtype) () =
  let a = Nx.create dt [| 4 |] [| 1.5; -2.0; 0.25; 3.0 |] in
  let b = Nx.create dt [| 4 |] [| 0.5; 0.5; -0.5; 2.0 |] in
  check_data ~eps:0.0 (name ^ " add") [| 2.0; -1.5; -0.25; 5.0 |] (Nx.add a b);
  check_data ~eps:0.0 (name ^ " sub") [| 1.0; -2.5; 0.75; 1.0 |] (Nx.sub a b);
  check_data ~eps:0.0 (name ^ " mul")
    [| 0.75; -1.0; -0.125; 6.0 |]
    (Nx.mul a b);
  check_data ~eps:0.0 (name ^ " div") [| 3.0; -4.0; -0.5; 1.5 |] (Nx.div a b)

(* Reductions accumulate wider than the half dtype: summing 4096 ones at
   float16 gives exactly 4096 (a naive float16 accumulator stalls at 2048
   because 2048 + 1 rounds back to 2048); same construction for bfloat16. *)
let test_half_sum_accumulates_wide () =
  let f16 = Nx_core.Dtype.float16 in
  let s = Nx.sum (Nx.ones f16 [| 4096 |]) in
  equal ~msg:"float16 sum of 4096 ones" (float 0.0) 4096.0 (Nx.item [] s);
  let bf16 = Nx_core.Dtype.bfloat16 in
  let s = Nx.sum (Nx.ones bf16 [| 1024 |]) in
  equal ~msg:"bfloat16 sum of 1024 ones" (float 0.0) 1024.0 (Nx.item [] s)

let test_half_reductions (type b) name (dt : (float, b) Nx.dtype) ~mean () =
  let t =
    Nx.create dt [| 2; 3 |] [| 1.0; -2.0; 3.5; 0.5; 4.0; -1.5 |]
  in
  check_t ~eps:0.0 (name ^ " sum axis 0") [| 3 |] [| 1.5; 2.0; 2.0 |]
    (Nx.sum ~axes:[ 0 ] t);
  check_t ~eps:0.0 (name ^ " sum axis 1") [| 2 |] [| 2.5; 3.0 |]
    (Nx.sum ~axes:[ 1 ] t);
  check_t ~eps:0.0 (name ^ " max") [||] [| 4.0 |] (Nx.max t);
  check_t ~eps:0.0
    (name ^ " max axis 1")
    [| 2 |] [| 3.5; 4.0 |]
    (Nx.max ~axes:[ 1 ] t);
  (* 5.5 / 6 = 0.91666..., rounded to the dtype's precision. *)
  check_t ~eps:0.0 (name ^ " mean") [||] [| mean |] (Nx.mean t)

let test_half_matmul (type b) name (dt : (float, b) Nx.dtype) ~eps () =
  let k = 16 in
  let a32 =
    Nx.create Nx.float32 [| 4; k |]
      (Array.init (4 * k) (fun i -> sin (float_of_int i)))
  in
  let b32 =
    Nx.create Nx.float32 [| k; 3 |]
      (Array.init (k * 3) (fun i -> cos (float_of_int i)))
  in
  let ah = Nx.cast dt a32 and bh = Nx.cast dt b32 in
  (* Reference: the exact product of the quantized inputs at float32. The
     half matmul may differ only by the final rounding of each output. *)
  let reference =
    Nx.matmul (Nx.cast Nx.float32 ah) (Nx.cast Nx.float32 bh)
  in
  let out = Nx.cast Nx.float32 (Nx.matmul ah bh) in
  check_data ~eps (name ^ " matmul vs f32 reference")
    (Nx.to_array reference) out

let test_half_compare_where (type b) name (dt : (float, b) Nx.dtype) () =
  let a = Nx.create dt [| 4 |] [| 1.0; -2.0; 3.0; 0.5 |] in
  let b = Nx.create dt [| 4 |] [| 0.5; -1.0; 3.0; 2.0 |] in
  check_data
    (name ^ " cmpgt")
    [| true; false; false; false |]
    (Nx.cmpgt a b);
  check_data (name ^ " equal") [| false; false; true; false |] (Nx.equal a b);
  check_data ~eps:0.0 (name ^ " where")
    [| 1.0; -1.0; 3.0; 2.0 |]
    (Nx.where (Nx.cmpgt a b) a b);
  check_data ~eps:0.0
    (name ^ " maximum")
    [| 1.0; -1.0; 3.0; 2.0 |]
    (Nx.maximum a b);
  (* Values that quantize to the same half bits compare equal. *)
  let x = Nx.scalar dt 1.0 and y = Nx.scalar dt 1.0001 in
  equal ~msg:(name ^ " quantized equality") bool true
    (Nx.item [] (Nx.equal x y))

(* ───── Dtype Property Tests ───── *)

let test_dtype_properties () =
  (* Test is_float *)
  equal ~msg:"bfloat16 is_float" bool true
    (Nx_core.Dtype.is_float Nx_core.Dtype.bfloat16);
  equal ~msg:"float8_e4m3 is_float" bool true
    (Nx_core.Dtype.is_float Nx_core.Dtype.float8_e4m3);
  equal ~msg:"float8_e5m2 is_float" bool true
    (Nx_core.Dtype.is_float Nx_core.Dtype.float8_e5m2);
  equal ~msg:"bool is_float" bool false
    (Nx_core.Dtype.is_float Nx_core.Dtype.bool);

  (* Test is_complex *)
  equal ~msg:"complex64 is_complex" bool true
    (Nx_core.Dtype.is_complex Nx_core.Dtype.complex64);
  equal ~msg:"complex128 is_complex" bool true
    (Nx_core.Dtype.is_complex Nx_core.Dtype.complex128);
  equal ~msg:"bfloat16 is_complex" bool false
    (Nx_core.Dtype.is_complex Nx_core.Dtype.bfloat16);

  (* Test is_int *)
  equal ~msg:"int4 is_int" bool true (Nx_core.Dtype.is_int Nx_core.Dtype.int4);
  equal ~msg:"uint4 is_int" bool true (Nx_core.Dtype.is_int Nx_core.Dtype.uint4);
  equal ~msg:"uint32 is_int" bool true
    (Nx_core.Dtype.is_int Nx_core.Dtype.uint32);
  equal ~msg:"uint64 is_int" bool true
    (Nx_core.Dtype.is_int Nx_core.Dtype.uint64);
  equal ~msg:"bool is_int" bool false (Nx_core.Dtype.is_int Nx_core.Dtype.bool);

  (* Test is_uint *)
  equal ~msg:"uint4 is_uint" bool true
    (Nx_core.Dtype.is_uint Nx_core.Dtype.uint4);
  equal ~msg:"uint32 is_uint" bool true
    (Nx_core.Dtype.is_uint Nx_core.Dtype.uint32);
  equal ~msg:"uint64 is_uint" bool true
    (Nx_core.Dtype.is_uint Nx_core.Dtype.uint64);
  equal ~msg:"int4 is_uint" bool false
    (Nx_core.Dtype.is_uint Nx_core.Dtype.int4);

  (* Test itemsize *)
  equal ~msg:"bfloat16 itemsize" int 2
    (Nx_core.Dtype.itemsize Nx_core.Dtype.bfloat16);
  equal ~msg:"bool itemsize" int 1 (Nx_core.Dtype.itemsize Nx_core.Dtype.bool);
  equal ~msg:"int4 itemsize" int 1 (Nx_core.Dtype.itemsize Nx_core.Dtype.int4);
  equal ~msg:"uint4 itemsize" int 1 (Nx_core.Dtype.itemsize Nx_core.Dtype.uint4);
  equal ~msg:"float8_e4m3 itemsize" int 1
    (Nx_core.Dtype.itemsize Nx_core.Dtype.float8_e4m3);
  equal ~msg:"float8_e5m2 itemsize" int 1
    (Nx_core.Dtype.itemsize Nx_core.Dtype.float8_e5m2);
  equal ~msg:"uint32 itemsize" int 4
    (Nx_core.Dtype.itemsize Nx_core.Dtype.uint32);
  equal ~msg:"uint64 itemsize" int 8
    (Nx_core.Dtype.itemsize Nx_core.Dtype.uint64);
  equal ~msg:"complex64 itemsize" int 8
    (Nx_core.Dtype.itemsize Nx_core.Dtype.complex64);
  equal ~msg:"complex128 itemsize" int 16
    (Nx_core.Dtype.itemsize Nx_core.Dtype.complex128);

  (* Test to_string *)
  equal ~msg:"bfloat16 to_string" string "bfloat16"
    (Nx_core.Dtype.to_string Nx_core.Dtype.bfloat16);
  equal ~msg:"bool to_string" string "bool"
    (Nx_core.Dtype.to_string Nx_core.Dtype.bool);
  equal ~msg:"int4 to_string" string "int4"
    (Nx_core.Dtype.to_string Nx_core.Dtype.int4);
  equal ~msg:"uint4 to_string" string "uint4"
    (Nx_core.Dtype.to_string Nx_core.Dtype.uint4);
  equal ~msg:"float8_e4m3 to_string" string "float8_e4m3"
    (Nx_core.Dtype.to_string Nx_core.Dtype.float8_e4m3);
  equal ~msg:"float8_e5m2 to_string" string "float8_e5m2"
    (Nx_core.Dtype.to_string Nx_core.Dtype.float8_e5m2);
  equal ~msg:"uint32 to_string" string "uint32"
    (Nx_core.Dtype.to_string Nx_core.Dtype.uint32);
  equal ~msg:"uint64 to_string" string "uint64"
    (Nx_core.Dtype.to_string Nx_core.Dtype.uint64);
  equal ~msg:"complex64 to_string" string "complex64"
    (Nx_core.Dtype.to_string Nx_core.Dtype.complex64);
  equal ~msg:"complex128 to_string" string "complex128"
    (Nx_core.Dtype.to_string Nx_core.Dtype.complex128)

let test_dtype_min_max_values () =
  (* Test min_value *)
  equal ~msg:"int4 min_value" int (-8)
    (Nx_core.Dtype.min_value Nx_core.Dtype.int4);
  equal ~msg:"uint4 min_value" int 0
    (Nx_core.Dtype.min_value Nx_core.Dtype.uint4);
  equal ~msg:"bool min_value" bool false
    (Nx_core.Dtype.min_value Nx_core.Dtype.bool);
  equal ~msg:"uint32 min_value" int32 0l
    (Nx_core.Dtype.min_value Nx_core.Dtype.uint32);
  equal ~msg:"uint64 min_value" int64 0L
    (Nx_core.Dtype.min_value Nx_core.Dtype.uint64);

  (* Test max_value *)
  equal ~msg:"int4 max_value" int 7 (Nx_core.Dtype.max_value Nx_core.Dtype.int4);
  equal ~msg:"uint4 max_value" int 15
    (Nx_core.Dtype.max_value Nx_core.Dtype.uint4);
  equal ~msg:"bool max_value" bool true
    (Nx_core.Dtype.max_value Nx_core.Dtype.bool);
  equal ~msg:"uint32 max_value" int32 (Int32.lognot 0l)
    (Nx_core.Dtype.max_value Nx_core.Dtype.uint32);
  equal ~msg:"uint64 max_value" int64 (Int64.lognot 0L)
    (Nx_core.Dtype.max_value Nx_core.Dtype.uint64)

(* ───── Test Suite Setup ───── *)

let suite =
  [
    group " "
      [
        (* BFloat16 tests - supported by Metal *)
        test "create bfloat16" test_create_bfloat16;
        test "scalar bfloat16" test_scalar_bfloat16;
        test "zeros bfloat16" test_zeros_bfloat16;
        test "ones bfloat16" test_ones_bfloat16;
        test "arange bfloat16" test_arange_bfloat16;
        (* Bool tests - supported by Metal *)
        test "create bool" test_create_bool;
        test "scalar bool" test_scalar_bool;
        test "zeros bool" test_zeros_bool;
        test "ones bool" test_ones_bool;
        (* Int4 tests - NOT supported by Metal *)
        test "create int4" test_create_int4;
        test "scalar int4" test_scalar_int4;
        test "zeros int4" test_zeros_int4;
        test "ones int4" test_ones_int4;
        test "arange int4" test_arange_int4;
        (* UInt4 tests - NOT supported by Metal *)
        test "create uint4" test_create_uint4;
        test "scalar uint4" test_scalar_uint4;
        test "zeros uint4" test_zeros_uint4;
        test "ones uint4" test_ones_uint4;
        test "arange uint4" test_arange_uint4;
        (* UInt32 tests - supported by Metal *)
        test "create uint32" test_create_uint32;
        test "scalar uint32" test_scalar_uint32;
        test "zeros uint32" test_zeros_uint32;
        test "ones uint32" test_ones_uint32;
        (* UInt64 tests - supported by Metal *)
        test "create uint64" test_create_uint64;
        test "scalar uint64" test_scalar_uint64;
        test "zeros uint64" test_zeros_uint64;
        test "ones uint64" test_ones_uint64;
        (* Float8_e4m3 tests - NOT supported by Metal *)
        test "create float8_e4m3" test_create_float8_e4m3;
        test "scalar float8_e4m3" test_scalar_float8_e4m3;
        test "zeros float8_e4m3" test_zeros_float8_e4m3;
        test "ones float8_e4m3" test_ones_float8_e4m3;
        (* Float8_e5m2 tests - NOT supported by Metal *)
        test "create float8_e5m2" test_create_float8_e5m2;
        test "scalar float8_e5m2" test_scalar_float8_e5m2;
        test "zeros float8_e5m2" test_zeros_float8_e5m2;
        test "ones float8_e5m2" test_ones_float8_e5m2;
        (* Dtype property tests - always included *)
        test "dtype properties" test_dtype_properties;
        test "dtype min/max values" test_dtype_min_max_values;
      ];
    group "half precision"
      [
        test "bfloat16 round-to-nearest-even" test_bfloat16_rne;
        test "float16 round-to-nearest-even" test_float16_rne;
        test "float16 subnormals" test_float16_subnormals;
        test "float16 overflow to inf" test_float16_overflow_to_inf;
        test "float16 cast roundtrip"
          (test_half_cast_roundtrip "float16" Nx_core.Dtype.float16);
        test "bfloat16 cast roundtrip"
          (test_half_cast_roundtrip "bfloat16" Nx_core.Dtype.bfloat16);
        test "float16 special values"
          (test_half_special_values "float16" Nx_core.Dtype.float16);
        test "bfloat16 special values"
          (test_half_special_values "bfloat16" Nx_core.Dtype.bfloat16);
        test "binary op rounding" test_half_binary_rounding;
        test "float16 binary exact"
          (test_half_binary_exact "float16" Nx_core.Dtype.float16);
        test "bfloat16 binary exact"
          (test_half_binary_exact "bfloat16" Nx_core.Dtype.bfloat16);
        test "sum accumulates wide" test_half_sum_accumulates_wide;
        test "float16 reductions"
          (test_half_reductions "float16" Nx_core.Dtype.float16
             ~mean:0.91650390625);
        test "bfloat16 reductions"
          (test_half_reductions "bfloat16" Nx_core.Dtype.bfloat16
             ~mean:0.91796875);
        test "float16 matmul"
          (test_half_matmul "float16" Nx_core.Dtype.float16 ~eps:0.01);
        test "bfloat16 matmul"
          (test_half_matmul "bfloat16" Nx_core.Dtype.bfloat16 ~eps:0.07);
        test "float16 compare and where"
          (test_half_compare_where "float16" Nx_core.Dtype.float16);
        test "bfloat16 compare and where"
          (test_half_compare_where "bfloat16" Nx_core.Dtype.bfloat16);
      ];
  ]

let () = run "Nx Extended Dtypes" suite
