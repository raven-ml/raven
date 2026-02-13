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
  ]

let () = run "Nx Extended Dtypes" suite
