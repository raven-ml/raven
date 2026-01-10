(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Tests for extended bigarray dtypes *)

open Alcotest
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
  check bool "bfloat16 is_float" true
    (Nx_core.Dtype.is_float Nx_core.Dtype.bfloat16);
  check bool "float8_e4m3 is_float" true
    (Nx_core.Dtype.is_float Nx_core.Dtype.float8_e4m3);
  check bool "float8_e5m2 is_float" true
    (Nx_core.Dtype.is_float Nx_core.Dtype.float8_e5m2);
  check bool "bool is_float" false (Nx_core.Dtype.is_float Nx_core.Dtype.bool);

  (* Test is_complex *)
  check bool "complex64 is_complex" true
    (Nx_core.Dtype.is_complex Nx_core.Dtype.complex64);
  check bool "complex128 is_complex" true
    (Nx_core.Dtype.is_complex Nx_core.Dtype.complex128);
  check bool "bfloat16 is_complex" false
    (Nx_core.Dtype.is_complex Nx_core.Dtype.bfloat16);

  (* Test is_int *)
  check bool "int4 is_int" true (Nx_core.Dtype.is_int Nx_core.Dtype.int4);
  check bool "uint4 is_int" true (Nx_core.Dtype.is_int Nx_core.Dtype.uint4);
  check bool "uint32 is_int" true (Nx_core.Dtype.is_int Nx_core.Dtype.uint32);
  check bool "uint64 is_int" true (Nx_core.Dtype.is_int Nx_core.Dtype.uint64);
  check bool "bool is_int" false (Nx_core.Dtype.is_int Nx_core.Dtype.bool);

  (* Test is_uint *)
  check bool "uint4 is_uint" true (Nx_core.Dtype.is_uint Nx_core.Dtype.uint4);
  check bool "uint32 is_uint" true (Nx_core.Dtype.is_uint Nx_core.Dtype.uint32);
  check bool "uint64 is_uint" true (Nx_core.Dtype.is_uint Nx_core.Dtype.uint64);
  check bool "int4 is_uint" false (Nx_core.Dtype.is_uint Nx_core.Dtype.int4);

  (* Test itemsize *)
  check int "bfloat16 itemsize" 2
    (Nx_core.Dtype.itemsize Nx_core.Dtype.bfloat16);
  check int "bool itemsize" 1 (Nx_core.Dtype.itemsize Nx_core.Dtype.bool);
  check int "int4 itemsize" 1 (Nx_core.Dtype.itemsize Nx_core.Dtype.int4);
  check int "uint4 itemsize" 1 (Nx_core.Dtype.itemsize Nx_core.Dtype.uint4);
  check int "float8_e4m3 itemsize" 1
    (Nx_core.Dtype.itemsize Nx_core.Dtype.float8_e4m3);
  check int "float8_e5m2 itemsize" 1
    (Nx_core.Dtype.itemsize Nx_core.Dtype.float8_e5m2);
  check int "uint32 itemsize" 4 (Nx_core.Dtype.itemsize Nx_core.Dtype.uint32);
  check int "uint64 itemsize" 8 (Nx_core.Dtype.itemsize Nx_core.Dtype.uint64);
  check int "complex64 itemsize" 8
    (Nx_core.Dtype.itemsize Nx_core.Dtype.complex64);
  check int "complex128 itemsize" 16
    (Nx_core.Dtype.itemsize Nx_core.Dtype.complex128);

  (* Test to_string *)
  check string "bfloat16 to_string" "bfloat16"
    (Nx_core.Dtype.to_string Nx_core.Dtype.bfloat16);
  check string "bool to_string" "bool"
    (Nx_core.Dtype.to_string Nx_core.Dtype.bool);
  check string "int4 to_string" "int4"
    (Nx_core.Dtype.to_string Nx_core.Dtype.int4);
  check string "uint4 to_string" "uint4"
    (Nx_core.Dtype.to_string Nx_core.Dtype.uint4);
  check string "float8_e4m3 to_string" "float8_e4m3"
    (Nx_core.Dtype.to_string Nx_core.Dtype.float8_e4m3);
  check string "float8_e5m2 to_string" "float8_e5m2"
    (Nx_core.Dtype.to_string Nx_core.Dtype.float8_e5m2);
  check string "uint32 to_string" "uint32"
    (Nx_core.Dtype.to_string Nx_core.Dtype.uint32);
  check string "uint64 to_string" "uint64"
    (Nx_core.Dtype.to_string Nx_core.Dtype.uint64);
  check string "complex64 to_string" "complex64"
    (Nx_core.Dtype.to_string Nx_core.Dtype.complex64);
  check string "complex128 to_string" "complex128"
    (Nx_core.Dtype.to_string Nx_core.Dtype.complex128)

let test_dtype_min_max_values () =
  (* Test min_value *)
  check int "int4 min_value" (-8) (Nx_core.Dtype.min_value Nx_core.Dtype.int4);
  check int "uint4 min_value" 0 (Nx_core.Dtype.min_value Nx_core.Dtype.uint4);
  check bool "bool min_value" false (Nx_core.Dtype.min_value Nx_core.Dtype.bool);
  check int32 "uint32 min_value" 0l
    (Nx_core.Dtype.min_value Nx_core.Dtype.uint32);
  check int64 "uint64 min_value" 0L
    (Nx_core.Dtype.min_value Nx_core.Dtype.uint64);

  (* Test max_value *)
  check int "int4 max_value" 7 (Nx_core.Dtype.max_value Nx_core.Dtype.int4);
  check int "uint4 max_value" 15 (Nx_core.Dtype.max_value Nx_core.Dtype.uint4);
  check bool "bool max_value" true (Nx_core.Dtype.max_value Nx_core.Dtype.bool);
  check int32 "uint32 max_value" (Int32.lognot 0l)
    (Nx_core.Dtype.max_value Nx_core.Dtype.uint32);
  check int64 "uint64 max_value" (Int64.lognot 0L)
    (Nx_core.Dtype.max_value Nx_core.Dtype.uint64)

(* ───── Test Suite Setup ───── *)

let suite =
  [
    ( "Extended Dtypes :: ",
      [
        (* BFloat16 tests - supported by Metal *)
        test_case "create bfloat16" `Quick test_create_bfloat16;
        test_case "scalar bfloat16" `Quick test_scalar_bfloat16;
        test_case "zeros bfloat16" `Quick test_zeros_bfloat16;
        test_case "ones bfloat16" `Quick test_ones_bfloat16;
        test_case "arange bfloat16" `Quick test_arange_bfloat16;
        (* Bool tests - supported by Metal *)
        test_case "create bool" `Quick test_create_bool;
        test_case "scalar bool" `Quick test_scalar_bool;
        test_case "zeros bool" `Quick test_zeros_bool;
        test_case "ones bool" `Quick test_ones_bool;
        (* Int4 tests - NOT supported by Metal *)
        test_case "create int4" `Quick test_create_int4;
        test_case "scalar int4" `Quick test_scalar_int4;
        test_case "zeros int4" `Quick test_zeros_int4;
        test_case "ones int4" `Quick test_ones_int4;
        test_case "arange int4" `Quick test_arange_int4;
        (* UInt4 tests - NOT supported by Metal *)
        test_case "create uint4" `Quick test_create_uint4;
        test_case "scalar uint4" `Quick test_scalar_uint4;
        test_case "zeros uint4" `Quick test_zeros_uint4;
        test_case "ones uint4" `Quick test_ones_uint4;
        test_case "arange uint4" `Quick test_arange_uint4;
        (* UInt32 tests - supported by Metal *)
        test_case "create uint32" `Quick test_create_uint32;
        test_case "scalar uint32" `Quick test_scalar_uint32;
        test_case "zeros uint32" `Quick test_zeros_uint32;
        test_case "ones uint32" `Quick test_ones_uint32;
        (* UInt64 tests - supported by Metal *)
        test_case "create uint64" `Quick test_create_uint64;
        test_case "scalar uint64" `Quick test_scalar_uint64;
        test_case "zeros uint64" `Quick test_zeros_uint64;
        test_case "ones uint64" `Quick test_ones_uint64;
        (* Float8_e4m3 tests - NOT supported by Metal *)
        test_case "create float8_e4m3" `Quick test_create_float8_e4m3;
        test_case "scalar float8_e4m3" `Quick test_scalar_float8_e4m3;
        test_case "zeros float8_e4m3" `Quick test_zeros_float8_e4m3;
        test_case "ones float8_e4m3" `Quick test_ones_float8_e4m3;
        (* Float8_e5m2 tests - NOT supported by Metal *)
        test_case "create float8_e5m2" `Quick test_create_float8_e5m2;
        test_case "scalar float8_e5m2" `Quick test_scalar_float8_e5m2;
        test_case "zeros float8_e5m2" `Quick test_zeros_float8_e5m2;
        test_case "ones float8_e5m2" `Quick test_ones_float8_e5m2;
        (* Dtype property tests - always included *)
        test_case "dtype properties" `Quick test_dtype_properties;
        test_case "dtype min/max values" `Quick test_dtype_min_max_values;
      ] );
  ]

let () = Alcotest.run "Nx Extended Dtypes" suite
