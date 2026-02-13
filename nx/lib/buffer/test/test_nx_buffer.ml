(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Nx_buffer
open Windtrap

(* Test creation of different array types *)
let test_create_bfloat16 () =
  let arr = Array1.create bfloat16 c_layout 10 in
  equal ~msg:"bfloat16 array size" int 10 (Array1.dim arr);
  (* Verify we can set and get values *)
  Array1.set arr 0 1.0;
  Array1.set arr 5 2.5;
  equal ~msg:"bfloat16 get" (float 0.1) 1.0 (Array1.get arr 0);
  equal ~msg:"bfloat16 get" (float 0.1) 2.5 (Array1.get arr 5)

let test_create_bool () =
  let arr = Array1.create Nx_buffer.bool c_layout 8 in
  equal ~msg:"bool array size" int 8 (Array1.dim arr);
  Array1.set arr 0 true;
  Array1.set arr 1 false;
  Array1.set arr 7 true;
  equal ~msg:"bool get" bool true (Array1.get arr 0);
  equal ~msg:"bool get" bool false (Array1.get arr 1);
  equal ~msg:"bool get" bool true (Array1.get arr 7)

let test_create_int4_signed () =
  let arr = Array1.create int4_signed c_layout 16 in
  equal ~msg:"int4_signed array size" int 16 (Array1.dim arr);
  (* int4_signed range: -8 to 7 *)
  Array1.set arr 0 (-8);
  Array1.set arr 1 7;
  Array1.set arr 2 0;
  equal ~msg:"int4_signed get" int (-8) (Array1.get arr 0);
  equal ~msg:"int4_signed get" int 7 (Array1.get arr 1);
  equal ~msg:"int4_signed get" int 0 (Array1.get arr 2)

let test_create_int4_unsigned () =
  let arr = Array1.create int4_unsigned c_layout 16 in
  equal ~msg:"int4_unsigned array size" int 16 (Array1.dim arr);
  (* int4_unsigned range: 0 to 15 *)
  Array1.set arr 0 0;
  Array1.set arr 1 15;
  Array1.set arr 2 8;
  equal ~msg:"int4_unsigned get" int 0 (Array1.get arr 0);
  equal ~msg:"int4_unsigned get" int 15 (Array1.get arr 1);
  equal ~msg:"int4_unsigned get" int 8 (Array1.get arr 2)

let test_create_float8_e4m3 () =
  let arr = Array1.create float8_e4m3 c_layout 10 in
  equal ~msg:"float8_e4m3 array size" int 10 (Array1.dim arr);
  Array1.set arr 0 0.0;
  Array1.set arr 1 1.0;
  Array1.set arr 2 (-1.5);
  (* Note: float8 has limited precision *)
  equal ~msg:"float8_e4m3 get" (float 0.1) 0.0 (Array1.get arr 0);
  equal ~msg:"float8_e4m3 get" (float 0.1) 1.0 (Array1.get arr 1);
  equal ~msg:"float8_e4m3 get" (float 0.1) (-1.5) (Array1.get arr 2)

let test_create_float8_e5m2 () =
  let arr = Array1.create float8_e5m2 c_layout 10 in
  equal ~msg:"float8_e5m2 array size" int 10 (Array1.dim arr);
  Array1.set arr 0 0.0;
  Array1.set arr 1 2.0;
  Array1.set arr 2 (-0.5);
  equal ~msg:"float8_e5m2 get" (float 0.1) 0.0 (Array1.get arr 0);
  equal ~msg:"float8_e5m2 get" (float 0.1) 2.0 (Array1.get arr 1);
  equal ~msg:"float8_e5m2 get" (float 0.1) (-0.5) (Array1.get arr 2)

(* Test multi-dimensional arrays *)
let test_create_2d_arrays () =
  let arr_bf16 = Array2.create bfloat16 c_layout 3 4 in
  equal ~msg:"2D bfloat16 dim1" int 3 (Array2.dim1 arr_bf16);
  equal ~msg:"2D bfloat16 dim2" int 4 (Array2.dim2 arr_bf16);

  let arr_bool = Array2.create Nx_buffer.bool c_layout 2 5 in
  equal ~msg:"2D bool dim1" int 2 (Array2.dim1 arr_bool);
  equal ~msg:"2D bool dim2" int 5 (Array2.dim2 arr_bool);

  Array2.set arr_bf16 0 0 1.5;
  Array2.set arr_bf16 2 3 3.14;
  equal ~msg:"2D bfloat16 get" (float 0.1) 1.5 (Array2.get arr_bf16 0 0);
  equal ~msg:"2D bfloat16 get" (float 0.1) 3.14 (Array2.get arr_bf16 2 3);

  Array2.set arr_bool 0 0 true;
  Array2.set arr_bool 1 4 false;
  equal ~msg:"2D bool get" bool true (Array2.get arr_bool 0 0);
  equal ~msg:"2D bool get" bool false (Array2.get arr_bool 1 4)

let test_create_3d_arrays () =
  let arr = Array3.create int4_unsigned c_layout 2 3 4 in
  equal ~msg:"3D int4_unsigned dim1" int 2 (Array3.dim1 arr);
  equal ~msg:"3D int4_unsigned dim2" int 3 (Array3.dim2 arr);
  equal ~msg:"3D int4_unsigned dim3" int 4 (Array3.dim3 arr);

  Array3.set arr 0 0 0 5;
  Array3.set arr 1 2 3 15;
  equal ~msg:"3D int4_unsigned get" int 5 (Array3.get arr 0 0 0);
  equal ~msg:"3D int4_unsigned get" int 15 (Array3.get arr 1 2 3)

(* Test Genarray creation *)
let test_genarray_creation () =
  let dims = [| 2; 3; 4 |] in
  let arr_bf16 = Genarray.create bfloat16 c_layout dims in
  let arr_bool = Genarray.create Nx_buffer.bool c_layout dims in
  let arr_fp8 = Genarray.create float8_e4m3 c_layout dims in

  equal ~msg:"Genarray bfloat16 dims" int 3 (Genarray.num_dims arr_bf16);
  equal ~msg:"Genarray bool dims" int 3 (Genarray.num_dims arr_bool);
  equal ~msg:"Genarray float8 dims" int 3 (Genarray.num_dims arr_fp8);

  equal ~msg:"Genarray dim 0" int 2 (Genarray.nth_dim arr_bf16 0);
  equal ~msg:"Genarray dim 1" int 3 (Genarray.nth_dim arr_bf16 1);
  equal ~msg:"Genarray dim 2" int 4 (Genarray.nth_dim arr_bf16 2)

(* Test Genarray get/set operations with extended types *)
let test_genarray_get_set () =
  (* Test bfloat16 *)
  let arr_bf16 = Genarray.create bfloat16 c_layout [| 2; 3 |] in
  Genarray.set arr_bf16 [| 0; 0 |] 1.5;
  Genarray.set arr_bf16 [| 1; 2 |] 3.14;
  equal ~msg:"Genarray bfloat16 get" (float 0.1) 1.5
    (Genarray.get arr_bf16 [| 0; 0 |]);
  equal ~msg:"Genarray bfloat16 get" (float 0.1) 3.14
    (Genarray.get arr_bf16 [| 1; 2 |]);

  (* Test bool *)
  let arr_bool = Genarray.create Nx_buffer.bool c_layout [| 3; 3 |] in
  Genarray.set arr_bool [| 0; 0 |] true;
  Genarray.set arr_bool [| 2; 1 |] false;
  equal ~msg:"Genarray bool get" bool true
    (Genarray.get arr_bool [| 0; 0 |]);
  equal ~msg:"Genarray bool get" bool false
    (Genarray.get arr_bool [| 2; 1 |]);

  (* Test int4_signed *)
  let arr_int4 = Genarray.create int4_signed c_layout [| 4 |] in
  Genarray.set arr_int4 [| 0 |] (-8);
  Genarray.set arr_int4 [| 1 |] 7;
  Genarray.set arr_int4 [| 2 |] 0;
  equal ~msg:"Genarray int4_signed get" int (-8)
    (Genarray.get arr_int4 [| 0 |]);
  equal ~msg:"Genarray int4_signed get" int 7
    (Genarray.get arr_int4 [| 1 |]);
  equal ~msg:"Genarray int4_signed get" int 0
    (Genarray.get arr_int4 [| 2 |]);

  (* Test float8_e4m3 *)
  let arr_fp8 = Genarray.create float8_e4m3 c_layout [| 2; 2 |] in
  Genarray.set arr_fp8 [| 0; 0 |] 1.0;
  Genarray.set arr_fp8 [| 1; 1 |] (-2.0);
  equal ~msg:"Genarray float8_e4m3 get" (float 0.2) 1.0
    (Genarray.get arr_fp8 [| 0; 0 |]);
  equal ~msg:"Genarray float8_e4m3 get" (float 0.2) (-2.0)
    (Genarray.get arr_fp8 [| 1; 1 |])

(* Test kind_size_in_bytes *)
let test_kind_sizes () =
  equal ~msg:"bfloat16 size" int 2 (kind_size_in_bytes bfloat16);
  equal ~msg:"bool size" int 1 (kind_size_in_bytes Nx_buffer.bool);
  equal ~msg:"int4_signed size" int 1 (kind_size_in_bytes int4_signed);
  equal ~msg:"int4_unsigned size" int 1 (kind_size_in_bytes int4_unsigned);
  equal ~msg:"float8_e4m3 size" int 1 (kind_size_in_bytes float8_e4m3);
  equal ~msg:"float8_e5m2 size" int 1 (kind_size_in_bytes float8_e5m2);
  equal ~msg:"uint32 size" int 4 (kind_size_in_bytes uint32);
  equal ~msg:"uint64 size" int 8 (kind_size_in_bytes uint64);
  (* Also test standard types *)
  equal ~msg:"float32 size" int 4 (kind_size_in_bytes float32);
  equal ~msg:"float64 size" int 8 (kind_size_in_bytes float64);
  equal ~msg:"int32 size" int 4 (kind_size_in_bytes Nx_buffer.int32)

(* Test fortran layout *)
let test_fortran_layout () =
  let arr = Array2.create bfloat16 fortran_layout 3 4 in
  equal ~msg:"Fortran layout dim1" int 3 (Array2.dim1 arr);
  equal ~msg:"Fortran layout dim2" int 4 (Array2.dim2 arr);

  (* Fortran layout uses 1-based indexing *)
  Array2.set arr 1 1 2.5;
  Array2.set arr 3 4 1.5;
  equal ~msg:"Fortran get" (float 0.1) 2.5 (Array2.get arr 1 1);
  equal ~msg:"Fortran get" (float 0.1) 1.5 (Array2.get arr 3 4)

(* Test Array0 *)
let test_array0 () =
  let arr = Array0.create Nx_buffer.bool c_layout in
  Array0.set arr true;
  equal ~msg:"Array0 bool" bool true (Array0.get arr);

  let arr_bf16 = Array0.create bfloat16 c_layout in
  Array0.set arr_bf16 3.14;
  equal ~msg:"Array0 bfloat16" (float 0.1) 3.14 (Array0.get arr_bf16)

(* Test suite *)
let () =
  run "Nx_buffer tests"
    [
      group "creation"
        [
          test "create bfloat16" test_create_bfloat16;
          test "create bool" test_create_bool;
          test "create int4_signed" test_create_int4_signed;
          test "create int4_unsigned" test_create_int4_unsigned;
          test "create float8_e4m3" test_create_float8_e4m3;
          test "create float8_e5m2" test_create_float8_e5m2;
        ];
      group "multi-dimensional"
        [
          test "2D arrays" test_create_2d_arrays;
          test "3D arrays" test_create_3d_arrays;
          test "Genarray" test_genarray_creation;
          test "Genarray get/set" test_genarray_get_set;
          test "Array0" test_array0;
        ];
      group "properties"
        [
          test "kind sizes" test_kind_sizes;
          test "fortran layout" test_fortran_layout;
        ];
    ]
