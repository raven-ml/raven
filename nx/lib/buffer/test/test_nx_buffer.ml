open Nx_buffer

(* Test creation of different array types *)
let test_create_bfloat16 () =
  let arr = Array1.create bfloat16 c_layout 10 in
  Alcotest.(check int) "bfloat16 array size" 10 (Array1.dim arr);
  (* Verify we can set and get values *)
  Array1.set arr 0 1.0;
  Array1.set arr 5 2.5;
  Alcotest.(check (float 0.1)) "bfloat16 get" 1.0 (Array1.get arr 0);
  Alcotest.(check (float 0.1)) "bfloat16 get" 2.5 (Array1.get arr 5)

let test_create_bool () =
  let arr = Array1.create bool c_layout 8 in
  Alcotest.(check int) "bool array size" 8 (Array1.dim arr);
  Array1.set arr 0 true;
  Array1.set arr 1 false;
  Array1.set arr 7 true;
  Alcotest.(check bool) "bool get" true (Array1.get arr 0);
  Alcotest.(check bool) "bool get" false (Array1.get arr 1);
  Alcotest.(check bool) "bool get" true (Array1.get arr 7)

let test_create_int4_signed () =
  let arr = Array1.create int4_signed c_layout 16 in
  Alcotest.(check int) "int4_signed array size" 16 (Array1.dim arr);
  (* int4_signed range: -8 to 7 *)
  Array1.set arr 0 (-8);
  Array1.set arr 1 7;
  Array1.set arr 2 0;
  Alcotest.(check int) "int4_signed get" (-8) (Array1.get arr 0);
  Alcotest.(check int) "int4_signed get" 7 (Array1.get arr 1);
  Alcotest.(check int) "int4_signed get" 0 (Array1.get arr 2)

let test_create_int4_unsigned () =
  let arr = Array1.create int4_unsigned c_layout 16 in
  Alcotest.(check int) "int4_unsigned array size" 16 (Array1.dim arr);
  (* int4_unsigned range: 0 to 15 *)
  Array1.set arr 0 0;
  Array1.set arr 1 15;
  Array1.set arr 2 8;
  Alcotest.(check int) "int4_unsigned get" 0 (Array1.get arr 0);
  Alcotest.(check int) "int4_unsigned get" 15 (Array1.get arr 1);
  Alcotest.(check int) "int4_unsigned get" 8 (Array1.get arr 2)

let test_create_float8_e4m3 () =
  let arr = Array1.create float8_e4m3 c_layout 10 in
  Alcotest.(check int) "float8_e4m3 array size" 10 (Array1.dim arr);
  Array1.set arr 0 0.0;
  Array1.set arr 1 1.0;
  Array1.set arr 2 (-1.5);
  (* Note: float8 has limited precision *)
  Alcotest.(check (float 0.1)) "float8_e4m3 get" 0.0 (Array1.get arr 0);
  Alcotest.(check (float 0.1)) "float8_e4m3 get" 1.0 (Array1.get arr 1);
  Alcotest.(check (float 0.1)) "float8_e4m3 get" (-1.5) (Array1.get arr 2)

let test_create_float8_e5m2 () =
  let arr = Array1.create float8_e5m2 c_layout 10 in
  Alcotest.(check int) "float8_e5m2 array size" 10 (Array1.dim arr);
  Array1.set arr 0 0.0;
  Array1.set arr 1 2.0;
  Array1.set arr 2 (-0.5);
  Alcotest.(check (float 0.1)) "float8_e5m2 get" 0.0 (Array1.get arr 0);
  Alcotest.(check (float 0.1)) "float8_e5m2 get" 2.0 (Array1.get arr 1);
  Alcotest.(check (float 0.1)) "float8_e5m2 get" (-0.5) (Array1.get arr 2)

(* Test multi-dimensional arrays *)
let test_create_2d_arrays () =
  let arr_bf16 = Array2.create bfloat16 c_layout 3 4 in
  Alcotest.(check int) "2D bfloat16 dim1" 3 (Array2.dim1 arr_bf16);
  Alcotest.(check int) "2D bfloat16 dim2" 4 (Array2.dim2 arr_bf16);

  let arr_bool = Array2.create bool c_layout 2 5 in
  Alcotest.(check int) "2D bool dim1" 2 (Array2.dim1 arr_bool);
  Alcotest.(check int) "2D bool dim2" 5 (Array2.dim2 arr_bool);

  Array2.set arr_bf16 0 0 1.5;
  Array2.set arr_bf16 2 3 3.14;
  Alcotest.(check (float 0.1)) "2D bfloat16 get" 1.5 (Array2.get arr_bf16 0 0);
  Alcotest.(check (float 0.1)) "2D bfloat16 get" 3.14 (Array2.get arr_bf16 2 3);

  Array2.set arr_bool 0 0 true;
  Array2.set arr_bool 1 4 false;
  Alcotest.(check bool) "2D bool get" true (Array2.get arr_bool 0 0);
  Alcotest.(check bool) "2D bool get" false (Array2.get arr_bool 1 4)

let test_create_3d_arrays () =
  let arr = Array3.create int4_unsigned c_layout 2 3 4 in
  Alcotest.(check int) "3D int4_unsigned dim1" 2 (Array3.dim1 arr);
  Alcotest.(check int) "3D int4_unsigned dim2" 3 (Array3.dim2 arr);
  Alcotest.(check int) "3D int4_unsigned dim3" 4 (Array3.dim3 arr);

  Array3.set arr 0 0 0 5;
  Array3.set arr 1 2 3 15;
  Alcotest.(check int) "3D int4_unsigned get" 5 (Array3.get arr 0 0 0);
  Alcotest.(check int) "3D int4_unsigned get" 15 (Array3.get arr 1 2 3)

(* Test Genarray creation *)
let test_genarray_creation () =
  let dims = [| 2; 3; 4 |] in
  let arr_bf16 = Genarray.create bfloat16 c_layout dims in
  let arr_bool = Genarray.create bool c_layout dims in
  let arr_fp8 = Genarray.create float8_e4m3 c_layout dims in

  Alcotest.(check int) "Genarray bfloat16 dims" 3 (Genarray.num_dims arr_bf16);
  Alcotest.(check int) "Genarray bool dims" 3 (Genarray.num_dims arr_bool);
  Alcotest.(check int) "Genarray float8 dims" 3 (Genarray.num_dims arr_fp8);

  Alcotest.(check int) "Genarray dim 0" 2 (Genarray.nth_dim arr_bf16 0);
  Alcotest.(check int) "Genarray dim 1" 3 (Genarray.nth_dim arr_bf16 1);
  Alcotest.(check int) "Genarray dim 2" 4 (Genarray.nth_dim arr_bf16 2)

(* Test Genarray get/set operations with extended types *)
let test_genarray_get_set () =
  (* Test bfloat16 *)
  let arr_bf16 = Genarray.create bfloat16 c_layout [| 2; 3 |] in
  Genarray.set arr_bf16 [| 0; 0 |] 1.5;
  Genarray.set arr_bf16 [| 1; 2 |] 3.14;
  Alcotest.(check (float 0.1))
    "Genarray bfloat16 get" 1.5
    (Genarray.get arr_bf16 [| 0; 0 |]);
  Alcotest.(check (float 0.1))
    "Genarray bfloat16 get" 3.14
    (Genarray.get arr_bf16 [| 1; 2 |]);

  (* Test bool *)
  let arr_bool = Genarray.create bool c_layout [| 3; 3 |] in
  Genarray.set arr_bool [| 0; 0 |] true;
  Genarray.set arr_bool [| 2; 1 |] false;
  Alcotest.(check bool)
    "Genarray bool get" true
    (Genarray.get arr_bool [| 0; 0 |]);
  Alcotest.(check bool)
    "Genarray bool get" false
    (Genarray.get arr_bool [| 2; 1 |]);

  (* Test int4_signed *)
  let arr_int4 = Genarray.create int4_signed c_layout [| 4 |] in
  Genarray.set arr_int4 [| 0 |] (-8);
  Genarray.set arr_int4 [| 1 |] 7;
  Genarray.set arr_int4 [| 2 |] 0;
  Alcotest.(check int)
    "Genarray int4_signed get" (-8)
    (Genarray.get arr_int4 [| 0 |]);
  Alcotest.(check int)
    "Genarray int4_signed get" 7
    (Genarray.get arr_int4 [| 1 |]);
  Alcotest.(check int)
    "Genarray int4_signed get" 0
    (Genarray.get arr_int4 [| 2 |]);

  (* Test float8_e4m3 *)
  let arr_fp8 = Genarray.create float8_e4m3 c_layout [| 2; 2 |] in
  Genarray.set arr_fp8 [| 0; 0 |] 1.0;
  Genarray.set arr_fp8 [| 1; 1 |] (-2.0);
  Alcotest.(check (float 0.2))
    "Genarray float8_e4m3 get" 1.0
    (Genarray.get arr_fp8 [| 0; 0 |]);
  Alcotest.(check (float 0.2))
    "Genarray float8_e4m3 get" (-2.0)
    (Genarray.get arr_fp8 [| 1; 1 |])

(* Test kind_size_in_bytes *)
let test_kind_sizes () =
  Alcotest.(check int) "bfloat16 size" 2 (kind_size_in_bytes bfloat16);
  Alcotest.(check int) "bool size" 1 (kind_size_in_bytes bool);
  Alcotest.(check int) "int4_signed size" 1 (kind_size_in_bytes int4_signed);
  Alcotest.(check int) "int4_unsigned size" 1 (kind_size_in_bytes int4_unsigned);
  Alcotest.(check int) "float8_e4m3 size" 1 (kind_size_in_bytes float8_e4m3);
  Alcotest.(check int) "float8_e5m2 size" 1 (kind_size_in_bytes float8_e5m2);
  Alcotest.(check int) "uint32 size" 4 (kind_size_in_bytes uint32);
  Alcotest.(check int) "uint64 size" 8 (kind_size_in_bytes uint64);
  (* Also test standard types *)
  Alcotest.(check int) "float32 size" 4 (kind_size_in_bytes float32);
  Alcotest.(check int) "float64 size" 8 (kind_size_in_bytes float64);
  Alcotest.(check int) "int32 size" 4 (kind_size_in_bytes int32)

(* Test fortran layout *)
let test_fortran_layout () =
  let arr = Array2.create bfloat16 fortran_layout 3 4 in
  Alcotest.(check int) "Fortran layout dim1" 3 (Array2.dim1 arr);
  Alcotest.(check int) "Fortran layout dim2" 4 (Array2.dim2 arr);

  (* Fortran layout uses 1-based indexing *)
  Array2.set arr 1 1 2.5;
  Array2.set arr 3 4 1.5;
  Alcotest.(check (float 0.1)) "Fortran get" 2.5 (Array2.get arr 1 1);
  Alcotest.(check (float 0.1)) "Fortran get" 1.5 (Array2.get arr 3 4)

(* Test Array0 *)
let test_array0 () =
  let arr = Array0.create bool c_layout in
  Array0.set arr true;
  Alcotest.(check bool) "Array0 bool" true (Array0.get arr);

  let arr_bf16 = Array0.create bfloat16 c_layout in
  Array0.set arr_bf16 3.14;
  Alcotest.(check (float 0.1)) "Array0 bfloat16" 3.14 (Array0.get arr_bf16)

(* Test suite *)
let () =
  let open Alcotest in
  run "Nx_buffer tests"
    [
      ( "creation",
        [
          test_case "create bfloat16" `Quick test_create_bfloat16;
          test_case "create bool" `Quick test_create_bool;
          test_case "create int4_signed" `Quick test_create_int4_signed;
          test_case "create int4_unsigned" `Quick test_create_int4_unsigned;
          test_case "create float8_e4m3" `Quick test_create_float8_e4m3;
          test_case "create float8_e5m2" `Quick test_create_float8_e5m2;
        ] );
      ( "multi-dimensional",
        [
          test_case "2D arrays" `Quick test_create_2d_arrays;
          test_case "3D arrays" `Quick test_create_3d_arrays;
          test_case "Genarray" `Quick test_genarray_creation;
          test_case "Genarray get/set" `Quick test_genarray_get_set;
          test_case "Array0" `Quick test_array0;
        ] );
      ( "properties",
        [
          test_case "kind sizes" `Quick test_kind_sizes;
          test_case "fortran layout" `Quick test_fortran_layout;
        ] );
    ]
