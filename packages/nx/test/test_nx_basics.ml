(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Core functionality tests for Nx - creation, indexing, properties *)

open Windtrap
open Test_nx_support

(* ───── Creation Edge Case Tests ───── *)

let test_create_1d_int32 () =
  let t = Nx.create Nx.int32 [| 3 |] [| 1l; 2l; 3l |] in
  check_t "create 1D int32" [| 3 |] [| 1l; 2l; 3l |] t

let test_create_empty_float32 () =
  let t = Nx.create Nx.float32 [| 0 |] [||] in
  check_shape "empty shape" [| 0 |] t

let test_create_2x2x2_float32 () =
  let t = Nx.create Nx.float32 [| 2; 2; 2 |] (Array.init 8 float_of_int) in
  check_t "create 2x2x2" [| 2; 2; 2 |] [| 0.; 1.; 2.; 3.; 4.; 5.; 6.; 7. |] t

let test_scalar_float32 () =
  let t = Nx.scalar Nx.float32 42.0 in
  check_t "scalar float32" [||] [| 42.0 |] t

let test_scalar_int64 () =
  let t = Nx.scalar Nx.int64 100L in
  check_t "scalar int64" [||] [| 100L |] t

let test_create_int16 () =
  let t = Nx.create Nx.int16 [| 4 |] [| 1; 2; 3; 4 |] in
  check_t "create int16" [| 4 |] [| 1; 2; 3; 4 |] t

let test_create_empty_shapes () =
  (* Empty 1D array *)
  let t1 = Nx.create Nx.float32 [| 0 |] [||] in
  check_shape "empty 1D shape" [| 0 |] t1;

  (* Empty multi-dimensional arrays *)
  let t2 = Nx.create Nx.float32 [| 0; 5 |] [||] in
  check_shape "empty 2D shape [0,5]" [| 0; 5 |] t2;

  let t3 = Nx.create Nx.float32 [| 5; 0; 3 |] [||] in
  check_shape "empty 3D shape [5,0,3]" [| 5; 0; 3 |] t3

let test_create_max_rank () =
  (* Create array with many dimensions but small total size *)
  (* Use shape like [1, 1, 1, ..., 2, 2, 2] to keep total size manageable *)
  let shape = Array.init 32 (fun i -> if i < 29 then 1 else 2) in
  let data_size = Array.fold_left ( * ) 1 shape in
  (* = 8 total elements *)
  let data = Array.init data_size float_of_int in
  let t = Nx.create Nx.float32 shape data in
  equal ~msg:"ndim of 32D array" int 32 (Nx.ndim t);
  check_shape "32D shape" shape t

let test_create_wrong_data_size () =
  check_invalid_arg "data size mismatch"
    "create: array size, got 3 elements, expected 6" (fun () ->
      ignore (Nx.create Nx.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0 |]))

let test_create_negative_shape () =
  check_invalid_arg "negative dimension"
    "create: array size, got 2 elements, expected -6" (fun () ->
      ignore (Nx.create Nx.float32 [| 2; -3 |] [| 1.0; 2.0 |]))

(* ───── Special Creation Function Tests ───── *)

let test_empty_float32 () =
  let t = Nx.empty Nx.float32 [| 2; 2 |] in
  check_shape "empty shape" [| 2; 2 |] t

let test_full_float32 () =
  let t = Nx.full Nx.float32 [| 2; 3 |] 5.5 in
  check_t "full" [| 2; 3 |] [| 5.5; 5.5; 5.5; 5.5; 5.5; 5.5 |] t

let test_full_like_int32 () =
  let ref_t = Nx.create Nx.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  let t = Nx.full_like ref_t 10l in
  check_t "full_like" [| 2; 2 |] [| 10l; 10l; 10l; 10l |] t

let test_empty_like_float32 () =
  let ref_t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t = Nx.empty_like ref_t in
  check_shape "empty_like shape" [| 2; 2 |] t

let test_zeros_like_float32 () =
  let ref_t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t = Nx.zeros_like ref_t in
  check_t "zeros_like" [| 2; 2 |] [| 0.; 0.; 0.; 0. |] t

let test_ones_like_int32 () =
  let ref_t = Nx.create Nx.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  let t = Nx.ones_like ref_t in
  check_t "ones_like" [| 2; 2 |] [| 1l; 1l; 1l; 1l |] t

let test_zeros_max_size () =
  let t = Nx.zeros Nx.float32 [| 256; 256; 16 |] in
  check_shape "large zeros shape" [| 256; 256; 16 |] t;
  equal ~msg:"zeros[0,0,0]" (float 1e-6) 0.0 (Nx.item [ 0; 0; 0 ] t)

(* ───── Eye Identity Tests ───── *)

let test_identity_1x1_int32 () =
  let t = Nx.identity Nx.int32 1 in
  check_t "identity 1x1" [| 1; 1 |] [| 1l |] t

let test_eye_3x4_float32 () =
  let t = Nx.eye ~m:3 Nx.float32 4 in
  check_t "eye 3x4" [| 3; 4 |]
    [| 1.; 0.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 0.; 1.; 0. |]
    t

let test_eye_4x3_k1_float32 () =
  let t = Nx.eye ~m:4 ~k:1 Nx.float32 3 in
  check_t "eye 4x3 k=1" [| 4; 3 |]
    [| 0.; 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 0.; 0.; 0. |]
    t

let test_eye_3x3_km1_int32 () =
  let t = Nx.eye ~k:(-1) Nx.int32 3 in
  check_t "eye 3x3 k=-1" [| 3; 3 |] [| 0l; 0l; 0l; 1l; 0l; 0l; 0l; 1l; 0l |] t

let test_eye_0x0 () =
  let t = Nx.eye Nx.float32 0 in
  check_shape "0x0 eye shape" [| 0; 0 |] t

let test_eye_k_out_of_range () =
  (* k offset larger than matrix dimensions *)
  let t1 = Nx.eye Nx.float32 ~k:10 3 in
  check_t "eye with k=10" [| 3; 3 |] [| 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0. |] t1;

  let t2 = Nx.eye Nx.float32 ~k:(-10) 3 in
  check_t "eye with k=-10" [| 3; 3 |]
    [| 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0. |]
    t2

let test_diag_extract () =
  let x = Nx.arange Nx.int32 0 9 1 |> Nx.reshape [| 3; 3 |] in
  check_t "diag main" [| 3 |] [| 0l; 4l; 8l |] (Nx.diag x);
  check_t "diag k=1" [| 2 |] [| 1l; 5l |] (Nx.diag ~k:1 x);
  check_t "diag k=-1" [| 2 |] [| 3l; 7l |] (Nx.diag ~k:(-1) x)

let test_diag_construct () =
  let v = Nx.create Nx.int32 [| 3 |] [| 1l; 2l; 3l |] in
  check_t "diag 1D" [| 3; 3 |]
    [| 1l; 0l; 0l; 0l; 2l; 0l; 0l; 0l; 3l |]
    (Nx.diag v);
  check_t "diag k=1" [| 4; 4 |]
    [| 0l; 1l; 0l; 0l; 0l; 0l; 2l; 0l; 0l; 0l; 0l; 3l; 0l; 0l; 0l; 0l |]
    (Nx.diag ~k:1 v)

(* ───── Range Generation Tests ───── *)

let test_arange_empty () =
  check_invalid_arg "arange empty"
    "arange: range [0, 0), empty with step=1, ensure start < stop for positive step, or start > stop for negative step"
    (fun () -> Nx.arange Nx.int32 0 0 1)

let test_arange_negative_step () =
  let t = Nx.arange Nx.int32 10 0 (-2) in
  check_t "arange negative step" [| 5 |] [| 10l; 8l; 6l; 4l; 2l |] t

let test_arange_wrong_direction () =
  let t = Nx.arange Nx.int32 0 10 (-1) in
  check_shape "arange wrong direction shape" [| 0 |] t

let test_linspace_no_endpoint_float32 () =
  let t = Nx.linspace ~endpoint:false Nx.float32 0.0 4.0 5 in
  check_t ~eps:1e-6 "linspace no endpoint" [| 5 |]
    [| 0.0; 0.8; 1.6; 2.4; 3.2 |]
    t

let test_linspace_single_point () =
  let t = Nx.linspace Nx.float32 5.0 5.0 1 in
  check_t "linspace single point" [| 1 |] [| 5.0 |] t

let test_linspace_zero_points () =
  (* linspace with 0 points returns empty array, doesn't raise error *)
  let t = Nx.linspace Nx.float32 0.0 1.0 0 in
  check_shape "linspace 0 points" [| 0 |] t

let test_logspace_base10_float32 () =
  let t = Nx.logspace ~base:10.0 Nx.float32 2.0 3.0 4 in
  check_t ~eps:1e-3 "logspace base 10" [| 4 |]
    [| 100.0; 215.443469003188454; 464.158883361277731; 1000.0 |]
    t

let test_logspace_base2_no_endpoint_float32 () =
  let t = Nx.logspace ~endpoint:false ~base:2.0 Nx.float32 0.0 4.0 5 in
  check_t ~eps:1e-6 "logspace base 2 no endpoint" [| 5 |]
    [|
      1.0;
      1.741101126592248;
      3.031433133020796;
      5.278031643091579;
      9.189586839976281;
    |]
    t

let test_geomspace_no_endpoint_float32 () =
  let t = Nx.geomspace ~endpoint:false Nx.float32 1.0 256.0 9 in
  check_t ~eps:1e-4 "geomspace no endpoint" [| 9 |]
    [|
      1.0;
      1.851749424574581;
      3.428975931412292;
      6.349604207872799;
      11.757875938204792;
      21.772640002790030;
      40.317473596635956;
      74.657858532871487;
      138.247646578215210;
    |]
    t

(* ───── Property Access Tests ───── *)

let test_shape_2x3 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] (Array.init 6 float_of_int) in
  check_shape "shape 2x3" [| 2; 3 |] t

let test_strides_2x3_float32 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] (Array.init 6 float_of_int) in
  equal ~msg:"strides" (array int) [| 12; 4 |] (Nx.strides t)

let test_stride_dim0_2x3_float32 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] (Array.init 6 float_of_int) in
  equal ~msg:"stride dim 0" int 12 (Nx.stride 0 t)

let test_stride_dim1_2x3_float32 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] (Array.init 6 float_of_int) in
  equal ~msg:"stride dim 1" int 4 (Nx.stride 1 t)

let test_strides_2x3_int64 () =
  let t = Nx.create Nx.int64 [| 2; 3 |] (Array.init 6 Int64.of_int) in
  equal ~msg:"strides int64" (array int) [| 24; 8 |] (Nx.strides t)

let test_itemsize_float32 () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  equal ~msg:"itemsize float32" int 4 (Nx.itemsize t)

let test_itemsize_int64 () =
  let t = Nx.create Nx.int64 [| 2; 2 |] [| 1L; 2L; 3L; 4L |] in
  equal ~msg:"itemsize int64" int 8 (Nx.itemsize t)

let test_ndim_scalar () =
  let t = Nx.scalar Nx.float32 1.0 in
  equal ~msg:"ndim scalar" int 0 (Nx.ndim t)

let test_ndim_2x2 () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  equal ~msg:"ndim 2x2" int 2 (Nx.ndim t)

let test_dim_0_2x3 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  equal ~msg:"dim 0" int 2 (Nx.dim 0 t)

let test_dims_2x3 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] (Array.init 6 float_of_int) in
  equal ~msg:"dims" (array int) [| 2; 3 |] (Nx.dims t)

let test_nbytes_float32 () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  equal ~msg:"nbytes float32" int 16 (Nx.nbytes t)

let test_nbytes_int64 () =
  let t = Nx.create Nx.int64 [| 2; 3 |] (Array.init 6 Int64.of_int) in
  equal ~msg:"nbytes int64" int 48 (Nx.nbytes t)

let test_nbytes_empty () =
  let t = Nx.create Nx.float32 [| 0 |] [||] in
  equal ~msg:"nbytes empty" int 0 (Nx.nbytes t)

let test_size_2x3 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  equal ~msg:"size 2x3" int 6 (Nx.size t)

let test_size_scalar () =
  let t = Nx.scalar Nx.float32 10.0 in
  equal ~msg:"size scalar" int 1 (Nx.size t)

let test_offset_basic () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  equal ~msg:"offset basic" int 0 (Nx.offset t)

let test_offset_slice () =
  let t = Nx.create Nx.float32 [| 3; 3 |] (Array.init 9 float_of_int) in
  let s = Nx.slice [ Nx.R (1, -1); Nx.R (1, -1) ] t in
  equal ~msg:"offset slice" int 4 (Nx.offset s)

(* ───── Element Access And Indexing Tests ───── *)

let test_get_item_2x2 () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  equal ~msg:"get [0,1]" (float 1e-6) 2.0 (Nx.item [ 0; 1 ] t);
  equal ~msg:"get [1,0]" (float 1e-6) 3.0 (Nx.item [ 1; 0 ] t)

let test_set_item_2x2 () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Nx.set_item [ 1; 0 ] 5.0 t;
  equal ~msg:"set [1,0]" (float 1e-6) 5.0 (Nx.item [ 1; 0 ] t)

let test_get_item_out_of_bounds () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  check_invalid_arg "out of bounds get"
    "get: index [2,0] out of bounds for shape [2,2], index 0 at dim 0: 2 not in [0, 2)"
    (fun () -> Nx.item [ 2; 0 ] t)

let test_set_item_out_of_bounds () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  check_invalid_arg "out of bounds set"
    "set: index 2 at dimension 1, out of bounds for shape [2,2], index 1 at dim 1: 2 not in [0, 2)"
    (fun () -> Nx.set_item [ 0; 2 ] 5.0 t)

let test_set_item_type_safety () =
  let t = Nx.create Nx.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  Nx.set_item [ 0; 0 ] 5l t;
  equal ~msg:"set int32" int32 5l (Nx.item [ 0; 0 ] t)

let test_get_scalar_from_0d () =
  let t = Nx.scalar Nx.float32 42.0 in
  equal ~msg:"get scalar" (float 1e-6) 42.0 (Nx.item [] t)

let test_set_scalar_in_0d () =
  let t = Nx.scalar Nx.float32 42.0 in
  Nx.set_item [] 99.0 t;
  equal ~msg:"set scalar" (float 1e-6) 99.0 (Nx.item [] t)

let test_get_view_row () =
  let t = Nx.create Nx.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  let row = Nx.get [ 0 ] t in
  check_t "get row 0" [| 2 |] [| 1l; 2l |] row

let test_get_scalar () =
  let t = Nx.create Nx.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  let scalar = Nx.get [ 1; 1 ] t in
  check_t "get scalar [1,1]" [||] [| 4l |] scalar

let test_set_view_row () =
  let t = Nx.create Nx.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  let v = Nx.create Nx.int32 [| 2 |] [| 8l; 9l |] in
  Nx.set_slice [ Nx.I 0 ] t v;
  check_t "set row 0" [| 2; 2 |] [| 8l; 9l; 3l; 4l |] t

let test_set_scalar () =
  let t = Nx.create Nx.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  Nx.set_item [ 1; 0 ] 99l t;
  check_t "set scalar [1,0]" [| 2; 2 |] [| 1l; 2l; 99l; 4l |] t

(* ───── Slicing Tests ───── *)

let test_slice_3x4 () =
  let t = Nx.create Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let s = Nx.slice [ Nx.R (1, 3); Nx.R (0, 4) ] t in
  check_t "slice [1:3, 0:4]" [| 2; 4 |] [| 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11. |] s

let test_slice_with_steps () =
  let t = Nx.create Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let s = Nx.slice [ Nx.Rs (0, 3, 2); Nx.Rs (0, 4, 2) ] t in
  check_t "slice with steps" [| 2; 2 |] [| 0.; 2.; 8.; 10. |] s

let test_slice_view () =
  let t = Nx.create Nx.float32 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let s = Nx.slice [ Nx.R (1, 2); Nx.R (0, 2) ] t in
  Nx.set_item [ 1; 0 ] 99.0 t;
  equal ~msg:"slice view modified" (float 1e-6) 99.0 (Nx.item [ 0; 0 ] s)

let test_slice_negative_indices () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let sliced = Nx.slice [ Nx.R (-3, -1) ] t in
  check_t "slice negative indices" [| 2 |] [| 3.; 4. |] sliced

let test_slice_empty_range () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let sliced = Nx.slice [ Nx.R (2, 1) ] t in
  check_shape "empty slice shape" [| 0 |] sliced

let test_slice_step_zero () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  check_invalid_arg "slice step zero"
    "slice: step cannot be zero, use positive step for forward slicing or negative for reverse"
    (fun () -> ignore (Nx.slice [ Nx.Rs (0, 5, 0) ] t))

let test_slice_negative_step () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let sliced = Nx.slice [ Nx.Rs (4, 0, -1) ] t in
  check_t "slice negative step" [| 4 |] [| 5.; 4.; 3.; 2. |] sliced

(* ───── Memory And View Tests ───── *)

let test_data_buffer_view () =
  let t = Nx.create Nx.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let d = Nx.data t in
  Nx_buffer.set d 0 99.0;
  equal ~msg:"data buffer view" (float 1e-6) 99.0 (Nx.item [ 0 ] t)

let test_strides_after_transpose () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let original_strides = Nx.strides t in
  let transposed = Nx.transpose t in
  let new_strides = Nx.strides transposed in
  equal ~msg:"transposed strides" (array int)
    [| original_strides.(1); original_strides.(0) |]
    new_strides

let test_strides_after_slice () =
  let t = Nx.create Nx.float32 [| 10 |] (Array.init 10 float_of_int) in
  let sliced = Nx.slice [ Nx.Rs (0, 10, 2) ] t in
  let strides = Nx.strides sliced in
  (* step!=1 slices are materialized via gather and are contiguous *)
  equal ~msg:"slice stride" int 4 strides.(0)

let test_is_c_contiguous_basic () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  equal ~msg:"fresh array is contiguous" bool true (Nx.is_c_contiguous t)

let test_is_c_contiguous_after_transpose () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let transposed = Nx.transpose t in
  equal ~msg:"transposed not contiguous" bool false
    (Nx.is_c_contiguous transposed)

let test_is_c_contiguous_after_slice () =
  let t = Nx.create Nx.float32 [| 10 |] (Array.init 10 float_of_int) in
  (* step!=1 slices are materialized via gather *)
  let sliced = Nx.slice [ Nx.Rs (0, 10, 2) ] t in
  equal ~msg:"slice step=2 is contiguous" bool true (Nx.is_c_contiguous sliced);
  (* step=1 slice is contiguous *)
  let sliced_step1 = Nx.slice [ Nx.Rs (0, 5, 1) ] t in
  equal ~msg:"slice step=1 is contiguous" bool true
    (Nx.is_c_contiguous sliced_step1)

let test_is_c_contiguous_after_double_transpose () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let transposed = Nx.transpose t in
  let restored = Nx.transpose transposed in
  equal ~msg:"transpose twice restores contiguous layout" bool true
    (Nx.is_c_contiguous restored)

let test_offset_after_multiple_slices () =
  let t = Nx.create Nx.float32 [| 5; 5 |] (Array.init 25 float_of_int) in
  let slice1 = Nx.slice [ Nx.R (1, 3); Nx.R (0, 5) ] t in
  let slice2 = Nx.slice [ Nx.R (0, 1); Nx.R (0, 5) ] slice1 in
  equal ~msg:"accumulated offset value" (float 1e-6) 5.0
    (Nx.item [ 0; 0 ] slice2)

(* ───── Utility Operation Tests ───── *)

let test_to_bigarray () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let ba = Nx.to_bigarray t in
  equal ~msg:"initial [0,0]" (float 1e-6) 1.0
    (Bigarray.Genarray.get ba [| 0; 0 |]);
  Nx.set_item [ 0; 0 ] 55.0 t;
  equal ~msg:"after set [0,0]" (float 1e-6) 55.0
    (Bigarray.Genarray.get ba [| 0; 0 |])

let test_to_bigarray_partial_slice () =
  let base = Nx.arange Nx.float32 0 5 1 |> Nx.reshape [| 5; 1 |] in
  let slice = Nx.slice [ Nx.R (0, 4); Nx.I 0 ] base in
  let ba = Nx.to_bigarray slice in
  equal ~msg:"slice dims" (array int) [| 4 |] (Bigarray.Genarray.dims ba);
  let expected = [| 0.0; 1.0; 2.0; 3.0 |] in
  Array.iteri
    (fun i value ->
      equal
        ~msg:(Printf.sprintf "slice[%d]" i)
        (float 1e-6) value
        (Bigarray.Genarray.get ba [| i |]))
    expected

let test_copy () =
  let original = Nx.create Nx.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let copy_arr = Nx.copy original in
  Nx.set_item [ 0 ] 10.0 original;
  equal ~msg:"original [0]" (float 1e-6) 10.0 (Nx.item [ 0 ] original);
  equal ~msg:"copy [0]" (float 1e-6) 1.0 (Nx.item [ 0 ] copy_arr)

let test_blit_incompatible () =
  let src = Nx.create Nx.float32 [| 2 |] [| 1.0; 2.0 |] in
  let dst = Nx.zeros Nx.float32 [| 3 |] in
  raises ~msg:"incompatible shapes"
    (Invalid_argument
       "blit: shape mismatch [2] vs [3], source and destination must have identical shapes")
    (fun () -> Nx.blit src dst)

let test_ifill_nan () =
  let t = Nx.empty Nx.float32 [| 2; 2 |] in
  ignore (Nx.ifill Float.nan t);
  let v = Nx.item [ 0; 0 ] t in
  equal ~msg:"ifill with NaN" bool true (Float.is_nan v)

let test_ifill_inf () =
  let t = Nx.empty Nx.float32 [| 2; 2 |] in
  ignore (Nx.ifill Float.infinity t);
  equal ~msg:"ifill with infinity" (float 1e-6) Float.infinity
    (Nx.item [ 0; 0 ] t);
  ignore (Nx.ifill Float.neg_infinity t);
  equal ~msg:"ifill with neg_infinity" (float 1e-6) Float.neg_infinity
    (Nx.item [ 0; 0 ] t)

let test_fill_returns_copy () =
  let t = Nx.zeros Nx.float32 [| 2; 2 |] in
  let filled = Nx.fill 7.0 t in
  equal ~msg:"fill copy result" (float 1e-6) 7.0 (Nx.item [ 0; 0 ] filled);
  equal ~msg:"fill copy leaves source intact" (float 1e-6) 0.0
    (Nx.item [ 0; 0 ] t)

let test_blit_self () =
  let t = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  Nx.blit t t;
  check_t "blit self" [| 3 |] [| 1.; 2.; 3. |] t

(* TODO: This test is currently failing due to overlapping memory regions in
   blit. See nx/test/failing/bug_blit_overlapping.ml for details. Uncomment when
   overlapping blit is properly handled (e.g., using
   https://github.com/dinosaure/overlap).

   let test_blit_overlapping_views () = let t = Nx.create Nx.float32 [| 5 |] [|
   1.; 2.; 3.; 4.; 5. |] in let view1 = Nx.slice [ Nx.R (0, 3) ] t in let view2
   = Nx.slice [ Nx.R (2, 5) ] t in Nx.blit view1 view2; check_t "blit
   overlapping views" [| 5 |] [| 1.; 2.; 1.; 2.; 3. |] t *)

(* ───── Type Conversion Tests ───── *)

let test_to_array () =
  let t = Nx.create Nx.int32 [| 3 |] [| 1l; 2l; 3l |] in
  let a = Nx.to_array t in
  equal ~msg:"to_array" (array int32) [| 1l; 2l; 3l |] a

let test_astype_float32_to_int32 () =
  let t = Nx.create Nx.float32 [| 3 |] [| 1.1; 2.9; -3.3 |] in
  let u = Nx.astype Nx.int32 t in
  check_t "astype to int32" [| 3 |] [| 1l; 2l; -3l |] u

let test_astype_int32_to_float32 () =
  let t = Nx.create Nx.int32 [| 3 |] [| 1l; 2l; 3l |] in
  let u = Nx.astype Nx.float32 t in
  check_t "astype to float32" [| 3 |] [| 1.0; 2.0; 3.0 |] u

let test_astype_float32_to_int16 () =
  let t = Nx.create Nx.float32 [| 4 |] [| 1.0; 2.5; 3.9; 255.0 |] in
  let u = Nx.astype Nx.int16 t in
  check_t "astype to int16" [| 4 |] [| 1; 2; 3; 255 |] u

let test_astype_int64_to_float32 () =
  let t = Nx.create Nx.int64 [| 3 |] [| 1000L; 2000L; 3000L |] in
  let u = Nx.astype Nx.float32 t in
  check_t "astype int64 to float32" [| 3 |] [| 1000.0; 2000.0; 3000.0 |] u

(* Test Suite Organization *)

let creation_edge_cases =
  [
    test "create 1D int32" test_create_1d_int32;
    test "create empty float32" test_create_empty_float32;
    test "create 2x2x2 float32" test_create_2x2x2_float32;
    test "scalar float32" test_scalar_float32;
    test "scalar int64" test_scalar_int64;
    test "create int16" test_create_int16;
    test "create empty shapes" test_create_empty_shapes;
    test "create max rank" test_create_max_rank;
    test "create wrong data size" test_create_wrong_data_size;
    test "create negative shape" test_create_negative_shape;
  ]

let special_creation =
  [
    test "empty float32" test_empty_float32;
    test "full float32" test_full_float32;
    test "full_like int32" test_full_like_int32;
    test "empty_like float32" test_empty_like_float32;
    test "zeros_like float32" test_zeros_like_float32;
    test "ones_like int32" test_ones_like_int32;
    test "zeros max size" test_zeros_max_size;
  ]

let eye_identity_tests =
  [
    test "identity 1x1 int32" test_identity_1x1_int32;
    test "eye 3x4 float32" test_eye_3x4_float32;
    test "eye 4x3 k=1 float32" test_eye_4x3_k1_float32;
    test "eye 3x3 k=-1 int32" test_eye_3x3_km1_int32;
    test "eye 0x0" test_eye_0x0;
    test "eye k out of range" test_eye_k_out_of_range;
    test "diag extract" test_diag_extract;
    test "diag construct" test_diag_construct;
  ]

let range_generation =
  [
    test "arange empty" test_arange_empty;
    test "arange negative step" test_arange_negative_step;
    test "arange wrong direction" test_arange_wrong_direction;
    test "linspace no endpoint float32" test_linspace_no_endpoint_float32;
    test "linspace single point" test_linspace_single_point;
    test "linspace zero points" test_linspace_zero_points;
    test "logspace base 10 float32" test_logspace_base10_float32;
    test "logspace base 2 no endpoint float32"
      test_logspace_base2_no_endpoint_float32;
    test "geomspace no endpoint float32" test_geomspace_no_endpoint_float32;
  ]

let property_access =
  [
    test "shape 2x3" test_shape_2x3;
    test "strides 2x3 float32" test_strides_2x3_float32;
    test "stride dim 0 2x3 float32" test_stride_dim0_2x3_float32;
    test "stride dim 1 2x3 float32" test_stride_dim1_2x3_float32;
    test "strides 2x3 int64" test_strides_2x3_int64;
    test "itemsize float32" test_itemsize_float32;
    test "itemsize int64" test_itemsize_int64;
    test "ndim scalar" test_ndim_scalar;
    test "ndim 2x2" test_ndim_2x2;
    test "dim 0 2x3" test_dim_0_2x3;
    test "dims 2x3" test_dims_2x3;
    test "nbytes float32" test_nbytes_float32;
    test "nbytes int64" test_nbytes_int64;
    test "nbytes empty" test_nbytes_empty;
    test "size 2x3" test_size_2x3;
    test "size scalar" test_size_scalar;
    test "offset basic" test_offset_basic;
    test "offset slice" test_offset_slice;
  ]

let element_access_indexing =
  [
    test "get item 2x2" test_get_item_2x2;
    test "set item 2x2" test_set_item_2x2;
    test "get item out of bounds" test_get_item_out_of_bounds;
    test "set item out of bounds" test_set_item_out_of_bounds;
    test "set item type safety" test_set_item_type_safety;
    test "get scalar from 0d" test_get_scalar_from_0d;
    test "set scalar in 0d" test_set_scalar_in_0d;
    test "get view row" test_get_view_row;
    test "get scalar" test_get_scalar;
    test "set view row" test_set_view_row;
    test "set scalar" test_set_scalar;
  ]

let slicing =
  [
    test "slice 3x4" test_slice_3x4;
    test "slice with steps" test_slice_with_steps;
    test "slice view" test_slice_view;
    test "slice negative indices" test_slice_negative_indices;
    test "slice empty range" test_slice_empty_range;
    test "slice step zero" test_slice_step_zero;
    test "slice negative step" test_slice_negative_step;
  ]

let memory_and_views =
  [
    test "data buffer view" test_data_buffer_view;
    test "strides after transpose" test_strides_after_transpose;
    test "strides after slice" test_strides_after_slice;
    test "is contiguous basic" test_is_c_contiguous_basic;
    test "is contiguous after transpose" test_is_c_contiguous_after_transpose;
    test "is contiguous after slice" test_is_c_contiguous_after_slice;
    test "is contiguous after double transpose"
      test_is_c_contiguous_after_double_transpose;
    test "offset after multiple slices" test_offset_after_multiple_slices;
  ]

let utility_operations =
  [
    test "to bigarray" test_to_bigarray;
    test "to bigarray partial slice" test_to_bigarray_partial_slice;
    test "copy" test_copy;
    test "blit incompatible" test_blit_incompatible;
    test "ifill nan" test_ifill_nan;
    test "ifill inf" test_ifill_inf;
    test "fill returns copy" test_fill_returns_copy;
    test "blit self" test_blit_self;
    (* ("blit overlapping views", `Quick, test_blit_overlapping_views ); *)
  ]

let type_conversion =
  [
    test "to array" test_to_array;
    test "astype float32 to int32" test_astype_float32_to_int32;
    test "astype int32 to float32" test_astype_int32_to_float32;
    test "astype float32 to int16" test_astype_float32_to_int16;
    test "astype int64 to float32" test_astype_int64_to_float32;
  ]

let suite =
  [
    group "Creation Edge Cases" creation_edge_cases;
    group "Special Creation Functions" special_creation;
    group "Eye and Identity" eye_identity_tests;
    group "Range Generation" range_generation;
    group "Property Access" property_access;
    group "Element Access and Indexing" element_access_indexing;
    group "Slicing" slicing;
    group "Memory and Views" memory_and_views;
    group "Utility Operations" utility_operations;
    group "Type Conversion" type_conversion;
  ]

let () = run "Nx Basics" suite
