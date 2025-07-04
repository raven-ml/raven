(* Core functionality tests for Nx - creation, indexing, properties *)

open Alcotest

module Make (Backend : Nx_core.Backend_intf.S) = struct
  module Support = Test_nx_support.Make (Backend)
  module Nx = Support.Nx
  open Support

  (* ───── Creation Edge Case Tests ───── *)

  let test_create_1d_int32 ctx () =
    let t = Nx.create ctx Nx_core.Dtype.int32 [| 3 |] [| 1l; 2l; 3l |] in
    check_t "create 1D int32" [| 3 |] [| 1l; 2l; 3l |] t

  let test_create_empty_float32 ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 0 |] [||] in
    check_shape "empty shape" [| 0 |] t

  let test_create_2x2x2_float32 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2; 2 |]
        (Array.init 8 float_of_int)
    in
    check_t "create 2x2x2" [| 2; 2; 2 |] [| 0.; 1.; 2.; 3.; 4.; 5.; 6.; 7. |] t

  let test_scalar_float32 ctx () =
    let t = Nx.scalar ctx Nx_core.Dtype.float32 42.0 in
    check_t "scalar float32" [||] [| 42.0 |] t

  let test_scalar_int64 ctx () =
    let t = Nx.scalar ctx Nx_core.Dtype.int64 100L in
    check_t "scalar int64" [||] [| 100L |] t

  let test_create_int16 ctx () =
    let t = Nx.create ctx Nx_core.Dtype.int16 [| 4 |] [| 1; 2; 3; 4 |] in
    check_t "create int16" [| 4 |] [| 1; 2; 3; 4 |] t

  let test_create_empty_shapes ctx () =
    (* Empty 1D array *)
    let t1 = Nx.create ctx Nx_core.Dtype.float32 [| 0 |] [||] in
    check_shape "empty 1D shape" [| 0 |] t1;

    (* Empty multi-dimensional arrays *)
    let t2 = Nx.create ctx Nx_core.Dtype.float32 [| 0; 5 |] [||] in
    check_shape "empty 2D shape [0,5]" [| 0; 5 |] t2;

    let t3 = Nx.create ctx Nx_core.Dtype.float32 [| 5; 0; 3 |] [||] in
    check_shape "empty 3D shape [5,0,3]" [| 5; 0; 3 |] t3

  let test_create_max_rank ctx () =
    (* Create array with many dimensions but small total size *)
    (* Use shape like [1, 1, 1, ..., 2, 2, 2] to keep total size manageable *)
    let shape = Array.init 32 (fun i -> if i < 29 then 1 else 2) in
    let data_size = Array.fold_left ( * ) 1 shape in
    (* = 8 total elements *)
    let data = Array.init data_size float_of_int in
    let t = Nx.create ctx Nx_core.Dtype.float32 shape data in
    check int "ndim of 32D array" 32 (Nx.ndim t);
    check_shape "32D shape" shape t

  let test_create_wrong_data_size ctx () =
    check_invalid_arg "data size mismatch"
      "create: invalid array size (got 3 elements, expected 6)" (fun () ->
        ignore
          (Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0 |]))

  let test_create_negative_shape ctx () =
    check_invalid_arg "negative dimension"
      "create: invalid array size (got 2 elements, expected -6)" (fun () ->
        ignore (Nx.create ctx Nx_core.Dtype.float32 [| 2; -3 |] [| 1.0; 2.0 |]))

  (* ───── Special Creation Function Tests ───── *)

  let test_empty_float32 ctx () =
    let t = Nx.empty ctx Nx_core.Dtype.float32 [| 2; 2 |] in
    check_shape "empty shape" [| 2; 2 |] t

  let test_full_float32 ctx () =
    let t = Nx.full ctx Nx_core.Dtype.float32 [| 2; 3 |] 5.5 in
    check_t "full" [| 2; 3 |] [| 5.5; 5.5; 5.5; 5.5; 5.5; 5.5 |] t

  let test_full_like_int32 ctx () =
    let ref_t =
      Nx.create ctx Nx_core.Dtype.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |]
    in
    let t = Nx.full_like ref_t 10l in
    check_t "full_like" [| 2; 2 |] [| 10l; 10l; 10l; 10l |] t

  let test_empty_like_float32 ctx () =
    let ref_t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    let t = Nx.empty_like ref_t in
    check_shape "empty_like shape" [| 2; 2 |] t

  let test_zeros_like_float32 ctx () =
    let ref_t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    let t = Nx.zeros_like ref_t in
    check_t "zeros_like" [| 2; 2 |] [| 0.; 0.; 0.; 0. |] t

  let test_ones_like_int32 ctx () =
    let ref_t =
      Nx.create ctx Nx_core.Dtype.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |]
    in
    let t = Nx.ones_like ref_t in
    check_t "ones_like" [| 2; 2 |] [| 1l; 1l; 1l; 1l |] t

  let test_zeros_max_size ctx () =
    let t = Nx.zeros ctx Nx_core.Dtype.float32 [| 256; 256; 16 |] in
    check_shape "large zeros shape" [| 256; 256; 16 |] t;
    check (float 1e-6) "zeros[0,0,0]" 0.0 (Nx.unsafe_get [ 0; 0; 0 ] t)

  (* ───── Eye Identity Tests ───── *)

  let test_identity_1x1_int32 ctx () =
    let t = Nx.identity ctx Nx_core.Dtype.int32 1 in
    check_t "identity 1x1" [| 1; 1 |] [| 1l |] t

  let test_eye_3x4_float32 ctx () =
    let t = Nx.eye ctx ~m:3 Nx_core.Dtype.float32 4 in
    check_t "eye 3x4" [| 3; 4 |]
      [| 1.; 0.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 0.; 1.; 0. |]
      t

  let test_eye_4x3_k1_float32 ctx () =
    let t = Nx.eye ctx ~m:4 ~k:1 Nx_core.Dtype.float32 3 in
    check_t "eye 4x3 k=1" [| 4; 3 |]
      [| 0.; 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 0.; 0.; 0. |]
      t

  let test_eye_3x3_km1_int32 ctx () =
    let t = Nx.eye ctx ~k:(-1) Nx_core.Dtype.int32 3 in
    check_t "eye 3x3 k=-1" [| 3; 3 |] [| 0l; 0l; 0l; 1l; 0l; 0l; 0l; 1l; 0l |] t

  let test_eye_0x0 ctx () =
    let t = Nx.eye ctx Nx_core.Dtype.float32 0 in
    check_shape "0x0 eye shape" [| 0; 0 |] t

  let test_eye_k_out_of_range ctx () =
    (* k offset larger than matrix dimensions *)
    let t1 = Nx.eye ctx Nx_core.Dtype.float32 ~k:10 3 in
    check_t "eye with k=10" [| 3; 3 |]
      [| 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0. |]
      t1;

    let t2 = Nx.eye ctx Nx_core.Dtype.float32 ~k:(-10) 3 in
    check_t "eye with k=-10" [| 3; 3 |]
      [| 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0. |]
      t2

  (* ───── Range Generation Tests ───── *)

  let test_arange_empty ctx () =
    check_invalid_arg "arange empty"
      "arange: invalid range [0, 0) (empty with step=1)\n\
       hint: ensure start < stop for positive step, or start > stop for \
       negative step" (fun () -> Nx.arange ctx Nx_core.Dtype.int32 0 0 1)

  let test_arange_negative_step ctx () =
    let t = Nx.arange ctx Nx_core.Dtype.int32 10 0 (-2) in
    check_t "arange negative step" [| 5 |] [| 10l; 8l; 6l; 4l; 2l |] t

  let test_arange_wrong_direction ctx () =
    let t = Nx.arange ctx Nx_core.Dtype.int32 0 10 (-1) in
    check_shape "arange wrong direction shape" [| 0 |] t

  let test_linspace_no_endpoint_float32 ctx () =
    let t = Nx.linspace ctx ~endpoint:false Nx_core.Dtype.float32 0.0 4.0 5 in
    check_t ~eps:1e-6 "linspace no endpoint" [| 5 |]
      [| 0.0; 0.8; 1.6; 2.4; 3.2 |]
      t

  let test_linspace_single_point ctx () =
    let t = Nx.linspace ctx Nx_core.Dtype.float32 5.0 5.0 1 in
    check_t "linspace single point" [| 1 |] [| 5.0 |] t

  let test_linspace_zero_points ctx () =
    (* linspace with 0 points returns empty array, doesn't raise error *)
    let t = Nx.linspace ctx Nx_core.Dtype.float32 0.0 1.0 0 in
    check_shape "linspace 0 points" [| 0 |] t

  let test_logspace_base10_float32 ctx () =
    let t = Nx.logspace ctx ~base:10.0 Nx_core.Dtype.float32 2.0 3.0 4 in
    check_t ~eps:1e-3 "logspace base 10" [| 4 |]
      [| 100.0; 215.443469003188454; 464.158883361277731; 1000.0 |]
      t

  let test_logspace_base2_no_endpoint_float32 ctx () =
    let t =
      Nx.logspace ctx ~endpoint:false ~base:2.0 Nx_core.Dtype.float32 0.0 4.0 5
    in
    check_t ~eps:1e-6 "logspace base 2 no endpoint" [| 5 |]
      [|
        1.0;
        1.741101126592248;
        3.031433133020796;
        5.278031643091579;
        9.189586839976281;
      |]
      t

  let test_geomspace_no_endpoint_float32 ctx () =
    let t =
      Nx.geomspace ctx ~endpoint:false Nx_core.Dtype.float32 1.0 256.0 9
    in
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

  let test_shape_2x3 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |] (Array.init 6 float_of_int)
    in
    check_shape "shape 2x3" [| 2; 3 |] t

  let test_strides_2x3_float32 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |] (Array.init 6 float_of_int)
    in
    check (array int) "strides" [| 12; 4 |] (Nx.strides t)

  let test_stride_dim0_2x3_float32 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |] (Array.init 6 float_of_int)
    in
    check int "stride dim 0" 12 (Nx.stride 0 t)

  let test_stride_dim1_2x3_float32 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |] (Array.init 6 float_of_int)
    in
    check int "stride dim 1" 4 (Nx.stride 1 t)

  let test_strides_2x3_int64 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.int64 [| 2; 3 |] (Array.init 6 Int64.of_int)
    in
    check (array int) "strides int64" [| 24; 8 |] (Nx.strides t)

  let test_itemsize_float32 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    check int "itemsize float32" 4 (Nx.itemsize t)

  let test_itemsize_int64 ctx () =
    let t = Nx.create ctx Nx_core.Dtype.int64 [| 2; 2 |] [| 1L; 2L; 3L; 4L |] in
    check int "itemsize int64" 8 (Nx.itemsize t)

  let test_ndim_scalar ctx () =
    let t = Nx.scalar ctx Nx_core.Dtype.float32 1.0 in
    check int "ndim scalar" 0 (Nx.ndim t)

  let test_ndim_2x2 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    check int "ndim 2x2" 2 (Nx.ndim t)

  let test_dim_0_2x3 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    in
    check int "dim 0" 2 (Nx.dim 0 t)

  let test_dims_2x3 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |] (Array.init 6 float_of_int)
    in
    check (array int) "dims" [| 2; 3 |] (Nx.dims t)

  let test_nbytes_float32 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    check int "nbytes float32" 16 (Nx.nbytes t)

  let test_nbytes_int64 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.int64 [| 2; 3 |] (Array.init 6 Int64.of_int)
    in
    check int "nbytes int64" 48 (Nx.nbytes t)

  let test_nbytes_empty ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 0 |] [||] in
    check int "nbytes empty" 0 (Nx.nbytes t)

  let test_size_2x3 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    in
    check int "size 2x3" 6 (Nx.size t)

  let test_size_scalar ctx () =
    let t = Nx.scalar ctx Nx_core.Dtype.float32 10.0 in
    check int "size scalar" 1 (Nx.size t)

  let test_offset_basic ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    check int "offset basic" 0 (Nx.offset t)

  let test_offset_slice ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 3; 3 |] (Array.init 9 float_of_int)
    in
    let s = Nx.slice [ R [ 1; -1 ]; R [ 1; -1 ] ] t in
    check int "offset slice" 4 (Nx.offset s)

  (* ───── Element Access And Indexing Tests ───── *)

  let test_get_item_2x2 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    check (float 1e-6) "get [0,1]" 2.0 (Nx.unsafe_get [ 0; 1 ] t);
    check (float 1e-6) "get [1,0]" 3.0 (Nx.unsafe_get [ 1; 0 ] t)

  let test_set_item_2x2 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    Nx.unsafe_set [ 1; 0 ] 5.0 t;
    check (float 1e-6) "set [1,0]" 5.0 (Nx.unsafe_get [ 1; 0 ] t)

  let test_get_item_out_of_bounds ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    check_invalid_arg "out of bounds get"
      "get: invalid index [2,0] (out of bounds for shape [2,2])\n\
       hint: index 0 at dim 0: 2 \226\136\137 [0, 2)" (fun () ->
        Nx.unsafe_get [ 2; 0 ] t)

  let test_set_item_out_of_bounds ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    check_invalid_arg "out of bounds set"
      "set: invalid index 2 at dimension 1 (out of bounds for shape [2,2])\n\
       hint: index 1 at dim 1: 2 \226\136\137 [0, 2)" (fun () ->
        Nx.unsafe_set [ 0; 2 ] 5.0 t)

  let test_set_item_type_safety ctx () =
    let t = Nx.create ctx Nx_core.Dtype.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
    Nx.unsafe_set [ 0; 0 ] 5l t;
    check int32 "set int32" 5l (Nx.unsafe_get [ 0; 0 ] t)

  let test_get_scalar_from_0d ctx () =
    let t = Nx.scalar ctx Nx_core.Dtype.float32 42.0 in
    check (float 1e-6) "get scalar" 42.0 (Nx.unsafe_get [] t)

  let test_set_scalar_in_0d ctx () =
    let t = Nx.scalar ctx Nx_core.Dtype.float32 42.0 in
    Nx.unsafe_set [] 99.0 t;
    check (float 1e-6) "set scalar" 99.0 (Nx.unsafe_get [] t)

  let test_get_view_row ctx () =
    let t = Nx.create ctx Nx_core.Dtype.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
    let row = Nx.get [ 0 ] t in
    check_t "get row 0" [| 2 |] [| 1l; 2l |] row

  let test_get_scalar ctx () =
    let t = Nx.create ctx Nx_core.Dtype.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
    let scalar = Nx.get [ 1; 1 ] t in
    check_t "get scalar [1,1]" [||] [| 4l |] scalar

  let test_set_view_row ctx () =
    let t = Nx.create ctx Nx_core.Dtype.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
    let v = Nx.create ctx Nx_core.Dtype.int32 [| 2 |] [| 8l; 9l |] in
    Nx.set_slice [ I 0 ] t v;
    check_t "set row 0" [| 2; 2 |] [| 8l; 9l; 3l; 4l |] t

  let test_set_scalar ctx () =
    let t = Nx.create ctx Nx_core.Dtype.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
    Nx.unsafe_set [ 1; 0 ] 99l t;
    check_t "set scalar [1,0]" [| 2; 2 |] [| 1l; 2l; 99l; 4l |] t

  (* ───── Slicing Tests ───── *)

  let test_slice_3x4 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 3; 4 |]
        (Array.init 12 float_of_int)
    in
    let s = Nx.slice_ranges [ 1; 0 ] [ 3; 4 ] t in
    check_t "slice [1:3, 0:4]" [| 2; 4 |]
      [| 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11. |]
      s

  let test_slice_with_steps ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 3; 4 |]
        (Array.init 12 float_of_int)
    in
    let s = Nx.slice_ranges ~steps:[ 2; 2 ] [ 0; 0 ] [ 3; 4 ] t in
    check_t "slice with steps" [| 2; 2 |] [| 0.; 2.; 8.; 10. |] s

  let test_slice_view ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 3; 2 |]
        [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    in
    let s = Nx.slice_ranges [ 1; 0 ] [ 2; 2 ] t in
    Nx.unsafe_set [ 1; 0 ] 99.0 t;
    check (float 1e-6) "slice view modified" 99.0 (Nx.unsafe_get [ 0; 0 ] s)

  let test_slice_negative_indices ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |]
    in
    let sliced = Nx.slice [ R [ -3; -1 ] ] t in
    check_t "slice negative indices" [| 2 |] [| 3.; 4. |] sliced

  let test_slice_empty_range ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |]
    in
    let sliced = Nx.slice [ R [ 2; 1 ] ] t in
    check_shape "empty slice shape" [| 0 |] sliced

  let test_slice_step_zero ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |]
    in
    check_invalid_arg "slice step zero"
      "slice: invalid step (cannot be zero)\n\
       hint: use positive step for forward slicing or negative for reverse"
      (fun () -> ignore (Nx.slice_ranges ~steps:[ 0 ] [ 0 ] [ 5 ] t))

  let test_slice_negative_step ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |]
    in
    let sliced = Nx.slice_ranges ~steps:[ -1 ] [ 4 ] [ 0 ] t in
    check_t "slice negative step" [| 4 |] [| 5.; 4.; 3.; 2. |] sliced

  (* ───── Memory And View Tests ───── *)

  let test_data_buffer_view ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
    let d = Nx.unsafe_data t in
    Bigarray.Array1.set d 0 99.0;
    check (float 1e-6) "data buffer view" 99.0 (Nx.unsafe_get [ 0 ] t)

  let test_strides_after_transpose ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let original_strides = Nx.strides t in
    let transposed = Nx.transpose t in
    let new_strides = Nx.strides transposed in
    check (array int) "transposed strides"
      [| original_strides.(1); original_strides.(0) |]
      new_strides

  let test_strides_after_slice ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 10 |] (Array.init 10 float_of_int)
    in
    let sliced = Nx.slice_ranges ~steps:[ 2 ] [ 0 ] [ 10 ] t in
    let strides = Nx.strides sliced in
    check int "slice stride" 4 strides.(0)

  let test_is_c_contiguous_basic ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    check bool "fresh array is contiguous" true (Nx.is_c_contiguous t)

  let test_is_c_contiguous_after_transpose ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let transposed = Nx.transpose t in
    check bool "transposed not contiguous" false (Nx.is_c_contiguous transposed)

  let test_is_c_contiguous_after_slice ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 10 |] (Array.init 10 float_of_int)
    in
    let sliced = Nx.slice_ranges ~steps:[ 2 ] [ 0 ] [ 10 ] t in
    check bool "slice step=2 is contiguous" true (Nx.is_c_contiguous sliced)

  let test_offset_after_multiple_slices ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 5; 5 |]
        (Array.init 25 float_of_int)
    in
    let slice1 = Nx.slice_ranges [ 1; 0 ] [ 3; 5 ] t in
    let slice2 = Nx.slice_ranges [ 0; 0 ] [ 1; 5 ] slice1 in
    check (float 1e-6) "accumulated offset value" 5.0
      (Nx.unsafe_get [ 0; 0 ] slice2)

  (* ───── Utility Operation Tests ───── *)

  let test_to_bigarray ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    let ba = Nx.unsafe_to_bigarray t in
    check (float 1e-6) "initial [0,0]" 1.0 (Bigarray.Genarray.get ba [| 0; 0 |]);
    Nx.unsafe_set [ 0; 0 ] 55.0 t;
    check (float 1e-6) "after set [0,0]" 55.0
      (Bigarray.Genarray.get ba [| 0; 0 |])

  let test_copy ctx () =
    let original =
      Nx.create ctx Nx_core.Dtype.float32 [| 3 |] [| 1.0; 2.0; 3.0 |]
    in
    let copy_arr = Nx.copy original in
    Nx.unsafe_set [ 0 ] 10.0 original;
    check (float 1e-6) "original [0]" 10.0 (Nx.unsafe_get [ 0 ] original);
    check (float 1e-6) "copy [0]" 1.0 (Nx.unsafe_get [ 0 ] copy_arr)

  let test_blit_incompatible ctx () =
    let src = Nx.create ctx Nx_core.Dtype.float32 [| 2 |] [| 1.0; 2.0 |] in
    let dst = Nx.zeros ctx Nx_core.Dtype.float32 [| 3 |] in
    check_raises "incompatible shapes"
      (Invalid_argument
         "blit: cannot reshape [3] to [2] (dim 0: 3≠2)\n\
          hint: source and destination must have identical shapes") (fun () ->
        Nx.blit src dst)

  let test_fill_nan ctx () =
    let t = Nx.empty ctx Nx_core.Dtype.float32 [| 2; 2 |] in
    ignore (Nx.fill Float.nan t);
    let v = Nx.unsafe_get [ 0; 0 ] t in
    check bool "fill with NaN" true (Float.is_nan v)

  let test_fill_inf ctx () =
    let t = Nx.empty ctx Nx_core.Dtype.float32 [| 2; 2 |] in
    ignore (Nx.fill Float.infinity t);
    check (float 1e-6) "fill with infinity" Float.infinity
      (Nx.unsafe_get [ 0; 0 ] t);
    ignore (Nx.fill Float.neg_infinity t);
    check (float 1e-6) "fill with neg_infinity" Float.neg_infinity
      (Nx.unsafe_get [ 0; 0 ] t)

  let test_blit_self ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 3 |] [| 1.; 2.; 3. |] in
    Nx.blit t t;
    check_t "blit self" [| 3 |] [| 1.; 2.; 3. |] t

  (* TODO: This test is currently failing due to overlapping memory regions in
     blit. See nx/test/failing/bug_blit_overlapping.ml for details. Uncomment
     when overlapping blit is properly handled (e.g., using
     https://github.com/dinosaure/overlap).

     let test_blit_overlapping_views ctx () = let t = Nx.create ctx
     Nx_core.Dtype.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in let view1 =
     Nx.slice [ R [ 0; 3 ] ] t in let view2 = Nx.slice [ R [ 2; 5 ] ] t in
     Nx.blit view1 view2; check_t "blit overlapping views" [| 5 |] [| 1.; 2.;
     1.; 2.; 3. |] t *)

  (* ───── Type Conversion Tests ───── *)

  let test_to_array ctx () =
    let t = Nx.create ctx Nx_core.Dtype.int32 [| 3 |] [| 1l; 2l; 3l |] in
    let a = Nx.unsafe_to_array t in
    check (array int32) "to_array" [| 1l; 2l; 3l |] a

  let test_astype_float32_to_int32 ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 3 |] [| 1.1; 2.9; -3.3 |] in
    let u = Nx.astype Nx_core.Dtype.int32 t in
    check_t "astype to int32" [| 3 |] [| 1l; 2l; -3l |] u

  let test_astype_int32_to_float32 ctx () =
    let t = Nx.create ctx Nx_core.Dtype.int32 [| 3 |] [| 1l; 2l; 3l |] in
    let u = Nx.astype Nx_core.Dtype.float32 t in
    check_t "astype to float32" [| 3 |] [| 1.0; 2.0; 3.0 |] u

  let test_astype_float32_to_int16 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 4 |] [| 1.0; 2.5; 3.9; 255.0 |]
    in
    let u = Nx.astype Nx_core.Dtype.int16 t in
    check_t "astype to int16" [| 4 |] [| 1; 2; 3; 255 |] u

  let test_astype_int64_to_float32 ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.int64 [| 3 |] [| 1000L; 2000L; 3000L |]
    in
    let u = Nx.astype Nx_core.Dtype.float32 t in
    check_t "astype int64 to float32" [| 3 |] [| 1000.0; 2000.0; 3000.0 |] u

  (* Test Suite Organization *)

  let creation_edge_cases ctx =
    [
      ("create 1D int32", `Quick, test_create_1d_int32 ctx);
      ("create empty float32", `Quick, test_create_empty_float32 ctx);
      ("create 2x2x2 float32", `Quick, test_create_2x2x2_float32 ctx);
      ("scalar float32", `Quick, test_scalar_float32 ctx);
      ("scalar int64", `Quick, test_scalar_int64 ctx);
      ("create int16", `Quick, test_create_int16 ctx);
      ("create empty shapes", `Quick, test_create_empty_shapes ctx);
      ("create max rank", `Quick, test_create_max_rank ctx);
      ("create wrong data size", `Quick, test_create_wrong_data_size ctx);
      ("create negative shape", `Quick, test_create_negative_shape ctx);
    ]

  let special_creation ctx =
    [
      ("empty float32", `Quick, test_empty_float32 ctx);
      ("full float32", `Quick, test_full_float32 ctx);
      ("full_like int32", `Quick, test_full_like_int32 ctx);
      ("empty_like float32", `Quick, test_empty_like_float32 ctx);
      ("zeros_like float32", `Quick, test_zeros_like_float32 ctx);
      ("ones_like int32", `Quick, test_ones_like_int32 ctx);
      ("zeros max size", `Quick, test_zeros_max_size ctx);
    ]

  let eye_identity_tests ctx =
    [
      ("identity 1x1 int32", `Quick, test_identity_1x1_int32 ctx);
      ("eye 3x4 float32", `Quick, test_eye_3x4_float32 ctx);
      ("eye 4x3 k=1 float32", `Quick, test_eye_4x3_k1_float32 ctx);
      ("eye 3x3 k=-1 int32", `Quick, test_eye_3x3_km1_int32 ctx);
      ("eye 0x0", `Quick, test_eye_0x0 ctx);
      ("eye k out of range", `Quick, test_eye_k_out_of_range ctx);
    ]

  let range_generation ctx =
    [
      ("arange empty", `Quick, test_arange_empty ctx);
      ("arange negative step", `Quick, test_arange_negative_step ctx);
      ("arange wrong direction", `Quick, test_arange_wrong_direction ctx);
      ( "linspace no endpoint float32",
        `Quick,
        test_linspace_no_endpoint_float32 ctx );
      ("linspace single point", `Quick, test_linspace_single_point ctx);
      ("linspace zero points", `Quick, test_linspace_zero_points ctx);
      ("logspace base 10 float32", `Quick, test_logspace_base10_float32 ctx);
      ( "logspace base 2 no endpoint float32",
        `Quick,
        test_logspace_base2_no_endpoint_float32 ctx );
      ( "geomspace no endpoint float32",
        `Quick,
        test_geomspace_no_endpoint_float32 ctx );
    ]

  let property_access ctx =
    [
      ("shape 2x3", `Quick, test_shape_2x3 ctx);
      ("strides 2x3 float32", `Quick, test_strides_2x3_float32 ctx);
      ("stride dim 0 2x3 float32", `Quick, test_stride_dim0_2x3_float32 ctx);
      ("stride dim 1 2x3 float32", `Quick, test_stride_dim1_2x3_float32 ctx);
      ("strides 2x3 int64", `Quick, test_strides_2x3_int64 ctx);
      ("itemsize float32", `Quick, test_itemsize_float32 ctx);
      ("itemsize int64", `Quick, test_itemsize_int64 ctx);
      ("ndim scalar", `Quick, test_ndim_scalar ctx);
      ("ndim 2x2", `Quick, test_ndim_2x2 ctx);
      ("dim 0 2x3", `Quick, test_dim_0_2x3 ctx);
      ("dims 2x3", `Quick, test_dims_2x3 ctx);
      ("nbytes float32", `Quick, test_nbytes_float32 ctx);
      ("nbytes int64", `Quick, test_nbytes_int64 ctx);
      ("nbytes empty", `Quick, test_nbytes_empty ctx);
      ("size 2x3", `Quick, test_size_2x3 ctx);
      ("size scalar", `Quick, test_size_scalar ctx);
      ("offset basic", `Quick, test_offset_basic ctx);
      ("offset slice", `Quick, test_offset_slice ctx);
    ]

  let element_access_indexing ctx =
    [
      ("get item 2x2", `Quick, test_get_item_2x2 ctx);
      ("set item 2x2", `Quick, test_set_item_2x2 ctx);
      ("get item out of bounds", `Quick, test_get_item_out_of_bounds ctx);
      ("set item out of bounds", `Quick, test_set_item_out_of_bounds ctx);
      ("set item type safety", `Quick, test_set_item_type_safety ctx);
      ("get scalar from 0d", `Quick, test_get_scalar_from_0d ctx);
      ("set scalar in 0d", `Quick, test_set_scalar_in_0d ctx);
      ("get view row", `Quick, test_get_view_row ctx);
      ("get scalar", `Quick, test_get_scalar ctx);
      ("set view row", `Quick, test_set_view_row ctx);
      ("set scalar", `Quick, test_set_scalar ctx);
    ]

  let slicing ctx =
    [
      ("slice 3x4", `Quick, test_slice_3x4 ctx);
      ("slice with steps", `Quick, test_slice_with_steps ctx);
      ("slice view", `Quick, test_slice_view ctx);
      ("slice negative indices", `Quick, test_slice_negative_indices ctx);
      ("slice empty range", `Quick, test_slice_empty_range ctx);
      ("slice step zero", `Quick, test_slice_step_zero ctx);
      ("slice negative step", `Quick, test_slice_negative_step ctx);
    ]

  let memory_and_views ctx =
    [
      ("data buffer view", `Quick, test_data_buffer_view ctx);
      ("strides after transpose", `Quick, test_strides_after_transpose ctx);
      ("strides after slice", `Quick, test_strides_after_slice ctx);
      ("is contiguous basic", `Quick, test_is_c_contiguous_basic ctx);
      ( "is contiguous after transpose",
        `Quick,
        test_is_c_contiguous_after_transpose ctx );
      ("is contiguous after slice", `Quick, test_is_c_contiguous_after_slice ctx);
      ( "offset after multiple slices",
        `Quick,
        test_offset_after_multiple_slices ctx );
    ]

  let utility_operations ctx =
    [
      ("to bigarray", `Quick, test_to_bigarray ctx);
      ("copy", `Quick, test_copy ctx);
      ("blit incompatible", `Quick, test_blit_incompatible ctx);
      ("fill nan", `Quick, test_fill_nan ctx);
      ("fill inf", `Quick, test_fill_inf ctx);
      ("blit self", `Quick, test_blit_self ctx);
      (* ("blit overlapping views", `Quick, test_blit_overlapping_views ctx); *)
    ]

  let type_conversion ctx =
    [
      ("to array", `Quick, test_to_array ctx);
      ("astype float32 to int32", `Quick, test_astype_float32_to_int32 ctx);
      ("astype int32 to float32", `Quick, test_astype_int32_to_float32 ctx);
      ("astype float32 to int16", `Quick, test_astype_float32_to_int16 ctx);
      ("astype int64 to float32", `Quick, test_astype_int64_to_float32 ctx);
    ]

  let suite backend_name ctx =
    [
      ( "Basic :: " ^ backend_name ^ " Creation Edge Cases",
        creation_edge_cases ctx );
      ( "Basic :: " ^ backend_name ^ " Special Creation Functions",
        special_creation ctx );
      ("Basic :: " ^ backend_name ^ " Eye and Identity", eye_identity_tests ctx);
      ("Basic :: " ^ backend_name ^ " Range Generation", range_generation ctx);
      ("Basic :: " ^ backend_name ^ " Property Access", property_access ctx);
      ( "Basic :: " ^ backend_name ^ " Element Access and Indexing",
        element_access_indexing ctx );
      ("Basic :: " ^ backend_name ^ " Slicing", slicing ctx);
      ("Basic :: " ^ backend_name ^ " Memory and Views", memory_and_views ctx);
      ( "Basic :: " ^ backend_name ^ " Utility Operations",
        utility_operations ctx );
      ("Basic :: " ^ backend_name ^ " Type Conversion", type_conversion ctx);
    ]
end
