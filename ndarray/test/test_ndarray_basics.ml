open Alcotest
module Nd = Ndarray

let ndarray_int32 : (int32, Nd.int32_elt) Nd.t testable =
  Alcotest.testable Ndarray.pp Ndarray.array_equal

let ndarray_float64 : (float, Nd.float64_elt) Nd.t testable =
  Alcotest.testable Ndarray.pp Ndarray.array_equal

let ndarray_complex32 : (Complex.t, Nd.complex32_elt) Nd.t testable =
  Alcotest.testable Ndarray.pp Ndarray.array_equal

(* Testable for complex numbers *)
let complex =
  Alcotest.testable
    (fun ppf c -> Format.fprintf ppf "(%f, %fi)" c.Complex.re c.Complex.im)
    (fun a b ->
      abs_float (a.Complex.re -. b.Complex.re) < 1e-10
      && abs_float (a.Complex.im -. b.Complex.im) < 1e-10)

let float = float 1e-10

(* Creation Tests *)
let test_create_2x2_float32 () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  check (array int) "Shape" [| 2; 2 |] (Nd.shape t);
  check float "Element [0,0]" 1.0 (Nd.get_item [| 0; 0 |] t);
  check float "Element [0,1]" 2.0 (Nd.get_item [| 0; 1 |] t);
  check float "Element [1,0]" 3.0 (Nd.get_item [| 1; 0 |] t);
  check float "Element [1,1]" 4.0 (Nd.get_item [| 1; 1 |] t)

let test_create_1d_int32 () =
  let t = Nd.create Nd.int32 [| 3 |] [| 1l; 2l; 3l |] in
  check (array int) "Shape" [| 3 |] (Nd.shape t);
  check int32 "Element [0]" 1l (Nd.get_item [| 0 |] t);
  check int32 "Element [1]" 2l (Nd.get_item [| 1 |] t);
  check int32 "Element [2]" 3l (Nd.get_item [| 2 |] t)

let test_create_empty_float32 () =
  let t = Nd.create Nd.float32 [| 0 |] [||] in
  check (array int) "Shape" [| 0 |] (Nd.shape t)

let test_create_2x2x2_float32 () =
  let t = Nd.create Nd.float32 [| 2; 2; 2 |] (Array.init 8 float_of_int) in
  check (array int) "Shape" [| 2; 2; 2 |] (Nd.shape t);
  check float "Element [0,0,0]" 0.0 (Nd.get_item [| 0; 0; 0 |] t);
  check float "Element [1,1,1]" 7.0 (Nd.get_item [| 1; 1; 1 |] t)

let test_scalar_float32 () =
  let t = Nd.scalar Nd.float32 42.0 in
  check (array int) "Shape" [||] (Nd.shape t);
  check float "Value" 42.0 (Nd.get_item [||] t)

let test_scalar_int64 () =
  let t = Nd.scalar Nd.int64 100L in
  check (array int) "Shape" [||] (Nd.shape t);
  check int64 "Value" 100L (Nd.get_item [||] t)

(* Custom Operation Tests *)
let test_to_bigarray () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let ba = Nd.to_bigarray t in
  check float "Initial [0,0]" 1.0 (Bigarray.Genarray.get ba [| 0; 0 |]);
  Nd.set_item [| 0; 0 |] 55.0 t;
  check float "After set_item [0,0]" 55.0 (Bigarray.Genarray.get ba [| 0; 0 |])

let test_copy () =
  let original = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let copy_arr = Nd.copy original in
  Nd.set_item [| 0 |] 10.0 original;
  check float "Original [0]" 10.0 (Nd.get_item [| 0 |] original);
  check float "Copy [0]" 1.0 (Nd.get_item [| 0 |] copy_arr)

(* Operation Tests *)
let test_empty_float32 () =
  let t = Nd.empty Nd.float32 [| 2; 2 |] in
  check (array int) "Shape" [| 2; 2 |] (Nd.shape t)

let test_fill_float32 () =
  let t = Nd.empty Nd.float32 [| 2; 2 |] in
  Nd.fill 7.0 t;
  check float "Element [0,0]" 7.0 (Nd.get_item [| 0; 0 |] t);
  check float "Element [1,1]" 7.0 (Nd.get_item [| 1; 1 |] t)

let test_blit () =
  let src = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  let dst = Nd.zeros Nd.float32 [| 2 |] in
  Nd.blit src dst;
  check float "Destination [0]" 1.0 (Nd.get_item [| 0 |] dst);
  check float "Destination [1]" 2.0 (Nd.get_item [| 1 |] dst)

let test_blit_incompatible () =
  let src = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  let dst = Nd.zeros Nd.float32 [| 3 |] in
  check_raises "Incompatible shapes"
    (Invalid_argument "blit: tensors must have the same shape") (fun () ->
      Nd.blit src dst)

let test_full_float32 () =
  let t = Nd.full Nd.float32 [| 2; 3 |] 5.5 in
  check (array int) "Shape" [| 2; 3 |] (Nd.shape t);
  check float "Element [0,0]" 5.5 (Nd.get_item [| 0; 0 |] t);
  check float "Element [1,2]" 5.5 (Nd.get_item [| 1; 2 |] t)

let test_full_like_int32 () =
  let t_ref = Nd.create Nd.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  let t = Nd.full_like 10l t_ref in
  check (array int) "Shape" [| 2; 2 |] (Nd.shape t);
  check int32 "Element [0,0]" 10l (Nd.get_item [| 0; 0 |] t)

let test_empty_like_float64 () =
  let t_ref = Nd.create Nd.float64 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t = Nd.empty_like t_ref in
  check (array int) "Shape" [| 2; 2 |] (Nd.shape t)

let test_zeros_float32 () =
  let t = Nd.zeros Nd.float32 [| 2; 2 |] in
  check (array int) "Shape" [| 2; 2 |] (Nd.shape t);
  check float "Element [0,0]" 0.0 (Nd.get_item [| 0; 0 |] t)

let test_zeros_like_float32 () =
  let t_ref = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t = Nd.zeros_like t_ref in
  check (array int) "Shape" [| 2; 2 |] (Nd.shape t);
  check float "Element [0,0]" 0.0 (Nd.get_item [| 0; 0 |] t)

let test_ones_float32 () =
  let t = Nd.ones Nd.float32 [| 2; 2 |] in
  check (array int) "Shape" [| 2; 2 |] (Nd.shape t);
  check float "Element [0,0]" 1.0 (Nd.get_item [| 0; 0 |] t)

let test_ones_like_int32 () =
  let t_ref = Nd.create Nd.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  let t = Nd.ones_like t_ref in
  check (array int) "Shape" [| 2; 2 |] (Nd.shape t);
  check int32 "Element [0,0]" 1l (Nd.get_item [| 0; 0 |] t)

let test_identity_float32 () =
  let t = Nd.identity Nd.float32 3 in
  check (array int) "Shape" [| 3; 3 |] (Nd.shape t);
  check float "Element [0,0]" 1.0 (Nd.get_item [| 0; 0 |] t);
  check float "Element [1,1]" 1.0 (Nd.get_item [| 1; 1 |] t);
  check float "Element [0,1]" 0.0 (Nd.get_item [| 0; 1 |] t)

(* Additional Creation Tests *)
let test_identity_1x1_int32 () =
  let t = Nd.identity Nd.int32 1 in
  check (array int) "Shape" [| 1; 1 |] (Nd.shape t);
  check int32 "Element [0,0]" 1l (Nd.get_item [| 0; 0 |] t)

let test_eye_2x2_float32 () =
  let t = Nd.eye Nd.float32 2 in
  check (array int) "Shape" [| 2; 2 |] (Nd.shape t);
  check float "Element [0,0]" 1.0 (Nd.get_item [| 0; 0 |] t);
  check float "Element [0,1]" 0.0 (Nd.get_item [| 0; 1 |] t);
  check float "Element [1,0]" 0.0 (Nd.get_item [| 1; 0 |] t);
  check float "Element [1,1]" 1.0 (Nd.get_item [| 1; 1 |] t)

let test_eye_3x4_float32 () =
  let t = Nd.eye ~m:3 Nd.float32 4 in
  check (array int) "Shape" [| 3; 4 |] (Nd.shape t);
  check float "Element [0,0]" 1.0 (Nd.get_item [| 0; 0 |] t);
  check float "Element [1,1]" 1.0 (Nd.get_item [| 1; 1 |] t);
  check float "Element [2,2]" 1.0 (Nd.get_item [| 2; 2 |] t);
  check float "Element [0,1]" 0.0 (Nd.get_item [| 0; 1 |] t);
  check float "Element [2,3]" 0.0 (Nd.get_item [| 2; 3 |] t)

let test_eye_4x3_k1_float32 () =
  let t = Nd.eye ~m:4 ~k:1 Nd.float32 3 in
  check (array int) "Shape" [| 4; 3 |] (Nd.shape t);
  check float "Element [0,1]" 1.0 (Nd.get_item [| 0; 1 |] t);
  check float "Element [1,2]" 1.0 (Nd.get_item [| 1; 2 |] t);
  check float "Element [0,0]" 0.0 (Nd.get_item [| 0; 0 |] t);
  check float "Element [2,2]" 0.0 (Nd.get_item [| 2; 2 |] t)

let test_eye_3x3_km1_int32 () =
  let t = Nd.eye ~k:(-1) Nd.int32 3 in
  check (array int) "Shape" [| 3; 3 |] (Nd.shape t);
  check int32 "Element [1,0]" 1l (Nd.get_item [| 1; 0 |] t);
  check int32 "Element [2,1]" 1l (Nd.get_item [| 2; 1 |] t);
  check int32 "Element [0,0]" 0l (Nd.get_item [| 0; 0 |] t);
  check int32 "Element [2,2]" 0l (Nd.get_item [| 2; 2 |] t)

let test_arange_int32 () =
  let t = Nd.arange Nd.int32 0 10 2 in
  check (array int) "Shape" [| 5 |] (Nd.shape t);
  check int32 "Element [0]" 0l (Nd.get_item [| 0 |] t);
  check int32 "Element [1]" 2l (Nd.get_item [| 1 |] t);
  check int32 "Element [2]" 4l (Nd.get_item [| 2 |] t);
  check int32 "Element [3]" 6l (Nd.get_item [| 3 |] t);
  check int32 "Element [4]" 8l (Nd.get_item [| 4 |] t)

let test_arange_f_float32 () =
  let t = Nd.arange_f Nd.float32 0.0 5.0 0.5 in
  check (array int) "Shape" [| 10 |] (Nd.shape t);
  let expected = [| 0.0; 0.5; 1.0; 1.5; 2.0; 2.5; 3.0; 3.5; 4.0; 4.5 |] in
  Array.iteri
    (fun i v ->
      check float
        ("Element [" ^ string_of_int i ^ "]")
        v (Nd.get_item [| i |] t))
    expected

let test_linspace_float32 () =
  let t = Nd.linspace Nd.float32 2.0 3.0 5 in
  check (array int) "Shape" [| 5 |] (Nd.shape t);
  let expected = [| 2.0; 2.25; 2.5; 2.75; 3.0 |] in
  Array.iteri
    (fun i v ->
      check float
        ("Element [" ^ string_of_int i ^ "]")
        v (Nd.get_item [| i |] t))
    expected

let test_linspace_no_endpoint_float64 () =
  let t = Nd.linspace ~endpoint:false Nd.float64 0.0 4.0 5 in
  check (array int) "Shape" [| 5 |] (Nd.shape t);
  let expected = [| 0.0; 0.8; 1.6; 2.4; 3.2 |] in
  Array.iteri
    (fun i v ->
      check float
        ("Element [" ^ string_of_int i ^ "]")
        v (Nd.get_item [| i |] t))
    expected

let test_logspace_base10_float64 () =
  let t = Nd.logspace ~base:10.0 Nd.float64 2.0 3.0 4 in
  check (array int) "Shape" [| 4 |] (Nd.shape t);
  let expected =
    [| 100.0; 215.443469003188454; 464.158883361277731; 1000.0 |]
  in
  Array.iteri
    (fun i v ->
      check float
        ("Element [" ^ string_of_int i ^ "]")
        v (Nd.get_item [| i |] t))
    expected

let test_logspace_base2_no_endpoint_float64 () =
  let t = Nd.logspace ~endpoint:false ~base:2.0 Nd.float64 0.0 4.0 5 in
  check (array int) "Shape" [| 5 |] (Nd.shape t);
  let expected =
    [|
      1.0;
      1.741101126592248;
      3.031433133020796;
      5.278031643091579;
      9.189586839976281;
    |]
  in
  Array.iteri
    (fun i v ->
      check float
        ("Element [" ^ string_of_int i ^ "]")
        v (Nd.get_item [| i |] t))
    expected

let test_logspace_default_float32 () =
  let t = Nd.logspace Nd.float32 1.0 3.0 3 in
  check (array int) "Shape" [| 3 |] (Nd.shape t);
  let expected = [| 10.0; 100.0; 1000.0 |] in
  Array.iteri
    (fun i v ->
      check float
        ("Element [" ^ string_of_int i ^ "]")
        v (Nd.get_item [| i |] t))
    expected

let test_geomspace_float32 () =
  let t = Nd.geomspace Nd.float32 2.0 32.0 5 in
  check (array int) "Shape" [| 5 |] (Nd.shape t);
  let expected = [| 2.0; 4.0; 8.0; 16.0; 32.0 |] in
  Array.iteri
    (fun i v ->
      check float
        ("Element [" ^ string_of_int i ^ "]")
        v (Nd.get_item [| i |] t))
    expected

let test_geomspace_no_endpoint_float64 () =
  let t = Nd.geomspace ~endpoint:false Nd.float64 1.0 256.0 9 in
  check (array int) "Shape" [| 9 |] (Nd.shape t);
  let expected =
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
  in
  Array.iteri
    (fun i v ->
      check float
        ("Element [" ^ string_of_int i ^ "]")
        v (Nd.get_item [| i |] t))
    expected

let test_create_complex64 () =
  let t =
    Nd.create Nd.complex64 [| 2 |]
      [| Complex.{ re = 1.0; im = 2.0 }; { re = 3.0; im = 4.0 } |]
  in
  check (array int) "Shape" [| 2 |] (Nd.shape t);
  check complex "Element [0]"
    { Complex.re = 1.0; im = 2.0 }
    (Nd.get_item [| 0 |] t);
  check complex "Element [1]"
    { Complex.re = 3.0; im = 4.0 }
    (Nd.get_item [| 1 |] t)

(* Property Tests *)
let test_shape_2x3 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] (Array.init 6 float_of_int) in
  check (array int) "Shape" [| 2; 3 |] (Nd.shape t)

let test_strides_2x3_float32 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] (Array.init 6 float_of_int) in
  check (array int) "Strides" [| 3; 1 |] (Nd.strides t)

let test_stride_dim0_2x3_float32 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] (Array.init 6 float_of_int) in
  check int "Stride dim 0" 3 (Nd.stride 0 t)

let test_stride_dim1_2x3_float32 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] (Array.init 6 float_of_int) in
  check int "Stride dim 1" 1 (Nd.stride 1 t)

let test_strides_2x3_int64 () =
  let t = Nd.create Nd.int64 [| 2; 3 |] (Array.init 6 Int64.of_int) in
  check (array int) "Strides" [| 3; 1 |] (Nd.strides t)

let test_itemsize_float32 () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  check int "Itemsize" 4 (Nd.itemsize t)

let test_itemsize_int64 () =
  let t = Nd.create Nd.int64 [| 2; 2 |] [| 1L; 2L; 3L; 4L |] in
  check int "Itemsize" 8 (Nd.itemsize t)

let test_ndim_scalar () =
  let t = Nd.scalar Nd.float32 1.0 in
  check int "Ndim" 0 (Nd.ndim t)

let test_ndim_2x2 () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  check int "Ndim" 2 (Nd.ndim t)

let test_dim_0_2x3 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  check int "Dim 0" 2 (Nd.dim 0 t)

let test_dims_2x3 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] (Array.init 6 float_of_int) in
  check (array int) "Dims" [| 2; 3 |] (Nd.dims t)

let test_nbytes_float32 () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  check int "Nbytes" 16 (Nd.nbytes t)

let test_nbytes_int64 () =
  let t = Nd.create Nd.int64 [| 2; 3 |] (Array.init 6 Int64.of_int) in
  check int "Nbytes" 48 (Nd.nbytes t)

let test_size_2x3 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  check int "Size" 6 (Nd.size t)

let test_size_scalar () =
  let t = Nd.scalar Nd.float32 10.0 in
  check int "Size" 1 (Nd.size t)

let test_offset_basic () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  check int "Offset" 0 (Nd.offset t)

let test_offset_slice () =
  let t = Nd.create Nd.float32 [| 3; 3 |] (Array.init 9 float_of_int) in
  let s = Nd.slice [| 1; -1 |] [| 1; -1 |] t in
  check int "Offset" 5 (Nd.offset s)

let test_data_buffer_view () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let d = Nd.data t in
  Bigarray.Array1.set d 0 99.0;
  check float "Element [0] after modification" 99.0 (Nd.get_item [| 0 |] t)

(* Element Access Tests *)
let test_get_item_2x2 () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let e = Nd.get_item [| 0; 1 |] t in
  Alcotest.check float "Element at [0,1]" 2.0 e

let test_get_item_multi_dim () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let e = Nd.get_item [| 0; 1 |] t in
  Alcotest.check float "Element at [0,1]" 2.0 e

let test_set_item_2x2 () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Nd.set_item [| 1; 0 |] 5.0 t;
  let e = Nd.get_item [| 1; 0 |] t in
  Alcotest.check float "Element at [1,0] after set" 5.0 e

let test_get_item_out_of_bounds () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Alcotest.check_raises "Out of bounds"
    (Invalid_argument
       "get_item: Index 2 at dimension 0 is out of bounds for shape [2; 2]")
    (fun () -> ignore (Nd.get_item [| 2; 0 |] t))

let test_set_item_out_of_bounds () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Alcotest.check_raises "Out of bounds"
    (Invalid_argument
       "set_item: Index 2 at dimension 1 is out of bounds for shape [2; 2]")
    (fun () -> Nd.set_item [| 0; 2 |] 5.0 t)

let test_get_view_row () =
  let t = Nd.create Nd.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  let row = Nd.get [| 0 |] t in
  let expected = Nd.create Nd.int32 [| 2 |] [| 1l; 2l |] in
  Alcotest.check ndarray_int32 "First row" expected row

let test_get_scalar () =
  let t = Nd.create Nd.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  let scalar = Nd.get [| 1; 1 |] t in
  let expected = Nd.scalar Nd.int32 4l in
  Alcotest.check ndarray_int32 "Element at [1,1]" expected scalar

let test_set_view_row () =
  let t = Nd.create Nd.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  let v = Nd.create Nd.int32 [| 2 |] [| 8l; 9l |] in
  Nd.set [| 0 |] v t;
  let expected = Nd.create Nd.int32 [| 2; 2 |] [| 8l; 9l; 3l; 4l |] in
  Alcotest.check ndarray_int32 "After setting first row" expected t

let test_set_scalar () =
  let t = Nd.create Nd.int32 [| 2; 2 |] [| 1l; 2l; 3l; 4l |] in
  let v = Nd.scalar Nd.int32 99l in
  Nd.set [| 1; 0 |] v t;
  let expected = Nd.create Nd.int32 [| 2; 2 |] [| 1l; 2l; 99l; 4l |] in
  Alcotest.check ndarray_int32 "After setting [1,0]" expected t

let test_slice_3x4 () =
  let t = Nd.create Nd.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let s = Nd.slice [| 1; 0 |] [| 3; 4 |] t in
  Alcotest.(check (array int)) "Shape of slice" [| 2; 4 |] (Nd.shape s);
  Alcotest.check float "Element [0,0] of slice" 4.0 (Nd.get_item [| 0; 0 |] s)

let test_slice_with_steps () =
  let t = Nd.create Nd.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let s = Nd.slice ~steps:[| 2; 2 |] [| 0; 0 |] [| 3; 4 |] t in
  Alcotest.(check (array int)) "Shape of slice" [| 2; 2 |] (Nd.shape s);
  Alcotest.check float "Element [0,0]" 0.0 (Nd.get_item [| 0; 0 |] s);
  Alcotest.check float "Element [0,1]" 2.0 (Nd.get_item [| 0; 1 |] s)

let test_slice_view () =
  let t = Nd.create Nd.float32 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let s = Nd.slice [| 1; 0 |] [| 2; 2 |] t in
  Nd.set_item [| 1; 0 |] 99.0 t;
  Alcotest.check float "Element [0,0] of slice after modification" 99.0
    (Nd.get_item [| 0; 0 |] s)

(* Conversion Tests *)
let test_to_array () =
  let t = Nd.create Nd.int32 [| 3 |] [| 1l; 2l; 3l |] in
  let a = Nd.to_array t in
  Alcotest.(check (array int32)) "to_array" [| 1l; 2l; 3l |] a

let test_astype_float32_to_int32 () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.1; 2.9; -3.3 |] in
  let u = Nd.astype Nd.int32 t in
  let expected = Nd.create Nd.int32 [| 3 |] [| 1l; 2l; -3l |] in
  Alcotest.check ndarray_int32 "astype to int32" expected u

let test_astype_int32_to_float64 () =
  let t = Nd.create Nd.int32 [| 3 |] [| 1l; 2l; 3l |] in
  let u = Nd.astype Nd.float64 t in
  let expected = Nd.create Nd.float64 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  Alcotest.check ndarray_float64 "astype to float64" expected u

let test_astype_float32_to_complex32 () =
  let t = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  let u = Nd.astype Nd.complex32 t in
  let expected =
    Nd.create Nd.complex32 [| 2 |]
      [| { Complex.re = 1.0; im = 0.0 }; { re = 2.0; im = 0.0 } |]
  in
  Alcotest.check ndarray_complex32 "astype to complex32" expected u

let test_astype_complex64_to_float64 () =
  let t =
    Nd.create Nd.complex64 [| 2 |]
      [| { Complex.re = 1.0; im = 2.0 }; { re = 3.0; im = 4.0 } |]
  in
  let u = Nd.astype Nd.float64 t in
  let expected = Nd.create Nd.float64 [| 2 |] [| 1.0; 3.0 |] in
  Alcotest.check ndarray_float64 "astype complex64 to float64" expected u

(* Test Suite Organization *)

let creation_tests =
  [
    ("create 2x2 float32", `Quick, test_create_2x2_float32);
    ("create 1D int32", `Quick, test_create_1d_int32);
    ("create empty float32", `Quick, test_create_empty_float32);
    ("create 2x2x2 float32", `Quick, test_create_2x2x2_float32);
    ("create scalar float32", `Quick, test_scalar_float32);
    ("create scalar int64", `Quick, test_scalar_int64);
    ("create identity 1x1 int32", `Quick, test_identity_1x1_int32);
    ("identity float32", `Quick, test_identity_float32);
    ("create eye 2x2 float32", `Quick, test_eye_2x2_float32);
    ("create eye 3x4 float32", `Quick, test_eye_3x4_float32);
    ("create eye 4x3 k=1 float32", `Quick, test_eye_4x3_k1_float32);
    ("create eye 3x3 k=-1 int32", `Quick, test_eye_3x3_km1_int32);
    ("create arange int32", `Quick, test_arange_int32);
    ("create arange_f float32", `Quick, test_arange_f_float32);
    ("create linspace float32", `Quick, test_linspace_float32);
    ( "create linspace no endpoint float64",
      `Quick,
      test_linspace_no_endpoint_float64 );
    ("create logspace base 10 float64", `Quick, test_logspace_base10_float64);
    ( "create logspace base 2 no endpoint float64",
      `Quick,
      test_logspace_base2_no_endpoint_float64 );
    ("create logspace default float32", `Quick, test_logspace_default_float32);
    ("create geomspace float32", `Quick, test_geomspace_float32);
    ( "create geomspace no endpoint float64",
      `Quick,
      test_geomspace_no_endpoint_float64 );
    ("create complex64", `Quick, test_create_complex64);
    ("empty float32", `Quick, test_empty_float32);
    ("full float32", `Quick, test_full_float32);
    ("full_like int32", `Quick, test_full_like_int32);
    ("empty_like float64", `Quick, test_empty_like_float64);
    ("zeros float32", `Quick, test_zeros_float32);
    ("zeros_like float32", `Quick, test_zeros_like_float32);
    ("ones float32", `Quick, test_ones_float32);
    ("ones_like int32", `Quick, test_ones_like_int32);
  ]

let property_tests =
  [
    ("get shape 2x3", `Quick, test_shape_2x3);
    ("get strides 2x3 float32", `Quick, test_strides_2x3_float32);
    ("get stride dim 0 2x3 float32", `Quick, test_stride_dim0_2x3_float32);
    ("get stride dim 1 2x3 float32", `Quick, test_stride_dim1_2x3_float32);
    ("get strides 2x3 int64", `Quick, test_strides_2x3_int64);
    ("check itemsize float32", `Quick, test_itemsize_float32);
    ("check itemsize int64", `Quick, test_itemsize_int64);
    ("get ndim scalar", `Quick, test_ndim_scalar);
    ("get ndim 2x2", `Quick, test_ndim_2x2);
    ("get dim 0 2x3", `Quick, test_dim_0_2x3);
    ("get dims 2x3", `Quick, test_dims_2x3);
    ("get nbytes float32", `Quick, test_nbytes_float32);
    ("get nbytes int64", `Quick, test_nbytes_int64);
    ("get size 2x3", `Quick, test_size_2x3);
    ("get size scalar", `Quick, test_size_scalar);
    ("get offset basic", `Quick, test_offset_basic);
    ("get offset slice", `Quick, test_offset_slice);
    ("data buffer view", `Quick, test_data_buffer_view);
  ]

let element_access_tests =
  [
    ("get item 2x2", `Quick, test_get_item_2x2);
    ("get item multi-dim", `Quick, test_get_item_multi_dim);
    ("set item 2x2", `Quick, test_set_item_2x2);
    ("get item out of bounds", `Quick, test_get_item_out_of_bounds);
    ("set item out of bounds", `Quick, test_set_item_out_of_bounds);
    ("get view row", `Quick, test_get_view_row);
    ("get scalar", `Quick, test_get_scalar);
    ("set view row", `Quick, test_set_view_row);
    ("set scalar", `Quick, test_set_scalar);
    ("slice 3x4", `Quick, test_slice_3x4);
    ("slice with steps", `Quick, test_slice_with_steps);
    ("slice view", `Quick, test_slice_view);
  ]

let utility_tests =
  [
    ("copy ndarray", `Quick, test_copy);
    ("fill float32", `Quick, test_fill_float32);
    ("blit 1D to 1D", `Quick, test_blit);
    ("blit incompatible shapes", `Quick, test_blit_incompatible);
  ]

let conversion_tests =
  [
    ("convert to Bigarray", `Quick, test_to_bigarray);
    ("to_array", `Quick, test_to_array);
    ("astype float32 to int32", `Quick, test_astype_float32_to_int32);
    ("astype int32 to float64", `Quick, test_astype_int32_to_float64);
    ("astype float32 to complex32", `Quick, test_astype_float32_to_complex32);
    ("astype complex64 to float64", `Quick, test_astype_complex64_to_float64);
  ]

let () =
  Printexc.record_backtrace true;
  Alcotest.run "Ndarray Basics"
    [
      ("Creation", creation_tests);
      ("Properties", property_tests);
      ("Element Access", element_access_tests);
      ("Utilities", utility_tests);
      ("Conversion", conversion_tests);
    ]
