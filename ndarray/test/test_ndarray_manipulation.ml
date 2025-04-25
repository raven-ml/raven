open Alcotest
module Nd = Ndarray

let ndarray_float32 : (float, Nd.float32_elt) Nd.t testable =
  Alcotest.testable Ndarray.pp Ndarray.array_equal

let float = float 1e-10

(* Array Manipulation Tests *)
let test_flatten_2x2 () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let flat = Nd.flatten t in
  let expected = Nd.create Nd.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Alcotest.check ndarray_float32 "Flattened 2D array" expected flat

let test_flatten_1d () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let flat = Nd.flatten t in
  Alcotest.check ndarray_float32 "Flattened 1D array" t flat

let test_ravel_contiguous_view () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let r = Nd.ravel t in
  Nd.set_item [| 0; 0 |] 99.0 t;
  Alcotest.check float "Raveled view after modification" 99.0
    (Nd.get_item [| 0 |] r)

let test_ravel_non_contiguous_copy () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let tr = Nd.transpose t in
  let r = Nd.ravel tr in
  Nd.set_item [| 0; 0 |] 99.0 t;
  Alcotest.check float "Raveled copy after modification" 1.0
    (Nd.get_item [| 0 |] r)

let test_ravel_3d () =
  let t =
    Nd.create Nd.float32 [| 2; 2; 2 |]
      [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0 |]
  in
  let r = Nd.ravel t in
  let expected =
    Nd.create Nd.float32 [| 8 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0 |]
  in
  Alcotest.check ndarray_float32 "Raveled 3D array" expected r

let test_reshape_1d_to_2x2 () =
  let t = Nd.create Nd.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let r = Nd.reshape [| 2; 2 |] t in
  let expected = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Alcotest.check ndarray_float32 "Reshaped to 2x2" expected r

let test_reshape_view () =
  let t = Nd.create Nd.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let r = Nd.reshape [| 2; 2 |] t in
  Nd.set_item [| 0 |] 99.0 t;
  let e = Nd.get_item [| 0; 0 |] r in
  Alcotest.check float "Reshaped view after modification" 99.0 e

let test_reshape_to_vector () =
  let t = Nd.create Nd.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let r = Nd.reshape [| 4; 1 |] t in
  let expected = Nd.create Nd.float32 [| 4; 1 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Alcotest.check ndarray_float32 "Reshaped to [4,1]" expected r

let test_reshape_incompatible () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  Alcotest.check_raises "Incompatible shapes"
    (Invalid_argument
       "reshape: cannot reshape array of size 3 into shape [2; 2]") (fun () ->
      ignore (Nd.reshape [| 2; 2 |] t))

let test_transpose_2d_default () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let tr = Nd.transpose t in
  let expected =
    Nd.create Nd.float32 [| 3; 2 |] [| 1.0; 4.0; 2.0; 5.0; 3.0; 6.0 |]
  in
  Alcotest.check ndarray_float32 "Transposed 2D array default" expected tr

let test_transpose_2d_with_axes () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let tr = Nd.transpose ~axes:[| 1; 0 |] t in
  let expected =
    Nd.create Nd.float32 [| 3; 2 |] [| 1.0; 4.0; 2.0; 5.0; 3.0; 6.0 |]
  in
  Alcotest.check ndarray_float32 "Transposed 2D array with axes" expected tr

let test_transpose_3d_with_axes () =
  let t = Nd.create Nd.float32 [| 2; 3; 4 |] (Array.init 24 float_of_int) in
  let result = Nd.transpose ~axes:[| 1; 2; 0 |] t in
  check (array int) "Shape" [| 3; 4; 2 |] (Nd.shape result);
  check float "Element [0,0,0]" 0.0 (Nd.get_item [| 0; 0; 0 |] result);
  check float "Element [0,0,1]" 12.0 (Nd.get_item [| 0; 0; 1 |] result);
  check float "Element [0,1,0]" 1.0 (Nd.get_item [| 0; 1; 0 |] result);
  check float "Element [1,0,0]" 4.0 (Nd.get_item [| 1; 0; 0 |] result)

let test_transpose_scalar () =
  let t = Nd.scalar Nd.float32 42.0 in
  let result = Nd.transpose t in
  check ndarray_float32 "Transpose scalar" t result

let test_transpose_view () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let tr = Nd.transpose t in
  Nd.set_item [| 0; 1 |] 99.0 t;
  Alcotest.check float "Transposed view after modification" 99.0
    (Nd.get_item [| 1; 0 |] tr)

let test_transpose_invalid_axes () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Alcotest.check_raises "Invalid axes length"
    (Invalid_argument "transpose: axes length 3 does not match tensor rank 2")
    (fun () -> ignore (Nd.transpose ~axes:[| 0; 1; 2 |] t))

let test_squeeze_basic () =
  let t =
    Nd.create Nd.float32 [| 1; 2; 1; 3 |] [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0 |]
  in
  let s = Nd.squeeze t in
  Alcotest.(check (array int)) "Shape after squeeze" [| 2; 3 |] (Nd.shape s)

let test_squeeze_specific_axes () =
  let t =
    Nd.create Nd.float32 [| 1; 2; 1; 3 |] [| 0.0; 1.0; 2.0; 3.0; 4.0; 5.0 |]
  in
  let s = Nd.squeeze ~axes:[| 0; 2 |] t in
  Alcotest.(check (array int))
    "Shape after squeeze axes" [| 2; 3 |] (Nd.shape s)

let test_squeeze_no_op () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let s = Nd.squeeze t in
  Alcotest.(check (array int))
    "Shape after squeeze no-op" [| 2; 2 |] (Nd.shape s)

let test_split_1d () =
  let t = Nd.create Nd.float32 [| 6 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let parts = Nd.split 3 t in
  let p1 = List.nth parts 0 in
  let expected = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  Alcotest.check ndarray_float32 "Split 1D part 1" expected p1

let test_split_2d_axis0 () =
  let t =
    Nd.create Nd.float32 [| 4; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0 |]
  in
  let parts = Nd.split ~axis:0 2 t in
  let p1 = List.nth parts 0 in
  let expected = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Alcotest.check ndarray_float32 "Split 2D axis 0 part 1" expected p1

let test_split_views () =
  let t = Nd.create Nd.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let parts = Nd.split 2 t in
  let p1 = List.nth parts 0 in
  Nd.set_item [| 0 |] 99.0 p1;
  Alcotest.check float "Split view after modification" 99.0
    (Nd.get_item [| 0 |] t)

let test_split_invalid () =
  let t = Nd.create Nd.float32 [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  Alcotest.check_raises "Size not divisible"
    (Invalid_argument "split: size 5 along axis 0 not divisible by 2")
    (fun () -> ignore (Nd.split 2 t))

let test_array_split_equal () =
  let t = Nd.create Nd.float32 [| 6 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let parts = Nd.array_split 3 t in
  let p1 = List.nth parts 0 in
  let expected = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  Alcotest.check ndarray_float32 "Array split equal part 1" expected p1

let test_array_split_unequal () =
  let t = Nd.create Nd.float32 [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let parts = Nd.array_split 3 t in
  let p3 = List.nth parts 2 in
  let expected = Nd.create Nd.float32 [| 1 |] [| 5.0 |] in
  Alcotest.check ndarray_float32 "Array split unequal part 3" expected p3

let test_array_split_views () =
  let t = Nd.create Nd.float32 [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let parts = Nd.array_split 2 t in
  let p1 = List.nth parts 0 in
  Nd.set_item [| 0 |] 99.0 p1;
  Alcotest.check float "Array split view after modification" 99.0
    (Nd.get_item [| 0 |] t)

let test_concatenate_1d () =
  let t1 = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  let t2 = Nd.create Nd.float32 [| 2 |] [| 3.0; 4.0 |] in
  let c = Nd.concatenate [ t1; t2 ] in
  let expected = Nd.create Nd.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Alcotest.check ndarray_float32 "Concatenated 1D arrays" expected c

let test_concatenate_2d_axis0 () =
  let t1 = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t2 = Nd.create Nd.float32 [| 1; 2 |] [| 5.0; 6.0 |] in
  let c = Nd.concatenate ~axis:0 [ t1; t2 ] in
  let expected =
    Nd.create Nd.float32 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
  in
  Alcotest.check ndarray_float32 "Concatenated 2D axis 0" expected c

let test_concatenate_new_array () =
  let t1 = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  let t2 = Nd.create Nd.float32 [| 2 |] [| 3.0; 4.0 |] in
  let c = Nd.concatenate [ t1; t2 ] in
  Nd.set_item [| 0 |] 99.0 t1;
  Alcotest.check float "Concatenated is new array" 1.0 (Nd.get_item [| 0 |] c)

let test_concatenate_invalid () =
  let t1 = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t2 = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  Alcotest.check_raises "Incompatible shapes"
    (Invalid_argument "concatenate: shape mismatch at dimension 1 (3 vs 2)")
    (fun () -> ignore (Nd.concatenate ~axis:0 [ t1; t2 ]))

let test_stack_axis0 () =
  let t1 = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t2 = Nd.create Nd.float32 [| 2; 2 |] [| 5.0; 6.0; 7.0; 8.0 |] in
  let s = Nd.stack ~axis:0 [ t1; t2 ] in
  let expected =
    Nd.create Nd.float32 [| 2; 2; 2 |]
      [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0 |]
  in
  Alcotest.check ndarray_float32 "Stacked along axis 0" expected s

let test_stack_new_array () =
  let t1 = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  let t2 = Nd.create Nd.float32 [| 2 |] [| 3.0; 4.0 |] in
  let s = Nd.stack ~axis:0 [ t1; t2 ] in
  Nd.set_item [| 0 |] 99.0 t1;
  Alcotest.check float "Stacked is new array" 1.0 (Nd.get_item [| 0; 0 |] s)

let test_stack_invalid () =
  let t1 = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  let t2 = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  Alcotest.check_raises "Shape mismatch"
    (Invalid_argument "concatenate: shape mismatch at dimension 1 (3 vs 2)")
    (fun () -> ignore (Nd.stack ~axis:0 [ t1; t2 ]))

let test_vstack_1d () =
  let t1 = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  let t2 = Nd.create Nd.float32 [| 2 |] [| 3.0; 4.0 |] in
  let v = Nd.vstack [ t1; t2 ] in
  let expected = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Alcotest.check ndarray_float32 "Vstack 1D arrays" expected v

let test_vstack_2d () =
  let t1 = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t2 = Nd.create Nd.float32 [| 2; 2 |] [| 5.0; 6.0; 7.0; 8.0 |] in
  let v = Nd.vstack [ t1; t2 ] in
  let expected =
    Nd.create Nd.float32 [| 4; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0 |]
  in
  Alcotest.check ndarray_float32 "Vstack 2D arrays" expected v

let test_vstack_invalid () =
  let t1 = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t2 = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  Alcotest.check_raises "Shape mismatch"
    (Invalid_argument "concatenate: shape mismatch at dimension 1 (3 vs 2)")
    (fun () -> ignore (Nd.vstack [ t1; t2 ]))

let test_hstack_1d () =
  let t1 = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  let t2 = Nd.create Nd.float32 [| 2 |] [| 3.0; 4.0 |] in
  let h = Nd.hstack [ t1; t2 ] in
  let expected = Nd.create Nd.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Alcotest.check ndarray_float32 "Hstack 1D arrays" expected h

let test_hstack_2d () =
  let t1 = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t2 = Nd.create Nd.float32 [| 2; 1 |] [| 5.0; 6.0 |] in
  let h = Nd.hstack [ t1; t2 ] in
  let expected =
    Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 5.0; 3.0; 4.0; 6.0 |]
  in
  Alcotest.check ndarray_float32 "Hstack 2D arrays" expected h

let test_hstack_invalid () =
  let t1 = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t2 = Nd.create Nd.float32 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  Alcotest.check_raises "Shape mismatch"
    (Invalid_argument "concatenate: shape mismatch at dimension 0 (3 vs 2)")
    (fun () -> ignore (Nd.hstack [ t1; t2 ]))

let test_dstack_2d () =
  let t1 = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t2 = Nd.create Nd.float32 [| 2; 2 |] [| 5.0; 6.0; 7.0; 8.0 |] in
  let d = Nd.dstack [ t1; t2 ] in
  let expected =
    Nd.create Nd.float32 [| 2; 2; 2 |]
      [| 1.0; 5.0; 2.0; 6.0; 3.0; 7.0; 4.0; 8.0 |]
  in
  Alcotest.check ndarray_float32 "Dstack 2D arrays" expected d

let test_dstack_1d () =
  let t1 = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  let t2 = Nd.create Nd.float32 [| 2 |] [| 3.0; 4.0 |] in
  let d = Nd.dstack [ t1; t2 ] in
  let expected = Nd.create Nd.float32 [| 1; 2; 2 |] [| 1.0; 3.0; 2.0; 4.0 |] in
  Alcotest.check ndarray_float32 "Dstack 1D arrays" expected d

let test_dstack_invalid () =
  let t1 = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t2 = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  Alcotest.check_raises "Shape mismatch"
    (Invalid_argument "concatenate: shape mismatch at dimension 1 (3 vs 2)")
    (fun () -> ignore (Nd.dstack [ t1; t2 ]))

let test_pad_1d () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let p = Nd.pad [| (1, 2) |] 0.0 t in
  let expected =
    Nd.create Nd.float32 [| 6 |] [| 0.0; 1.0; 2.0; 3.0; 0.0; 0.0 |]
  in
  Alcotest.check ndarray_float32 "Padded 1D array" expected p

let test_pad_2d () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let p = Nd.pad [| (1, 1); (0, 1) |] 0.0 t in
  let expected =
    Nd.create Nd.float32 [| 4; 3 |]
      [| 0.0; 0.0; 0.0; 1.0; 2.0; 0.0; 3.0; 4.0; 0.0; 0.0; 0.0; 0.0 |]
  in
  Alcotest.check ndarray_float32 "Padded 2D array" expected p

let test_pad_invalid () =
  let t = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  Alcotest.check_raises "Negative padding"
    (Invalid_argument
       "pad: padding values must be non-negative, got (-1, 2) for axis 0")
    (fun () -> ignore (Nd.pad [| (-1, 2) |] 0.0 t))

let test_expand_dims_axis_0 () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let e = Nd.expand_dims 0 t in
  Alcotest.(check (array int))
    "Shape after expand axis 0" [| 1; 3 |] (Nd.shape e)

let test_expand_dims_axis_1 () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let e = Nd.expand_dims 1 t in
  Alcotest.(check (array int))
    "Shape after expand axis 1" [| 3; 1 |] (Nd.shape e)

let test_expand_dims_invalid_axis () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  Alcotest.check_raises "Invalid axis"
    (Invalid_argument "expand_dims: axis 2 out of bounds for shape [3]")
    (fun () -> ignore (Nd.expand_dims 2 t))

let test_broadcast_1d_to_3x3 () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let b = Nd.broadcast_to [| 3; 3 |] t in
  Alcotest.(check (array int)) "Shape after broadcast" [| 3; 3 |] (Nd.shape b)

let test_broadcast_scalar_to_2x2 () =
  let t = Nd.scalar Nd.float32 5.0 in
  let b = Nd.broadcast_to [| 2; 2 |] t in
  Alcotest.check float "Broadcast scalar element" 5.0 (Nd.get_item [| 0; 0 |] b)

let test_broadcast_incompatible () =
  let t = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  Alcotest.check_raises "Incompatible broadcast"
    (Invalid_argument
       "broadcast_to: shapes [2] and [3; 3] are not compatible for \
        broadcasting at dimension 1 (original size 2 vs target size 3)")
    (fun () -> ignore (Nd.broadcast_to [| 3; 3 |] t))

let test_broadcast_arrays () =
  let t1 = Nd.create Nd.float32 [| 3; 1 |] [| 1.0; 2.0; 3.0 |] in
  let t2 = Nd.create Nd.float32 [| 1; 4 |] [| 10.0; 20.0; 30.0; 40.0 |] in
  let[@warning "-8"] [ b1; _b2 ] = Nd.broadcast_arrays [ t1; t2 ] in
  Alcotest.(check (array int)) "Shape of b1" [| 3; 4 |] (Nd.shape b1);
  Alcotest.check float "b1[0,0]" 1.0 (Nd.get_item [| 0; 0 |] b1)

let test_broadcast_arrays_views () =
  let t1 = Nd.create Nd.float32 [| 3; 1 |] [| 1.0; 2.0; 3.0 |] in
  let t2 = Nd.create Nd.float32 [| 1; 1 |] [| 10.0 |] in
  let[@warning "-8"] [ b1; _ ] = Nd.broadcast_arrays [ t1; t2 ] in
  Nd.set_item [| 0; 0 |] 99.0 t1;
  Alcotest.check float "Broadcast view after modification" 99.0
    (Nd.get_item [| 0; 0 |] b1)

let test_broadcast_arrays_invalid () =
  let t1 = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  let t2 = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  Alcotest.check_raises "Incompatible shapes"
    (Invalid_argument
       "broadcast_shapes: shapes [2] and [3] cannot be broadcast together")
    (fun () -> ignore (Nd.broadcast_arrays [ t1; t2 ]))

let test_tile_1d () =
  let t = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  let tiled = Nd.tile [| 3 |] t in
  let expected =
    Nd.create Nd.float32 [| 6 |] [| 1.0; 2.0; 1.0; 2.0; 1.0; 2.0 |]
  in
  Alcotest.check ndarray_float32 "Tiled 1D array" expected tiled

let test_tile_2d () =
  let t = Nd.create Nd.float32 [| 2; 1 |] [| 1.0; 2.0 |] in
  let tiled = Nd.tile [| 2; 3 |] t in
  Alcotest.(check (array int)) "Shape after tile" [| 4; 3 |] (Nd.shape tiled)

let test_tile_invalid () =
  let t = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  Alcotest.check_raises "Length mismatch"
    (Invalid_argument "tile: reps length must be <= tensor rank") (fun () ->
      ignore (Nd.tile [| 2; 2 |] t))

let test_repeat_axis0 () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let r = Nd.repeat ~axis:0 2 t in
  let expected =
    Nd.create Nd.float32 [| 6 |] [| 1.0; 1.0; 2.0; 2.0; 3.0; 3.0 |]
  in
  Alcotest.check ndarray_float32 "Repeated along axis 0" expected r

let test_repeat_no_axis () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let r = Nd.repeat 2 t in
  let expected =
    Nd.create Nd.float32 [| 8 |] [| 1.0; 1.0; 2.0; 2.0; 3.0; 3.0; 4.0; 4.0 |]
  in
  Alcotest.check ndarray_float32 "Repeated no axis" expected r

let test_repeat_invalid () =
  let t = Nd.create Nd.float32 [| 2 |] [| 1.0; 2.0 |] in
  Alcotest.check_raises "Negative count"
    (Invalid_argument "repeat: count must be non-negative") (fun () ->
      ignore (Nd.repeat ~axis:0 (-1) t))

let test_flip_all () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let f = Nd.flip t in
  let expected = Nd.create Nd.float32 [| 2; 2 |] [| 4.0; 3.0; 2.0; 1.0 |] in
  Alcotest.check ndarray_float32 "Flipped all axes" expected f

let test_flip_axis1 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let f = Nd.flip ~axes:[| 1 |] t in
  let expected =
    Nd.create Nd.float32 [| 2; 3 |] [| 3.0; 2.0; 1.0; 6.0; 5.0; 4.0 |]
  in
  Alcotest.check ndarray_float32 "Flipped axis 1" expected f

let test_flip_view () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let f = Nd.flip t in
  Nd.set_item [| 0; 0 |] 99.0 t;
  Alcotest.check float "Flipped view after modification" 99.0
    (Nd.get_item [| 1; 1 |] f)

let test_roll_axis0 () =
  let t = Nd.create Nd.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let r = Nd.roll ~axis:0 2 t in
  let expected = Nd.create Nd.float32 [| 4 |] [| 3.0; 4.0; 1.0; 2.0 |] in
  Alcotest.check ndarray_float32 "Rolled along axis 0" expected r

let test_roll_no_axis () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let r = Nd.roll 1 t in
  let expected = Nd.create Nd.float32 [| 2; 2 |] [| 4.0; 1.0; 2.0; 3.0 |] in
  Alcotest.check ndarray_float32 "Rolled no axis" expected r

let test_roll_negative () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let r = Nd.roll ~axis:0 (-1) t in
  let expected = Nd.create Nd.float32 [| 3 |] [| 2.0; 3.0; 1.0 |] in
  Alcotest.check ndarray_float32 "Rolled negative" expected r

let test_moveaxis () =
  let t = Nd.create Nd.float32 [| 2; 3; 4 |] (Array.init 24 float_of_int) in
  let m = Nd.moveaxis 0 2 t in
  Alcotest.(check (array int)) "Shape after moveaxis" [| 3; 4; 2 |] (Nd.shape m)

let test_moveaxis_view () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let m = Nd.moveaxis 0 1 t in
  Nd.set_item [| 0; 0 |] 99.0 t;
  Alcotest.check float "Moveaxis view after modification" 99.0
    (Nd.get_item [| 0; 0 |] m)

let test_moveaxis_invalid () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Alcotest.check_raises "Invalid axis"
    (Invalid_argument
       "moveaxis: source 2 or destination 0 out of bounds for shape [2; 2]")
    (fun () -> ignore (Nd.moveaxis 2 0 t))

let test_swapaxes () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let s = Nd.swapaxes 0 1 t in
  let expected =
    Nd.create Nd.float32 [| 3; 2 |] [| 1.0; 4.0; 2.0; 5.0; 3.0; 6.0 |]
  in
  Alcotest.check ndarray_float32 "Swapped axes" expected s

let test_swapaxes_view () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let s = Nd.swapaxes 0 1 t in
  Nd.set_item [| 0; 1 |] 99.0 t;
  Alcotest.check float "Swapaxes view after modification" 99.0
    (Nd.get_item [| 1; 0 |] s)

let test_swapaxes_invalid () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Alcotest.check_raises "Invalid axis"
    (Invalid_argument "swapaxes: axes (2, 0) out of bounds for shape [2; 2]")
    (fun () -> ignore (Nd.swapaxes 2 0 t))

(* Test Suite Organization *)
let manipulation_tests =
  [
    ("flatten 2x2", `Quick, test_flatten_2x2);
    ("flatten 1d", `Quick, test_flatten_1d);
    ("ravel contiguous view", `Quick, test_ravel_contiguous_view);
    ("ravel non-contiguous copy", `Quick, test_ravel_non_contiguous_copy);
    ("ravel 3d", `Quick, test_ravel_3d);
    ("reshape 1d to 2x2", `Quick, test_reshape_1d_to_2x2);
    ("reshape view", `Quick, test_reshape_view);
    ("reshape to vector", `Quick, test_reshape_to_vector);
    ("reshape incompatible", `Quick, test_reshape_incompatible);
    ("transpose 2d default", `Quick, test_transpose_2d_default);
    ("transpose 2d axes", `Quick, test_transpose_2d_with_axes);
    ("transpose 3d axes", `Quick, test_transpose_3d_with_axes);
    ("transpose scalar", `Quick, test_transpose_scalar);
    ("transpose view", `Quick, test_transpose_view);
    ("transpose invalid axes", `Quick, test_transpose_invalid_axes);
    ("squeeze basic", `Quick, test_squeeze_basic);
    ("squeeze specific axes", `Quick, test_squeeze_specific_axes);
    ("squeeze no-op", `Quick, test_squeeze_no_op);
    ("split 1d", `Quick, test_split_1d);
    ("split 2d axis0", `Quick, test_split_2d_axis0);
    ("split views", `Quick, test_split_views);
    ("split invalid", `Quick, test_split_invalid);
    ("array_split equal", `Quick, test_array_split_equal);
    ("array_split unequal", `Quick, test_array_split_unequal);
    ("array_split views", `Quick, test_array_split_views);
    ("concatenate 1d", `Quick, test_concatenate_1d);
    ("concatenate 2d axis0", `Quick, test_concatenate_2d_axis0);
    ("concatenate new array", `Quick, test_concatenate_new_array);
    ("concatenate invalid", `Quick, test_concatenate_invalid);
    ("stack axis0", `Quick, test_stack_axis0);
    ("stack new array", `Quick, test_stack_new_array);
    ("stack invalid", `Quick, test_stack_invalid);
    ("vstack 1d", `Quick, test_vstack_1d);
    ("vstack 2d", `Quick, test_vstack_2d);
    ("vstack invalid", `Quick, test_vstack_invalid);
    ("hstack 1d", `Quick, test_hstack_1d);
    ("hstack 2d", `Quick, test_hstack_2d);
    ("hstack invalid", `Quick, test_hstack_invalid);
    ("dstack 2d", `Quick, test_dstack_2d);
    ("dstack 1d", `Quick, test_dstack_1d);
    ("dstack invalid", `Quick, test_dstack_invalid);
    ("pad 1d", `Quick, test_pad_1d);
    ("pad 2d", `Quick, test_pad_2d);
    ("pad invalid", `Quick, test_pad_invalid);
    ("expand dims axis 0", `Quick, test_expand_dims_axis_0);
    ("expand dims axis 1", `Quick, test_expand_dims_axis_1);
    ("expand dims invalid axis", `Quick, test_expand_dims_invalid_axis);
    ("broadcast 1d to 3x3", `Quick, test_broadcast_1d_to_3x3);
    ("broadcast scalar to 2x2", `Quick, test_broadcast_scalar_to_2x2);
    ("broadcast incompatible", `Quick, test_broadcast_incompatible);
    ("broadcast_arrays", `Quick, test_broadcast_arrays);
    ("broadcast_arrays views", `Quick, test_broadcast_arrays_views);
    ("broadcast_arrays invalid", `Quick, test_broadcast_arrays_invalid);
    ("tile 1d", `Quick, test_tile_1d);
    ("tile 2d", `Quick, test_tile_2d);
    ("tile invalid", `Quick, test_tile_invalid);
    ("repeat axis0", `Quick, test_repeat_axis0);
    ("repeat no axis", `Quick, test_repeat_no_axis);
    ("repeat invalid", `Quick, test_repeat_invalid);
    ("flip all", `Quick, test_flip_all);
    ("flip axis1", `Quick, test_flip_axis1);
    ("flip view", `Quick, test_flip_view);
    ("roll axis0", `Quick, test_roll_axis0);
    ("roll no axis", `Quick, test_roll_no_axis);
    ("roll negative", `Quick, test_roll_negative);
    ("moveaxis", `Quick, test_moveaxis);
    ("moveaxis view", `Quick, test_moveaxis_view);
    ("moveaxis invalid", `Quick, test_moveaxis_invalid);
    ("swapaxes", `Quick, test_swapaxes);
    ("swapaxes view", `Quick, test_swapaxes_view);
    ("swapaxes invalid", `Quick, test_swapaxes_invalid);
  ]

let () =
  Printexc.record_backtrace true;
  Alcotest.run "Ndarray Manipulation" [ ("Manipulation", manipulation_tests) ]
