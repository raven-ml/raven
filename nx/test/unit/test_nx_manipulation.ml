(* Shape manipulation tests for Nx *)

open Alcotest

module Make (Backend : Nx_core.Backend_intf.S) = struct
  module Support = Test_nx_support.Make (Backend)
  module Nx = Support.Nx
  open Support

  (* ───── Reshape Tests ───── *)

  let test_reshape_minus_one ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3; 4 |]
        (Array.init 24 float_of_int)
    in
    (* Single -1 inference *)
    let r1 = Nx.reshape [| -1 |] t in
    check_shape "reshape [-1]" [| 24 |] r1;

    let r2 = Nx.reshape [| 2; -1 |] t in
    check_shape "reshape [2,-1]" [| 2; 12 |] r2;

    let r3 = Nx.reshape [| -1; 6 |] t in
    check_shape "reshape [-1,6]" [| 4; 6 |] r3

  let test_reshape_multiple_minus_one ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    check_invalid_arg "multiple -1"
      "reshape: invalid shape specification (multiple -1 dimensions)\n\
       hint: can only specify one unknown dimension" (fun () ->
        ignore (Nx.reshape [| -1; -1 |] t))

  let test_reshape_wrong_size ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    check_invalid_arg "wrong size"
      "reshape: cannot reshape [5] to [2,3] (5→6 elements)" (fun () ->
        ignore (Nx.reshape [| 5 |] t))

  let test_reshape_0d_to_1d ctx () =
    let t = Nx.scalar ctx Nx_core.Dtype.float32 42.0 in
    let r = Nx.reshape [| 1 |] t in
    check_t "reshape scalar to [1]" [| 1 |] [| 42.0 |] r

  let test_reshape_to_0d ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 1 |] [| 42.0 |] in
    let r = Nx.reshape [||] t in
    check_t "reshape [1] to scalar" [||] [| 42.0 |] r

  let test_reshape_empty ctx () =
    (* Empty array reshapes *)
    let t1 = Nx.create ctx Nx_core.Dtype.float32 [| 0; 5 |] [||] in
    let r1 = Nx.reshape [| 0 |] t1 in
    check_shape "reshape [0,5] to [0]" [| 0 |] r1;

    let t2 = Nx.create ctx Nx_core.Dtype.float32 [| 0; 5 |] [||] in
    let r2 = Nx.reshape [| 5; 0 |] t2 in
    check_shape "reshape [0,5] to [5,0]" [| 5; 0 |] r2

  let test_reshape_view_when_contiguous ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    let r = Nx.reshape [| 2; 2 |] t in
    (* Modify original and check if view is affected *)
    Nx.unsafe_set [ 0 ] 99.0 t;
    check (float 1e-6) "reshape view modified" 99.0 (Nx.unsafe_get [ 0; 0 ] r)

  let test_reshape_copy_when_not_contiguous ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let transposed = Nx.transpose t in
    check_invalid_arg "reshape non-contiguous"
      "reshape: cannot reshape strided view [3,2] to [6] (incompatible strides \
       [1,3] (expected [2,1]))\n\
       hint: call contiguous() before reshape to create a C-contiguous copy"
      (fun () -> Nx.reshape [| 6 |] transposed)

  let test_reshape_to_vector ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    let r = Nx.reshape [| 4; 1 |] t in
    check_t "reshape to column vector" [| 4; 1 |] [| 1.0; 2.0; 3.0; 4.0 |] r

  (* ───── Transpose Tests ───── *)

  let test_transpose_1d ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |]
    in
    let tr = Nx.transpose t in
    check_t "transpose 1D is no-op" [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] tr

  let test_transpose_0d ctx () =
    let t = Nx.scalar ctx Nx_core.Dtype.float32 42.0 in
    let tr = Nx.transpose t in
    check_t "transpose scalar is no-op" [||] [| 42.0 |] tr

  let test_transpose_high_d ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3; 4; 5 |]
        (Array.init 120 float_of_int)
    in
    let tr = Nx.transpose t in
    check_shape "transpose high-d shape" [| 5; 4; 3; 2 |] tr;
    (* Check a few values to ensure correct transpose *)
    check (float 1e-6) "transpose[0,0,0,0]" 0.0
      (Nx.unsafe_get [ 0; 0; 0; 0 ] tr);
    check (float 1e-6) "transpose[0,0,0,1]" 60.0
      (Nx.unsafe_get [ 0; 0; 0; 1 ] tr)

  let test_transpose_axes ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3; 4 |]
        (Array.init 24 float_of_int)
    in
    let tr = Nx.transpose ~axes:[| 1; 2; 0 |] t in
    check_shape "transpose custom axes" [| 3; 4; 2 |] tr;
    check (float 1e-6) "transpose[0,0,0]" 0.0 (Nx.unsafe_get [ 0; 0; 0 ] tr);
    check (float 1e-6) "transpose[0,0,1]" 12.0 (Nx.unsafe_get [ 0; 0; 1 ] tr)

  let test_transpose_invalid_axes ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    check_invalid_arg "invalid axes length"
      "transpose: invalid axes (length 3) (expected rank 2, got 3)\n\
       hint: provide exactly one axis per dimension" (fun () ->
        Nx.transpose ~axes:[| 0; 1; 2 |] t)

  let test_transpose_view ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let tr = Nx.transpose t in
    Nx.unsafe_set [ 0; 1 ] 99.0 t;
    check (float 1e-6) "transpose view modified" 99.0
      (Nx.unsafe_get [ 1; 0 ] tr)

  (* ───── Concatenate Tests ───── *)

  let test_concat_axis_1 ctx () =
    let t1 =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let t2 =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 7.; 8.; 9.; 10. |]
    in
    let c = Nx.concatenate ~axis:1 [ t1; t2 ] in
    check_t "concat axis 1" [| 2; 5 |]
      [| 1.; 2.; 3.; 7.; 8.; 4.; 5.; 6.; 9.; 10. |]
      c

  let test_concat_single ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let c = Nx.concatenate [ t ] in
    check_t "concat single array" [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] c

  let test_concat_empty_list _ctx () =
    check_invalid_arg "concat empty list"
      "concatenate: invalid tensor list (empty)\n\
       hint: provide at least one tensor" (fun () -> Nx.concatenate [])

  let test_concat_different_dtypes ctx () =
    (* For now, assuming concatenate requires same dtype - adjust if it
       promotes *)
    let t1 = Nx.create ctx Nx_core.Dtype.float32 [| 2 |] [| 1.0; 2.0 |] in
    let t2 = Nx.create ctx Nx_core.Dtype.int32 [| 2 |] [| 3l; 4l |] in
    check_invalid_arg "concat different dtypes"
      "concatenate: cannot concatenate float32 to with int32 (dtype mismatch)\n\
       hint: cast one array to float32" (fun () ->
        ignore (Nx.concatenate [ t1; Obj.magic t2 ]))

  let test_concat_with_empty ctx () =
    let t1 =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let t2 = Nx.create ctx Nx_core.Dtype.float32 [| 0; 3 |] [||] in
    let c = Nx.concatenate ~axis:0 [ t1; t2 ] in
    check_t "concat with empty" [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] c

  let test_concat_shape_mismatch ctx () =
    let t1 =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let t2 =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 4 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
    in
    check_invalid_arg "shape mismatch"
      "concatenate: invalid dimension 1 (size 4\226\137\1603)" (fun () ->
        Nx.concatenate ~axis:0 [ t1; t2 ])

  let test_concat_new_array ctx () =
    let t1 = Nx.create ctx Nx_core.Dtype.float32 [| 2 |] [| 1.0; 2.0 |] in
    let t2 = Nx.create ctx Nx_core.Dtype.float32 [| 2 |] [| 3.0; 4.0 |] in
    let c = Nx.concatenate [ t1; t2 ] in
    Nx.unsafe_set [ 0 ] 99.0 t1;
    check (float 1e-6) "concat is new array" 1.0 (Nx.unsafe_get [ 0 ] c)

  (* ───── Stack Tests ───── *)

  let test_stack_new_axis ctx () =
    let t1 =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let t2 =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 7.; 8.; 9.; 10.; 11.; 12. |]
    in
    let s = Nx.stack ~axis:1 [ t1; t2 ] in
    check_shape "stack axis 1 shape" [| 2; 2; 3 |] s;
    check_t "stack axis 1 values" [| 2; 2; 3 |]
      [| 1.; 2.; 3.; 7.; 8.; 9.; 4.; 5.; 6.; 10.; 11.; 12. |]
      s

  let test_stack_shape_mismatch ctx () =
    let t1 = Nx.create ctx Nx_core.Dtype.float32 [| 3 |] [| 1.; 2.; 3. |] in
    let t2 = Nx.create ctx Nx_core.Dtype.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
    check_invalid_arg "stack shape mismatch"
      "concatenate: invalid dimension 1 (size 4\226\137\1603)" (fun () ->
        Nx.stack ~axis:0 [ t1; t2 ])

  let test_stack_new_array ctx () =
    let t1 = Nx.create ctx Nx_core.Dtype.float32 [| 2 |] [| 1.0; 2.0 |] in
    let t2 = Nx.create ctx Nx_core.Dtype.float32 [| 2 |] [| 3.0; 4.0 |] in
    let s = Nx.stack ~axis:0 [ t1; t2 ] in
    Nx.unsafe_set [ 0 ] 99.0 t1;
    check (float 1e-6) "stack is new array" 1.0 (Nx.unsafe_get [ 0; 0 ] s)

  (* ───── Split Tests ───── *)

  let test_split_equal ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 12 |] (Array.init 12 float_of_int)
    in
    let parts = Nx.split ~axis:0 3 t in
    check int "split count" 3 (List.length parts);
    check_t "split part 0" [| 4 |] [| 0.; 1.; 2.; 3. |] (List.nth parts 0);
    check_t "split part 1" [| 4 |] [| 4.; 5.; 6.; 7. |] (List.nth parts 1);
    check_t "split part 2" [| 4 |] [| 8.; 9.; 10.; 11. |] (List.nth parts 2)

  let test_split_unequal ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 10 |] (Array.init 10 float_of_int)
    in
    check_invalid_arg "split unequal"
      "split: cannot divide evenly axis 0 (size 10) to 3 sections (10 % 3 = 1)\n\
       hint: use array_split for uneven division" (fun () ->
        ignore (Nx.split ~axis:0 3 t))

  let test_split_axis ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 4; 6 |]
        (Array.init 24 float_of_int)
    in
    let parts = Nx.split ~axis:1 2 t in
    check int "split axis 1 count" 2 (List.length parts);
    check_shape "split axis 1 shape" [| 4; 3 |] (List.nth parts 0)

  let test_split_one ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let parts = Nx.split ~axis:0 1 t in
    check int "split into 1 count" 1 (List.length parts);
    check_t "split into 1 part" [| 6 |]
      [| 1.; 2.; 3.; 4.; 5.; 6. |]
      (List.nth parts 0)

  let test_split_views ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    let parts = Nx.split ~axis:0 2 t in
    let p1 = List.nth parts 0 in
    Nx.unsafe_set [ 0 ] 99.0 p1;
    check (float 1e-6) "split view modified" 99.0 (Nx.unsafe_get [ 0 ] t)

  (* ───── Array Split Tests ───── *)

  let test_array_split_equal ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 6 |]
        [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    in
    let parts = Nx.array_split ~axis:0 (`Count 3) t in
    check int "array_split equal count" 3 (List.length parts);
    check_t "array_split equal part 0" [| 2 |] [| 1.0; 2.0 |] (List.nth parts 0)

  let test_array_split_unequal ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |]
    in
    let parts = Nx.array_split ~axis:0 (`Count 3) t in
    check int "array_split unequal count" 3 (List.length parts);
    check_t "array_split unequal part 0" [| 2 |] [| 1.0; 2.0 |]
      (List.nth parts 0);
    check_t "array_split unequal part 1" [| 2 |] [| 3.0; 4.0 |]
      (List.nth parts 1);
    check_t "array_split unequal part 2" [| 1 |] [| 5.0 |] (List.nth parts 2)

  let test_array_split_views ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |]
    in
    let parts = Nx.array_split ~axis:0 (`Count 2) t in
    let p1 = List.nth parts 0 in
    Nx.unsafe_set [ 0 ] 99.0 p1;
    check (float 1e-6) "array_split view modified" 99.0 (Nx.unsafe_get [ 0 ] t)

  (* ───── Squeeze Expand Tests ───── *)

  let test_squeeze_all ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 1; 3; 1; 4; 1 |]
        (Array.init 12 float_of_int)
    in
    let s = Nx.squeeze t in
    check_shape "squeeze all" [| 3; 4 |] s

  let test_squeeze_specific ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 1; 3; 1; 4 |]
        (Array.init 12 float_of_int)
    in
    let s = Nx.squeeze ~axes:[| 0; 2 |] t in
    check_shape "squeeze specific axes" [| 3; 4 |] s

  let test_squeeze_no_ones ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3; 4 |]
        (Array.init 24 float_of_int)
    in
    let s = Nx.squeeze t in
    check_shape "squeeze no ones" [| 2; 3; 4 |] s

  let test_squeeze_invalid_axis ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    check_invalid_arg "squeeze invalid axis"
      "squeeze: cannot remove dimension axis 1 (size 3) to squeezed (size \
       3\226\137\1601)" (fun () -> ignore (Nx.squeeze ~axes:[| 1 |] t))

  let test_expand_dims_various ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 3 |] [| 1.; 2.; 3. |] in

    (* Add dim at position 0 *)
    let e0 = Nx.expand_dims [| 0 |] t in
    check_shape "expand_dims at 0" [| 1; 3 |] e0;

    (* Add dim at position -1 (end) *)
    let e_end = Nx.expand_dims [| -1 |] t in
    check_shape "expand_dims at -1" [| 3; 1 |] e_end;

    (* Add dim in middle *)
    let t2 =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let e_mid = Nx.expand_dims [| 1 |] t2 in
    check_shape "expand_dims in middle" [| 2; 1; 3 |] e_mid

  let test_expand_dims_invalid_axis ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
    check_invalid_arg "expand_dims invalid axis"
      "unsqueeze: invalid axis 2 (out of bounds for output rank 2)\n\
       hint: valid range is [-2, 2)" (fun () -> Nx.expand_dims [| 2 |] t)

  (* ───── Broadcasting Tests ───── *)

  let test_broadcast_to_valid ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 3; 1 |] [| 1.; 2.; 3. |] in
    let b = Nx.broadcast_to [| 3; 4 |] t in
    check_shape "broadcast valid shape" [| 3; 4 |] b;
    check (float 1e-6) "broadcast[0,0]" 1.0 (Nx.unsafe_get [ 0; 0 ] b);
    check (float 1e-6) "broadcast[0,3]" 1.0 (Nx.unsafe_get [ 0; 3 ] b);
    check (float 1e-6) "broadcast[2,2]" 3.0 (Nx.unsafe_get [ 2; 2 ] b)

  let test_broadcast_to_invalid ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 3 |] [| 1.; 2.; 3. |] in
    check_invalid_arg "broadcast invalid"
      "broadcast_to: cannot broadcast [3] to [4] (dim 0: 3≠4)\n\
       hint: broadcasting requires dimensions to be either equal or 1"
      (fun () -> ignore (Nx.broadcast_to [| 4 |] t))

  let test_broadcast_to_same ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 3; 4 |]
        (Array.init 12 float_of_int)
    in
    let b = Nx.broadcast_to [| 3; 4 |] t in
    check_t "broadcast to same" [| 3; 4 |] (Array.init 12 float_of_int) b

  let test_broadcast_scalar ctx () =
    let t = Nx.scalar ctx Nx_core.Dtype.float32 5.0 in
    let b = Nx.broadcast_to [| 3; 4; 5 |] t in
    check_shape "broadcast scalar shape" [| 3; 4; 5 |] b;
    check (float 1e-6) "broadcast scalar value" 5.0
      (Nx.unsafe_get [ 2; 3; 4 ] b)

  let test_broadcast_arrays_compatible ctx () =
    let t1 =
      Nx.create ctx Nx_core.Dtype.float32 [| 3; 1 |] [| 1.0; 2.0; 3.0 |]
    in
    let t2 =
      Nx.create ctx Nx_core.Dtype.float32 [| 1; 4 |]
        [| 10.0; 20.0; 30.0; 40.0 |]
    in
    let broadcasted = Nx.broadcast_arrays [ t1; t2 ] in
    check int "broadcast_arrays count" 2 (List.length broadcasted);
    let b1 = List.nth broadcasted 0 in
    let b2 = List.nth broadcasted 1 in
    check_shape "broadcast_arrays shape 1" [| 3; 4 |] b1;
    check_shape "broadcast_arrays shape 2" [| 3; 4 |] b2

  let test_broadcast_arrays_views ctx () =
    let t1 =
      Nx.create ctx Nx_core.Dtype.float32 [| 3; 1 |] [| 1.0; 2.0; 3.0 |]
    in
    let t2 = Nx.create ctx Nx_core.Dtype.float32 [| 1; 1 |] [| 10.0 |] in
    let broadcasted = Nx.broadcast_arrays [ t1; t2 ] in
    let b1 = List.nth broadcasted 0 in
    Nx.unsafe_set [ 0; 0 ] 99.0 t1;
    check (float 1e-6) "broadcast array view modified" 99.0
      (Nx.unsafe_get [ 0; 0 ] b1)

  let test_broadcast_arrays_invalid ctx () =
    let t1 = Nx.create ctx Nx_core.Dtype.float32 [| 2 |] [| 1.0; 2.0 |] in
    let t2 = Nx.create ctx Nx_core.Dtype.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
    check_invalid_arg "broadcast_arrays invalid"
      "broadcast: cannot broadcast [2] to [3] (dim 0: 2\226\137\1603)\n\
       hint: broadcasting requires dimensions to be either equal or 1"
      (fun () -> Nx.broadcast_arrays [ t1; t2 ])

  (* ───── Tile Repeat Tests ───── *)

  let test_tile_1d ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 3 |] [| 1.; 2.; 3. |] in
    let tiled = Nx.tile [| 2 |] t in
    check_t "tile 1d" [| 6 |] [| 1.; 2.; 3.; 1.; 2.; 3. |] tiled

  let test_tile_2d ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let tiled = Nx.tile [| 2; 1 |] t in
    check_t "tile 2d" [| 4; 3 |]
      [| 1.; 2.; 3.; 4.; 5.; 6.; 1.; 2.; 3.; 4.; 5.; 6. |]
      tiled

  let test_tile_broadcast ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 3 |] [| 1.; 2.; 3. |] in
    let tiled = Nx.tile [| 2; 3 |] t in
    check_shape "tile broadcast shape" [| 2; 9 |] tiled;
    check_t "tile broadcast" [| 2; 9 |]
      [|
        1.; 2.; 3.; 1.; 2.; 3.; 1.; 2.; 3.; 1.; 2.; 3.; 1.; 2.; 3.; 1.; 2.; 3.;
      |]
      tiled

  let test_tile_invalid ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 2 |] [| 1.0; 2.0 |] in
    (* This test is incorrect - tile should work with more reps than tensor dims
       by promoting the tensor. Let's test a different invalid case. *)
    check_invalid_arg "tile invalid"
      "tile: invalid reps[0] (negative (-1<0))\n\
       hint: use positive integers (or 0 for empty result)" (fun () ->
        Nx.tile [| -1 |] t)

  let test_repeat_axis ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let r = Nx.repeat ~axis:0 2 t in
    check_t "repeat axis 0" [| 4; 3 |]
      [| 1.; 2.; 3.; 1.; 2.; 3.; 4.; 5.; 6.; 4.; 5.; 6. |]
      r

  let test_repeat_no_axis ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let r = Nx.repeat 2 t in
    check_t "repeat no axis" [| 12 |]
      [| 1.; 1.; 2.; 2.; 3.; 3.; 4.; 4.; 5.; 5.; 6.; 6. |]
      r

  let test_repeat_invalid ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 2 |] [| 1.0; 2.0 |] in
    check_invalid_arg "repeat negative" "repeat: invalid count (count=-1 < 0)"
      (fun () -> Nx.repeat ~axis:0 (-1) t)

  (* ───── Other Shape Manipulation Tests ───── *)

  let test_flatten_view ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    let flat = Nx.flatten t in
    Nx.unsafe_set [ 0; 0 ] 99.0 t;
    check (float 1e-6) "flatten view modified" 99.0 (Nx.unsafe_get [ 0 ] flat)

  let test_ravel_contiguous_view ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    let r = Nx.ravel t in
    Nx.unsafe_set [ 0; 0 ] 99.0 t;
    check (float 1e-6) "ravel view modified" 99.0 (Nx.unsafe_get [ 0 ] r)

  let test_ravel_non_contiguous_copy ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    let tr = Nx.transpose t in
    check_invalid_arg "ravel non-contiguous"
      "reshape: cannot reshape strided view [2,2] to [4] (incompatible strides \
       [1,2] (expected [2,1]))\n\
       hint: call contiguous() before reshape to create a C-contiguous copy"
      (fun () -> Nx.ravel tr)

  let test_pad_2d ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    let p = Nx.pad [| (1, 1); (0, 1) |] 0.0 t in
    check_t "pad 2d" [| 4; 3 |]
      [| 0.0; 0.0; 0.0; 1.0; 2.0; 0.0; 3.0; 4.0; 0.0; 0.0; 0.0; 0.0 |]
      p

  let test_pad_invalid ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 2 |] [| 1.0; 2.0 |] in
    check_invalid_arg "pad negative"
      "pad: invalid padding values (negative values not allowed)\n\
       hint: use shrink or slice to remove elements" (fun () ->
        Nx.pad [| (-1, 2) |] 0.0 t)

  let test_flip_axis ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    in
    let f = Nx.flip ~axes:[| 1 |] t in
    check_t "flip axis 1" [| 2; 3 |] [| 3.0; 2.0; 1.0; 6.0; 5.0; 4.0 |] f

  let test_flip_view ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    let f = Nx.flip t in
    Nx.unsafe_set [ 0; 0 ] 99.0 t;
    check (float 1e-6) "flip view modified" 99.0 (Nx.unsafe_get [ 1; 1 ] f)

  let test_roll_no_axis ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    let r = Nx.roll 1 t in
    check_t "roll no axis" [| 2; 2 |] [| 4.0; 1.0; 2.0; 3.0 |] r

  let test_roll_negative ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
    let r = Nx.roll ~axis:0 (-1) t in
    check_t "roll negative" [| 3 |] [| 2.0; 3.0; 1.0 |] r

  let test_moveaxis_view ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    let m = Nx.moveaxis 0 1 t in
    Nx.unsafe_set [ 0; 0 ] 99.0 t;
    check (float 1e-6) "moveaxis view modified" 99.0 (Nx.unsafe_get [ 0; 0 ] m)

  let test_moveaxis_invalid ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    check_invalid_arg "moveaxis invalid"
      "moveaxis: invalid source 2 or destination 0 (out of bounds for shape \
       [2,2])" (fun () -> Nx.moveaxis 2 0 t)

  let test_swapaxes_view ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    let s = Nx.swapaxes 0 1 t in
    Nx.unsafe_set [ 0; 1 ] 99.0 t;
    check (float 1e-6) "swapaxes view modified" 99.0 (Nx.unsafe_get [ 1; 0 ] s)

  let test_swapaxes_invalid ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    check_invalid_arg "swapaxes invalid"
      "swapaxes: invalid axes (2, 0) (out of bounds for shape [2,2])" (fun () ->
        Nx.swapaxes 2 0 t)

  let test_dstack_1d ctx () =
    let t1 = Nx.create ctx Nx_core.Dtype.float32 [| 2 |] [| 1.0; 2.0 |] in
    let t2 = Nx.create ctx Nx_core.Dtype.float32 [| 2 |] [| 3.0; 4.0 |] in
    let d = Nx.dstack [ t1; t2 ] in
    check_t "dstack 1d" [| 1; 2; 2 |] [| 1.0; 3.0; 2.0; 4.0 |] d

  let test_vstack_invalid ctx () =
    let t1 =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    let t2 =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    in
    check_invalid_arg "vstack invalid"
      "concatenate: invalid dimension 1 (size 3\226\137\1602)" (fun () ->
        Nx.vstack [ t1; t2 ])

  let test_hstack_invalid ctx () =
    let t1 =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    let t2 =
      Nx.create ctx Nx_core.Dtype.float32 [| 3; 2 |]
        [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    in
    check_invalid_arg "hstack invalid"
      "concatenate: invalid dimension 0 (size 3\226\137\1602)" (fun () ->
        Nx.hstack [ t1; t2 ])

  let test_dstack_invalid ctx () =
    let t1 =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
    in
    let t2 =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
    in
    check_invalid_arg "dstack invalid"
      "concatenate: invalid dimension 1 (size 3\226\137\1602)" (fun () ->
        Nx.dstack [ t1; t2 ])

  (* ───── With View Tests ───── *)

  let test_with_view_basic ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 2; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let new_view = Nx_core.View.create [| 3; 2 |] in
    let t_reshaped = Nx.with_view t new_view in

    check_shape "with_view reshape" [| 3; 2 |] t_reshaped;
    check_t "with_view data preserved" [| 3; 2 |]
      [| 1.; 2.; 3.; 4.; 5.; 6. |]
      t_reshaped

  let test_with_view_offset ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let offset_view = Nx_core.View.create ~offset:2 [| 3 |] in
    let t_offset = Nx.with_view t offset_view in

    check_shape "with_view offset shape" [| 3 |] t_offset;
    check_t "with_view offset data" [| 3 |] [| 3.; 4.; 5. |] t_offset

  let test_with_view_strided ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
    in
    let strided_view = Nx_core.View.create ~strides:[| 2 |] [| 3 |] in
    let t_strided = Nx.with_view t strided_view in

    check_shape "with_view strided shape" [| 3 |] t_strided;
    check_t "with_view strided data" [| 3 |] [| 1.; 3.; 5. |] t_strided

  let test_with_view_scalar ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 1 |] [| 42. |] in
    let scalar_view = Nx_core.View.create [||] in
    let t_scalar = Nx.with_view t scalar_view in

    check_shape "with_view scalar shape" [||] t_scalar;
    check_t "with_view scalar data" [||] [| 42. |] t_scalar

  let test_with_view_shared_data ctx () =
    let t = Nx.create ctx Nx_core.Dtype.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
    let view_2x2 = Nx_core.View.create [| 2; 2 |] in
    let t_view = Nx.with_view t view_2x2 in

    Nx.unsafe_set [ 0 ] 99. t;

    check (float 1e-6) "with_view shared data" 99.
      (Nx.unsafe_get [ 0; 0 ] t_view)

  let test_with_view_complex_strides ctx () =
    let t =
      Nx.create ctx Nx_core.Dtype.float32 [| 12 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
    in

    let complex_view = Nx_core.View.create ~strides:[| 6; 2 |] [| 2; 2 |] in
    let t_complex = Nx.with_view t complex_view in

    check_shape "with_view complex shape" [| 2; 2 |] t_complex;
    check (float 1e-6) "with_view complex [0,0]" 1.
      (Nx.unsafe_get [ 0; 0 ] t_complex);
    check (float 1e-6) "with_view complex [0,1]" 3.
      (Nx.unsafe_get [ 0; 1 ] t_complex);
    check (float 1e-6) "with_view complex [1,0]" 7.
      (Nx.unsafe_get [ 1; 0 ] t_complex);
    check (float 1e-6) "with_view complex [1,1]" 9.
      (Nx.unsafe_get [ 1; 1 ] t_complex)

  let test_with_view_different_dtypes ctx () =
    let t_int =
      Nx.create ctx Nx_core.Dtype.int32 [| 4 |] [| 1l; 2l; 3l; 4l |]
    in
    let view_2x2 = Nx_core.View.create [| 2; 2 |] in
    let t_view = Nx.with_view t_int view_2x2 in

    check_shape "with_view int32 shape" [| 2; 2 |] t_view;
    check_t "with_view int32 data" [| 2; 2 |] [| 1l; 2l; 3l; 4l |] t_view

  (* Test Suite Organization *)

  let with_view_tests ctx =
    [
      ("with_view basic", `Quick, test_with_view_basic ctx);
      ("with_view offset", `Quick, test_with_view_offset ctx);
      ("with_view strided", `Quick, test_with_view_strided ctx);
      ("with_view scalar", `Quick, test_with_view_scalar ctx);
      ("with_view shared data", `Quick, test_with_view_shared_data ctx);
      ("with_view complex strides", `Quick, test_with_view_complex_strides ctx);
      ("with_view different dtypes", `Quick, test_with_view_different_dtypes ctx);
    ]

  let reshape_tests ctx =
    [
      ("reshape minus one", `Quick, test_reshape_minus_one ctx);
      ("reshape multiple minus one", `Quick, test_reshape_multiple_minus_one ctx);
      ("reshape wrong size", `Quick, test_reshape_wrong_size ctx);
      ("reshape 0d to 1d", `Quick, test_reshape_0d_to_1d ctx);
      ("reshape to 0d", `Quick, test_reshape_to_0d ctx);
      ("reshape empty", `Quick, test_reshape_empty ctx);
      ( "reshape view when contiguous",
        `Quick,
        test_reshape_view_when_contiguous ctx );
      ( "reshape copy when not contiguous",
        `Quick,
        test_reshape_copy_when_not_contiguous ctx );
      ("reshape to vector", `Quick, test_reshape_to_vector ctx);
    ]

  let transpose_tests ctx =
    [
      ("transpose 1d", `Quick, test_transpose_1d ctx);
      ("transpose 0d", `Quick, test_transpose_0d ctx);
      ("transpose high d", `Quick, test_transpose_high_d ctx);
      ("transpose axes", `Quick, test_transpose_axes ctx);
      ("transpose invalid axes", `Quick, test_transpose_invalid_axes ctx);
      ("transpose view", `Quick, test_transpose_view ctx);
    ]

  let concatenate_tests ctx =
    [
      ("concat axis 1", `Quick, test_concat_axis_1 ctx);
      ("concat single", `Quick, test_concat_single ctx);
      ("concat empty list", `Quick, test_concat_empty_list ctx);
      ("concat different dtypes", `Quick, test_concat_different_dtypes ctx);
      ("concat with empty", `Quick, test_concat_with_empty ctx);
      ("concat shape mismatch", `Quick, test_concat_shape_mismatch ctx);
      ("concat new array", `Quick, test_concat_new_array ctx);
    ]

  let stack_tests ctx =
    [
      ("stack new axis", `Quick, test_stack_new_axis ctx);
      ("stack shape mismatch", `Quick, test_stack_shape_mismatch ctx);
      ("stack new array", `Quick, test_stack_new_array ctx);
    ]

  let split_tests ctx =
    [
      ("split equal", `Quick, test_split_equal ctx);
      ("split unequal", `Quick, test_split_unequal ctx);
      ("split axis", `Quick, test_split_axis ctx);
      ("split one", `Quick, test_split_one ctx);
      ("split views", `Quick, test_split_views ctx);
      ("array split equal", `Quick, test_array_split_equal ctx);
      ("array split unequal", `Quick, test_array_split_unequal ctx);
      ("array split views", `Quick, test_array_split_views ctx);
    ]

  let squeeze_expand_tests ctx =
    [
      ("squeeze all", `Quick, test_squeeze_all ctx);
      ("squeeze specific", `Quick, test_squeeze_specific ctx);
      ("squeeze no ones", `Quick, test_squeeze_no_ones ctx);
      ("squeeze invalid axis", `Quick, test_squeeze_invalid_axis ctx);
      ("expand dims various", `Quick, test_expand_dims_various ctx);
      ("expand dims invalid axis", `Quick, test_expand_dims_invalid_axis ctx);
    ]

  let broadcast_tests ctx =
    [
      ("broadcast to valid", `Quick, test_broadcast_to_valid ctx);
      ("broadcast to invalid", `Quick, test_broadcast_to_invalid ctx);
      ("broadcast to same", `Quick, test_broadcast_to_same ctx);
      ("broadcast scalar", `Quick, test_broadcast_scalar ctx);
      ( "broadcast arrays compatible",
        `Quick,
        test_broadcast_arrays_compatible ctx );
      ("broadcast arrays views", `Quick, test_broadcast_arrays_views ctx);
      ("broadcast arrays invalid", `Quick, test_broadcast_arrays_invalid ctx);
    ]

  let tile_repeat_tests ctx =
    [
      ("tile 1d", `Quick, test_tile_1d ctx);
      ("tile 2d", `Quick, test_tile_2d ctx);
      ("tile broadcast", `Quick, test_tile_broadcast ctx);
      ("tile invalid", `Quick, test_tile_invalid ctx);
      ("repeat axis", `Quick, test_repeat_axis ctx);
      ("repeat no axis", `Quick, test_repeat_no_axis ctx);
      ("repeat invalid", `Quick, test_repeat_invalid ctx);
    ]

  let other_manipulation_tests ctx =
    [
      ("flatten view", `Quick, test_flatten_view ctx);
      ("ravel contiguous view", `Quick, test_ravel_contiguous_view ctx);
      ("ravel non-contiguous copy", `Quick, test_ravel_non_contiguous_copy ctx);
      ("pad 2d", `Quick, test_pad_2d ctx);
      ("pad invalid", `Quick, test_pad_invalid ctx);
      ("flip axis", `Quick, test_flip_axis ctx);
      ("flip view", `Quick, test_flip_view ctx);
      ("roll no axis", `Quick, test_roll_no_axis ctx);
      ("roll negative", `Quick, test_roll_negative ctx);
      ("moveaxis view", `Quick, test_moveaxis_view ctx);
      ("moveaxis invalid", `Quick, test_moveaxis_invalid ctx);
      ("swapaxes view", `Quick, test_swapaxes_view ctx);
      ("swapaxes invalid", `Quick, test_swapaxes_invalid ctx);
      ("dstack 1d", `Quick, test_dstack_1d ctx);
      ("vstack invalid", `Quick, test_vstack_invalid ctx);
      ("hstack invalid", `Quick, test_hstack_invalid ctx);
      ("dstack invalid", `Quick, test_dstack_invalid ctx);
    ]

  let suite backend_name ctx =
    [
      ("Manipulation :: " ^ backend_name ^ " With View", with_view_tests ctx);
      ("Manipulation :: " ^ backend_name ^ " Reshape", reshape_tests ctx);
      ("Manipulation :: " ^ backend_name ^ " Transpose", transpose_tests ctx);
      ("Manipulation :: " ^ backend_name ^ " Concatenate", concatenate_tests ctx);
      ("Manipulation :: " ^ backend_name ^ " Stack", stack_tests ctx);
      ("Manipulation :: " ^ backend_name ^ " Split", split_tests ctx);
      ( "Manipulation :: " ^ backend_name ^ " Squeeze/Expand",
        squeeze_expand_tests ctx );
      ("Manipulation :: " ^ backend_name ^ " Broadcasting", broadcast_tests ctx);
      ("Manipulation :: " ^ backend_name ^ " Tile/Repeat", tile_repeat_tests ctx);
      ( "Manipulation :: " ^ backend_name ^ " Other Manipulation",
        other_manipulation_tests ctx );
    ]
end
