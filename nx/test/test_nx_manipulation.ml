(* Shape manipulation tests for Nx *)

open Alcotest
open Test_nx_support

(* ───── Reshape Tests ───── *)

let test_reshape_minus_one () =
  let t = Nx.create Nx.float32 [| 2; 3; 4 |] (Array.init 24 float_of_int) in
  (* Single -1 inference *)
  let r1 = Nx.reshape [| -1 |] t in
  check_shape "reshape [-1]" [| 24 |] r1;

  let r2 = Nx.reshape [| 2; -1 |] t in
  check_shape "reshape [2,-1]" [| 2; 12 |] r2;

  let r3 = Nx.reshape [| -1; 6 |] t in
  check_shape "reshape [-1,6]" [| 4; 6 |] r3

let test_reshape_multiple_minus_one () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  check_invalid_arg "multiple -1"
    "reshape: invalid shape specification (multiple -1 dimensions)\n\
     hint: can only specify one unknown dimension" (fun () ->
      ignore (Nx.reshape [| -1; -1 |] t))

let test_reshape_wrong_size () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  check_invalid_arg "wrong size"
    "reshape: cannot reshape [5] to [2,3] (5→6 elements)" (fun () ->
      ignore (Nx.reshape [| 5 |] t))

let test_reshape_0d_to_1d () =
  let t = Nx.scalar Nx.float32 42.0 in
  let r = Nx.reshape [| 1 |] t in
  check_t "reshape scalar to [1]" [| 1 |] [| 42.0 |] r

let test_reshape_to_0d () =
  let t = Nx.create Nx.float32 [| 1 |] [| 42.0 |] in
  let r = Nx.reshape [||] t in
  check_t "reshape [1] to scalar" [||] [| 42.0 |] r

let test_reshape_empty () =
  (* Empty array reshapes *)
  let t1 = Nx.create Nx.float32 [| 0; 5 |] [||] in
  let r1 = Nx.reshape [| 0 |] t1 in
  check_shape "reshape [0,5] to [0]" [| 0 |] r1;

  let t2 = Nx.create Nx.float32 [| 0; 5 |] [||] in
  let r2 = Nx.reshape [| 5; 0 |] t2 in
  check_shape "reshape [0,5] to [5,0]" [| 5; 0 |] r2

let test_reshape_view_when_contiguous () =
  let t = Nx.create Nx.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let r = Nx.reshape [| 2; 2 |] t in
  Nx.set_item [ 0 ] 77.0 t;
  check (float 1e-6) "reshape view sees source mutations" 77.0
    (Nx.item [ 0; 0 ] r);
  Nx.set_item [ 0; 0 ] 42.0 r;
  check (float 1e-6) "reshape view mutates source" 42.0 (Nx.item [ 0 ] t)

let test_reshape_copy_when_not_contiguous () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let transposed = Nx.transpose t in
  check_invalid_arg "reshape non-contiguous"
    "reshape: cannot reshape strided view [3,2] to [6] (incompatible strides \
     [1,3] (expected [1]))\n\
     hint: call contiguous() before reshape to create a C-contiguous copy"
    (fun () -> Nx.reshape [| 6 |] transposed)

let test_reshape_to_vector () =
  let t = Nx.create Nx.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let r = Nx.reshape [| 4; 1 |] t in
  check_t "reshape to column vector" [| 4; 1 |] [| 1.0; 2.0; 3.0; 4.0 |] r

(* ───── Transpose Tests ───── *)

let test_transpose_1d () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let tr = Nx.transpose t in
  check_t "transpose 1D is no-op" [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] tr

let test_transpose_0d () =
  let t = Nx.scalar Nx.float32 42.0 in
  let tr = Nx.transpose t in
  check_t "transpose scalar is no-op" [||] [| 42.0 |] tr

let test_transpose_high_d () =
  let t = Nx.create Nx.float32 [| 2; 3; 4; 5 |] (Array.init 120 float_of_int) in
  let tr = Nx.transpose t in
  check_shape "transpose high-d shape" [| 5; 4; 3; 2 |] tr;
  (* Check a few values to ensure correct transpose *)
  check (float 1e-6) "transpose[0,0,0,0]" 0.0 (Nx.item [ 0; 0; 0; 0 ] tr);
  check (float 1e-6) "transpose[0,0,0,1]" 60.0 (Nx.item [ 0; 0; 0; 1 ] tr)

let test_transpose_axes () =
  let t = Nx.create Nx.float32 [| 2; 3; 4 |] (Array.init 24 float_of_int) in
  let tr = Nx.transpose ~axes:[ 1; 2; 0 ] t in
  check_shape "transpose custom axes" [| 3; 4; 2 |] tr;
  check (float 1e-6) "transpose[0,0,0]" 0.0 (Nx.item [ 0; 0; 0 ] tr);
  check (float 1e-6) "transpose[0,0,1]" 12.0 (Nx.item [ 0; 0; 1 ] tr)

let test_transpose_invalid_axes () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  check_invalid_arg "invalid axes length"
    "transpose: invalid axes (length 3) (expected rank 2, got 3)\n\
     hint: provide exactly one axis per dimension" (fun () ->
      Nx.transpose ~axes:[ 0; 1; 2 ] t)

let test_transpose_view () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let tr = Nx.transpose t in
  Nx.set_item [ 0; 1 ] 99.0 t;
  check (float 1e-6) "transpose view modified" 99.0 (Nx.item [ 1; 0 ] tr)

(* ───── Concatenate Tests ───── *)

let test_concat_axis_1 () =
  let t1 = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let t2 = Nx.create Nx.float32 [| 2; 2 |] [| 7.; 8.; 9.; 10. |] in
  let c = Nx.concatenate ~axis:1 [ t1; t2 ] in
  check_t "concat axis 1" [| 2; 5 |]
    [| 1.; 2.; 3.; 7.; 8.; 4.; 5.; 6.; 9.; 10. |]
    c

let test_concat_single () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let c = Nx.concatenate [ t ] in
  check_t "concat single array" [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] c

let test_concat_empty_list () =
  check_invalid_arg "concat empty list"
    "concatenate: invalid tensor list (empty)\n\
     hint: provide at least one tensor" (fun () -> Nx.concatenate [])

let test_concat_different_dtypes () =
  (* For now, assuming concatenate requires same dtype - adjust if it
     promotes *)
  let t1 = Nx.create Nx.float32 [| 2 |] [| 1.0; 2.0 |] in
  let t2 = Nx.create Nx.int32 [| 2 |] [| 3l; 4l |] in
  check_invalid_arg "concat different dtypes"
    "concatenate: cannot concatenate float32 to with int32 (dtype mismatch)\n\
     hint: cast one array to float32" (fun () ->
      ignore (Nx.concatenate [ t1; Obj.magic t2 ]))

let test_concat_with_empty () =
  let t1 = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let t2 = Nx.create Nx.float32 [| 0; 3 |] [||] in
  let c = Nx.concatenate ~axis:0 [ t1; t2 ] in
  check_t "concat with empty" [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] c

let test_concat_shape_mismatch () =
  let t1 = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let t2 =
    Nx.create Nx.float32 [| 2; 4 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
  in
  check_invalid_arg "shape mismatch"
    "concatenate: invalid dimension 1 (size 4\226\137\1603)" (fun () ->
      Nx.concatenate ~axis:0 [ t1; t2 ])

let test_concat_new_array () =
  let t1 = Nx.create Nx.float32 [| 2 |] [| 1.0; 2.0 |] in
  let t2 = Nx.create Nx.float32 [| 2 |] [| 3.0; 4.0 |] in
  let c = Nx.concatenate [ t1; t2 ] in
  Nx.set_item [ 0 ] 99.0 t1;
  check (float 1e-6) "concat is new array" 1.0 (Nx.item [ 0 ] c)

(* ───── Stack Tests ───── *)

let test_stack_new_axis () =
  let t1 = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let t2 = Nx.create Nx.float32 [| 2; 3 |] [| 7.; 8.; 9.; 10.; 11.; 12. |] in
  let s = Nx.stack ~axis:1 [ t1; t2 ] in
  check_shape "stack axis 1 shape" [| 2; 2; 3 |] s;
  check_t "stack axis 1 values" [| 2; 2; 3 |]
    [| 1.; 2.; 3.; 7.; 8.; 9.; 4.; 5.; 6.; 10.; 11.; 12. |]
    s

let test_stack_shape_mismatch () =
  let t1 = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let t2 = Nx.create Nx.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  check_invalid_arg "stack shape mismatch"
    "concatenate: invalid dimension 1 (size 4\226\137\1603)" (fun () ->
      Nx.stack ~axis:0 [ t1; t2 ])

let test_stack_new_array () =
  let t1 = Nx.create Nx.float32 [| 2 |] [| 1.0; 2.0 |] in
  let t2 = Nx.create Nx.float32 [| 2 |] [| 3.0; 4.0 |] in
  let s = Nx.stack ~axis:0 [ t1; t2 ] in
  Nx.set_item [ 0 ] 99.0 t1;
  check (float 1e-6) "stack is new array" 1.0 (Nx.item [ 0; 0 ] s)

(* ───── Split Tests ───── *)

let test_split_equal () =
  let t = Nx.create Nx.float32 [| 12 |] (Array.init 12 float_of_int) in
  let parts = Nx.split ~axis:0 3 t in
  check int "split count" 3 (List.length parts);
  check_t "split part 0" [| 4 |] [| 0.; 1.; 2.; 3. |] (List.nth parts 0);
  check_t "split part 1" [| 4 |] [| 4.; 5.; 6.; 7. |] (List.nth parts 1);
  check_t "split part 2" [| 4 |] [| 8.; 9.; 10.; 11. |] (List.nth parts 2)

let test_split_unequal () =
  let t = Nx.create Nx.float32 [| 10 |] (Array.init 10 float_of_int) in
  check_invalid_arg "split unequal"
    "split: cannot divide evenly axis 0 (size 10) to 3 sections (10 % 3 = 1)\n\
     hint: use array_split for uneven division" (fun () ->
      ignore (Nx.split ~axis:0 3 t))

let test_split_axis () =
  let t = Nx.create Nx.float32 [| 4; 6 |] (Array.init 24 float_of_int) in
  let parts = Nx.split ~axis:1 2 t in
  check int "split axis 1 count" 2 (List.length parts);
  check_shape "split axis 1 shape" [| 4; 3 |] (List.nth parts 0)

let test_split_one () =
  let t = Nx.create Nx.float32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let parts = Nx.split ~axis:0 1 t in
  check int "split into 1 count" 1 (List.length parts);
  check_t "split into 1 part" [| 6 |]
    [| 1.; 2.; 3.; 4.; 5.; 6. |]
    (List.nth parts 0)

let test_split_views () =
  let t = Nx.create Nx.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let parts = Nx.split ~axis:0 2 t in
  let p1 = List.nth parts 0 in
  Nx.set_item [ 0 ] 99.0 p1;
  check (float 1e-6) "split view modified" 99.0 (Nx.item [ 0 ] t)

(* ───── Array Split Tests ───── *)

let test_array_split_equal () =
  let t = Nx.create Nx.float32 [| 6 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let parts = Nx.array_split ~axis:0 (`Count 3) t in
  check int "array_split equal count" 3 (List.length parts);
  check_t "array_split equal part 0" [| 2 |] [| 1.0; 2.0 |] (List.nth parts 0)

let test_array_split_unequal () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let parts = Nx.array_split ~axis:0 (`Count 3) t in
  check int "array_split unequal count" 3 (List.length parts);
  check_t "array_split unequal part 0" [| 2 |] [| 1.0; 2.0 |] (List.nth parts 0);
  check_t "array_split unequal part 1" [| 2 |] [| 3.0; 4.0 |] (List.nth parts 1);
  check_t "array_split unequal part 2" [| 1 |] [| 5.0 |] (List.nth parts 2)

let test_array_split_views () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let parts = Nx.array_split ~axis:0 (`Count 2) t in
  let p1 = List.nth parts 0 in
  Nx.set_item [ 0 ] 99.0 p1;
  check (float 1e-6) "array_split view modified" 99.0 (Nx.item [ 0 ] t)

(* ───── Squeeze Expand Tests ───── *)

let test_squeeze_all () =
  let t =
    Nx.create Nx.float32 [| 1; 3; 1; 4; 1 |] (Array.init 12 float_of_int)
  in
  let s = Nx.squeeze t in
  check_shape "squeeze all" [| 3; 4 |] s

let test_squeeze_specific () =
  let t = Nx.create Nx.float32 [| 1; 3; 1; 4 |] (Array.init 12 float_of_int) in
  let s = Nx.squeeze ~axes:[ 0; 2 ] t in
  check_shape "squeeze specific axes" [| 3; 4 |] s

let test_squeeze_no_ones () =
  let t = Nx.create Nx.float32 [| 2; 3; 4 |] (Array.init 24 float_of_int) in
  let s = Nx.squeeze t in
  check_shape "squeeze no ones" [| 2; 3; 4 |] s

let test_squeeze_invalid_axis () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  check_invalid_arg "squeeze invalid axis"
    "squeeze: cannot remove dimension axis 1 (size 3) to squeezed (size \
     3\226\137\1601)" (fun () -> ignore (Nx.squeeze ~axes:[ 1 ] t))

let test_expand_dims_various () =
  let t = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in

  (* Add dim at position 0 *)
  let e0 = Nx.expand_dims [ 0 ] t in
  check_shape "expand_dims at 0" [| 1; 3 |] e0;

  (* Add dim at position -1 (end) *)
  let e_end = Nx.expand_dims [ -1 ] t in
  check_shape "expand_dims at -1" [| 3; 1 |] e_end;

  (* Add dim in middle *)
  let t2 = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let e_mid = Nx.expand_dims [ 1 ] t2 in
  check_shape "expand_dims in middle" [| 2; 1; 3 |] e_mid

let test_expand_dims_invalid_axis () =
  let t = Nx.create Nx.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  check_invalid_arg "expand_dims invalid axis"
    "unsqueeze: invalid axis 2 (out of bounds for output rank 2)\n\
     hint: valid range is [-2, 2)" (fun () -> Nx.expand_dims [ 2 ] t)

(* ───── Broadcasting Tests ───── *)

let test_broadcast_to_valid () =
  let t = Nx.create Nx.float32 [| 3; 1 |] [| 1.; 2.; 3. |] in
  let b = Nx.broadcast_to [| 3; 4 |] t in
  check_shape "broadcast valid shape" [| 3; 4 |] b;
  check (float 1e-6) "broadcast[0,0]" 1.0 (Nx.item [ 0; 0 ] b);
  check (float 1e-6) "broadcast[0,3]" 1.0 (Nx.item [ 0; 3 ] b);
  check (float 1e-6) "broadcast[2,2]" 3.0 (Nx.item [ 2; 2 ] b)

let test_broadcast_to_invalid () =
  let t = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  check_invalid_arg "broadcast invalid"
    "broadcast_to: cannot broadcast [3] to [4] (dim 0: 3≠4)\n\
     hint: broadcasting requires dimensions to be either equal or 1" (fun () ->
      ignore (Nx.broadcast_to [| 4 |] t))

let test_broadcast_to_same () =
  let t = Nx.create Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let b = Nx.broadcast_to [| 3; 4 |] t in
  check_t "broadcast to same" [| 3; 4 |] (Array.init 12 float_of_int) b

let test_broadcast_scalar () =
  let t = Nx.scalar Nx.float32 5.0 in
  let b = Nx.broadcast_to [| 3; 4; 5 |] t in
  check_shape "broadcast scalar shape" [| 3; 4; 5 |] b;
  check (float 1e-6) "broadcast scalar value" 5.0 (Nx.item [ 2; 3; 4 ] b)

let test_broadcast_arrays_compatible () =
  let t1 = Nx.create Nx.float32 [| 3; 1 |] [| 1.0; 2.0; 3.0 |] in
  let t2 = Nx.create Nx.float32 [| 1; 4 |] [| 10.0; 20.0; 30.0; 40.0 |] in
  let broadcasted = Nx.broadcast_arrays [ t1; t2 ] in
  check int "broadcast_arrays count" 2 (List.length broadcasted);
  let b1 = List.nth broadcasted 0 in
  let b2 = List.nth broadcasted 1 in
  check_shape "broadcast_arrays shape 1" [| 3; 4 |] b1;
  check_shape "broadcast_arrays shape 2" [| 3; 4 |] b2

let test_broadcast_arrays_views () =
  let t1 = Nx.create Nx.float32 [| 3; 1 |] [| 1.0; 2.0; 3.0 |] in
  let t2 = Nx.create Nx.float32 [| 1; 1 |] [| 10.0 |] in
  let broadcasted = Nx.broadcast_arrays [ t1; t2 ] in
  let b1 = List.nth broadcasted 0 in
  Nx.set_item [ 0; 0 ] 99.0 t1;
  check (float 1e-6) "broadcast array view modified" 99.0 (Nx.item [ 0; 0 ] b1)

let test_broadcast_arrays_invalid () =
  let t1 = Nx.create Nx.float32 [| 2 |] [| 1.0; 2.0 |] in
  let t2 = Nx.create Nx.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  check_invalid_arg "broadcast_arrays invalid"
    "broadcast: cannot broadcast [2] to [3] (dim 0: 2\226\137\1603)\n\
     hint: broadcasting requires dimensions to be either equal or 1" (fun () ->
      Nx.broadcast_arrays [ t1; t2 ])

(* ───── Tile Repeat Tests ───── *)

let test_tile_1d () =
  let t = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let tiled = Nx.tile [| 2 |] t in
  check_t "tile 1d" [| 6 |] [| 1.; 2.; 3.; 1.; 2.; 3. |] tiled

let test_tile_2d () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let tiled = Nx.tile [| 2; 1 |] t in
  check_t "tile 2d" [| 4; 3 |]
    [| 1.; 2.; 3.; 4.; 5.; 6.; 1.; 2.; 3.; 4.; 5.; 6. |]
    tiled

let test_tile_broadcast () =
  let t = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let tiled = Nx.tile [| 2; 3 |] t in
  check_shape "tile broadcast shape" [| 2; 9 |] tiled;
  check_t "tile broadcast" [| 2; 9 |]
    [| 1.; 2.; 3.; 1.; 2.; 3.; 1.; 2.; 3.; 1.; 2.; 3.; 1.; 2.; 3.; 1.; 2.; 3. |]
    tiled

let test_tile_invalid () =
  let t = Nx.create Nx.float32 [| 2 |] [| 1.0; 2.0 |] in
  (* This test is incorrect - tile should work with more reps than tensor dims
     by promoting the tensor. Let's test a different invalid case. *)
  check_invalid_arg "tile invalid"
    "tile: invalid reps[0] (negative (-1<0))\n\
     hint: use positive integers (or 0 for empty result)" (fun () ->
      Nx.tile [| -1 |] t)

let test_repeat_axis () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let r = Nx.repeat ~axis:0 2 t in
  check_t "repeat axis 0" [| 4; 3 |]
    [| 1.; 2.; 3.; 1.; 2.; 3.; 4.; 5.; 6.; 4.; 5.; 6. |]
    r

let test_repeat_no_axis () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let r = Nx.repeat 2 t in
  check_t "repeat no axis" [| 12 |]
    [| 1.; 1.; 2.; 2.; 3.; 3.; 4.; 4.; 5.; 5.; 6.; 6. |]
    r

let test_repeat_invalid () =
  let t = Nx.create Nx.float32 [| 2 |] [| 1.0; 2.0 |] in
  check_invalid_arg "repeat negative" "repeat: invalid count (count=-1 < 0)"
    (fun () -> Nx.repeat ~axis:0 (-1) t)

(* ───── Other Shape Manipulation Tests ───── *)

let test_flatten_view () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let flat = Nx.flatten t in
  Nx.set_item [ 0; 0 ] 99.0 t;
  check (float 1e-6) "flatten view modified" 99.0 (Nx.item [ 0 ] flat)

let test_ravel_contiguous_view () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let r = Nx.ravel t in
  Nx.set_item [ 0; 0 ] 99.0 t;
  check (float 1e-6) "ravel view modified" 99.0 (Nx.item [ 0 ] r)

let test_ravel_non_contiguous_copy () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let tr = Nx.transpose t in
  check_invalid_arg "ravel non-contiguous"
    "reshape: cannot reshape strided view [2,2] to [4] (incompatible strides \
     [1,2] (expected [1]))\n\
     hint: call contiguous() before reshape to create a C-contiguous copy"
    (fun () -> Nx.ravel tr)

let test_pad_2d () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let p = Nx.pad [| (1, 1); (0, 1) |] 0.0 t in
  check_t "pad 2d" [| 4; 3 |]
    [| 0.0; 0.0; 0.0; 1.0; 2.0; 0.0; 3.0; 4.0; 0.0; 0.0; 0.0; 0.0 |]
    p

let test_pad_invalid () =
  let t = Nx.create Nx.float32 [| 2 |] [| 1.0; 2.0 |] in
  check_invalid_arg "pad negative"
    "pad: invalid padding values (negative values not allowed)\n\
     hint: use shrink or slice to remove elements" (fun () ->
      Nx.pad [| (-1, 2) |] 0.0 t)

let test_flip_axis () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let f = Nx.flip ~axes:[ 1 ] t in
  check_t "flip axis 1" [| 2; 3 |] [| 3.0; 2.0; 1.0; 6.0; 5.0; 4.0 |] f

let test_flip_view () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let f = Nx.flip t in
  Nx.set_item [ 0; 0 ] 99.0 t;
  check (float 1e-6) "flip view modified" 99.0 (Nx.item [ 1; 1 ] f)

let test_roll_no_axis () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let r = Nx.roll 1 t in
  check_t "roll no axis" [| 2; 2 |] [| 4.0; 1.0; 2.0; 3.0 |] r

let test_roll_negative () =
  let t = Nx.create Nx.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let r = Nx.roll ~axis:0 (-1) t in
  check_t "roll negative" [| 3 |] [| 2.0; 3.0; 1.0 |] r

let test_moveaxis_view () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let m = Nx.moveaxis 0 1 t in
  Nx.set_item [ 0; 0 ] 99.0 t;
  check (float 1e-6) "moveaxis view modified" 99.0 (Nx.item [ 0; 0 ] m)

let test_moveaxis_invalid () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  check_invalid_arg "moveaxis invalid"
    "moveaxis: invalid source 2 or destination 0 (out of bounds for shape \
     [2,2])" (fun () -> Nx.moveaxis 2 0 t)

let test_swapaxes_view () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let s = Nx.swapaxes 0 1 t in
  Nx.set_item [ 0; 1 ] 99.0 t;
  check (float 1e-6) "swapaxes view modified" 99.0 (Nx.item [ 1; 0 ] s)

let test_swapaxes_invalid () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  check_invalid_arg "swapaxes invalid"
    "swapaxes: invalid axes (2, 0) (out of bounds for shape [2,2])" (fun () ->
      Nx.swapaxes 2 0 t)

let test_dstack_1d () =
  let t1 = Nx.create Nx.float32 [| 2 |] [| 1.0; 2.0 |] in
  let t2 = Nx.create Nx.float32 [| 2 |] [| 3.0; 4.0 |] in
  let d = Nx.dstack [ t1; t2 ] in
  check_t "dstack 1d" [| 1; 2; 2 |] [| 1.0; 3.0; 2.0; 4.0 |] d

let test_vstack_invalid () =
  let t1 = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t2 = Nx.create Nx.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  check_invalid_arg "vstack invalid"
    "concatenate: invalid dimension 1 (size 3\226\137\1602)" (fun () ->
      Nx.vstack [ t1; t2 ])

let test_hstack_invalid () =
  let t1 = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t2 = Nx.create Nx.float32 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  check_invalid_arg "hstack invalid"
    "concatenate: invalid dimension 0 (size 3\226\137\1602)" (fun () ->
      Nx.hstack [ t1; t2 ])

let test_dstack_invalid () =
  let t1 = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let t2 = Nx.create Nx.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  check_invalid_arg "dstack invalid"
    "concatenate: invalid dimension 1 (size 3\226\137\1602)" (fun () ->
      Nx.dstack [ t1; t2 ])

(* ───── as_strided Tests ───── *)

let test_as_strided_basic () =
  (* Create a simple 1D array and view it as 2D with overlapping windows *)
  let x = Nx.create Nx.float32 [| 8 |] [| 0.; 1.; 2.; 3.; 4.; 5.; 6.; 7. |] in
  (* Create overlapping windows of size 3 with stride 2 *)
  let result = Nx.as_strided [| 3; 3 |] [| 2; 1 |] ~offset:0 x in
  check_t "as_strided overlapping windows" [| 3; 3 |]
    [| 0.; 1.; 2.; 2.; 3.; 4.; 4.; 5.; 6. |]
    result

let test_as_strided_2d_transpose () =
  (* Use as_strided to implement a transpose *)
  let x = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  (* Transpose by swapping strides: original strides are [3, 1], transposed are
     [1, 3] *)
  let result = Nx.as_strided [| 3; 2 |] [| 1; 3 |] ~offset:0 x in
  check_t "as_strided transpose" [| 3; 2 |] [| 1.; 4.; 2.; 5.; 3.; 6. |] result

let test_as_strided_diagonal () =
  (* Extract diagonal using as_strided *)
  let x =
    Nx.create Nx.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
  in
  (* Diagonal has stride of 4 (3+1) to skip to next diagonal element *)
  let result = Nx.as_strided [| 3 |] [| 4 |] ~offset:0 x in
  check_t "as_strided diagonal" [| 3 |] [| 1.; 5.; 9. |] result

let test_as_strided_sliding_window () =
  (* Create sliding windows over a 1D array *)
  let x = Nx.create Nx.float32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  (* Windows of size 3 with stride 1 *)
  let result = Nx.as_strided [| 4; 3 |] [| 1; 1 |] ~offset:0 x in
  check_t "as_strided sliding window" [| 4; 3 |]
    [| 1.; 2.; 3.; 2.; 3.; 4.; 3.; 4.; 5.; 4.; 5.; 6. |]
    result

let test_as_strided_with_offset () =
  (* Test as_strided with non-zero offset *)
  let x =
    Nx.create Nx.float32 [| 10 |] [| 0.; 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
  in
  (* Start from offset 2, create a 2x3 view *)
  let result = Nx.as_strided [| 2; 3 |] [| 3; 1 |] ~offset:2 x in
  check_t "as_strided with offset" [| 2; 3 |]
    [| 2.; 3.; 4.; 5.; 6.; 7. |]
    result

let test_as_strided_scalar_broadcast () =
  (* Use as_strided to broadcast a scalar *)
  let x = Nx.create Nx.float32 [| 1 |] [| 42. |] in
  (* Broadcast to 3x3 using zero strides *)
  let result = Nx.as_strided [| 3; 3 |] [| 0; 0 |] ~offset:0 x in
  check_t "as_strided scalar broadcast" [| 3; 3 |]
    [| 42.; 42.; 42.; 42.; 42.; 42.; 42.; 42.; 42. |]
    result

let test_as_strided_skip_elements () =
  (* Test skipping elements using strides *)
  let x = Nx.create Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  (* Skip every other element *)
  let result = Nx.as_strided [| 3 |] [| 2 |] ~offset:0 x in
  check_t "as_strided skip elements" [| 3 |] [| 1.; 3.; 5. |] result

let test_as_strided_int_dtype () =
  (* Test as_strided with integer dtype *)
  let x = Nx.create Nx.int32 [| 6 |] [| 10l; 20l; 30l; 40l; 50l; 60l |] in
  let result = Nx.as_strided [| 2; 2 |] [| 2; 1 |] ~offset:1 x in
  check_t "as_strided int32" [| 2; 2 |] [| 20l; 30l; 40l; 50l |] result

(* Test Suite Organization *)

let reshape_tests =
  [
    ("reshape minus one", `Quick, test_reshape_minus_one);
    ("reshape multiple minus one", `Quick, test_reshape_multiple_minus_one);
    ("reshape wrong size", `Quick, test_reshape_wrong_size);
    ("reshape 0d to 1d", `Quick, test_reshape_0d_to_1d);
    ("reshape to 0d", `Quick, test_reshape_to_0d);
    ("reshape empty", `Quick, test_reshape_empty);
    ("reshape view when contiguous", `Quick, test_reshape_view_when_contiguous);
    ( "reshape copy when not contiguous",
      `Quick,
      test_reshape_copy_when_not_contiguous );
    ("reshape to vector", `Quick, test_reshape_to_vector);
  ]

let transpose_tests =
  [
    ("transpose 1d", `Quick, test_transpose_1d);
    ("transpose 0d", `Quick, test_transpose_0d);
    ("transpose high d", `Quick, test_transpose_high_d);
    ("transpose axes", `Quick, test_transpose_axes);
    ("transpose invalid axes", `Quick, test_transpose_invalid_axes);
    ("transpose view", `Quick, test_transpose_view);
  ]

let concatenate_tests =
  [
    ("concat axis 1", `Quick, test_concat_axis_1);
    ("concat single", `Quick, test_concat_single);
    ("concat empty list", `Quick, test_concat_empty_list);
    ("concat different dtypes", `Quick, test_concat_different_dtypes);
    ("concat with empty", `Quick, test_concat_with_empty);
    ("concat shape mismatch", `Quick, test_concat_shape_mismatch);
    ("concat new array", `Quick, test_concat_new_array);
  ]

let stack_tests =
  [
    ("stack new axis", `Quick, test_stack_new_axis);
    ("stack shape mismatch", `Quick, test_stack_shape_mismatch);
    ("stack new array", `Quick, test_stack_new_array);
  ]

let split_tests =
  [
    ("split equal", `Quick, test_split_equal);
    ("split unequal", `Quick, test_split_unequal);
    ("split axis", `Quick, test_split_axis);
    ("split one", `Quick, test_split_one);
    ("split views", `Quick, test_split_views);
    ("array split equal", `Quick, test_array_split_equal);
    ("array split unequal", `Quick, test_array_split_unequal);
    ("array split views", `Quick, test_array_split_views);
  ]

let squeeze_expand_tests =
  [
    ("squeeze all", `Quick, test_squeeze_all);
    ("squeeze specific", `Quick, test_squeeze_specific);
    ("squeeze no ones", `Quick, test_squeeze_no_ones);
    ("squeeze invalid axis", `Quick, test_squeeze_invalid_axis);
    ("expand dims various", `Quick, test_expand_dims_various);
    ("expand dims invalid axis", `Quick, test_expand_dims_invalid_axis);
  ]

let broadcast_tests =
  [
    ("broadcast to valid", `Quick, test_broadcast_to_valid);
    ("broadcast to invalid", `Quick, test_broadcast_to_invalid);
    ("broadcast to same", `Quick, test_broadcast_to_same);
    ("broadcast scalar", `Quick, test_broadcast_scalar);
    ("broadcast arrays compatible", `Quick, test_broadcast_arrays_compatible);
    ("broadcast arrays views", `Quick, test_broadcast_arrays_views);
    ("broadcast arrays invalid", `Quick, test_broadcast_arrays_invalid);
  ]

let tile_repeat_tests =
  [
    ("tile 1d", `Quick, test_tile_1d);
    ("tile 2d", `Quick, test_tile_2d);
    ("tile broadcast", `Quick, test_tile_broadcast);
    ("tile invalid", `Quick, test_tile_invalid);
    ("repeat axis", `Quick, test_repeat_axis);
    ("repeat no axis", `Quick, test_repeat_no_axis);
    ("repeat invalid", `Quick, test_repeat_invalid);
  ]

let other_manipulation_tests =
  [
    ("flatten view", `Quick, test_flatten_view);
    ("ravel contiguous view", `Quick, test_ravel_contiguous_view);
    ("ravel non-contiguous copy", `Quick, test_ravel_non_contiguous_copy);
    ("pad 2d", `Quick, test_pad_2d);
    ("pad invalid", `Quick, test_pad_invalid);
    ("flip axis", `Quick, test_flip_axis);
    ("flip view", `Quick, test_flip_view);
    ("roll no axis", `Quick, test_roll_no_axis);
    ("roll negative", `Quick, test_roll_negative);
    ("moveaxis view", `Quick, test_moveaxis_view);
    ("moveaxis invalid", `Quick, test_moveaxis_invalid);
    ("swapaxes view", `Quick, test_swapaxes_view);
    ("swapaxes invalid", `Quick, test_swapaxes_invalid);
    ("dstack 1d", `Quick, test_dstack_1d);
    ("vstack invalid", `Quick, test_vstack_invalid);
    ("hstack invalid", `Quick, test_hstack_invalid);
    ("dstack invalid", `Quick, test_dstack_invalid);
    (* as_strided tests *)
    ("as_strided basic", `Quick, test_as_strided_basic);
    ("as_strided 2d transpose", `Quick, test_as_strided_2d_transpose);
    ("as_strided diagonal", `Quick, test_as_strided_diagonal);
    ("as_strided sliding window", `Quick, test_as_strided_sliding_window);
    ("as_strided with offset", `Quick, test_as_strided_with_offset);
    ("as_strided scalar broadcast", `Quick, test_as_strided_scalar_broadcast);
    ("as_strided skip elements", `Quick, test_as_strided_skip_elements);
    ("as_strided int dtype", `Quick, test_as_strided_int_dtype);
  ]

let suite =
  [
    ("Manipulation :: Reshape", reshape_tests);
    ("Manipulation :: Transpose", transpose_tests);
    ("Manipulation :: Concatenate", concatenate_tests);
    ("Manipulation :: Stack", stack_tests);
    ("Manipulation :: Split", split_tests);
    ("Manipulation :: Squeeze/Expand", squeeze_expand_tests);
    ("Manipulation :: Broadcasting", broadcast_tests);
    ("Manipulation :: Tile/Repeat", tile_repeat_tests);
    ("Manipulation :: Other Manipulation", other_manipulation_tests);
  ]

let () = Alcotest.run "Nx Manipulation" suite
