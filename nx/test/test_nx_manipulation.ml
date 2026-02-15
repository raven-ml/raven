(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Shape manipulation tests for Nx *)

open Windtrap
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
  equal ~msg:"reshape view sees source mutations" (float 1e-6) 77.0
    (Nx.item [ 0; 0 ] r);
  Nx.set_item [ 0; 0 ] 42.0 r;
  equal ~msg:"reshape view mutates source" (float 1e-6) 42.0 (Nx.item [ 0 ] t)

let test_reshape_copy_when_not_contiguous () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let transposed = Nx.transpose t in
  check_invalid_arg "reshape non-contiguous"
    "reshape: cannot reshape strided view [3,2] to [6] (incompatible strides \
     [1,3] (expected [1]))\n\
     hint: call contiguous() before reshape to create a C-contiguous copy"
    (fun () -> Nx.reshape [| 6 |] transposed)

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
  equal ~msg:"transpose[0,0,0,0]" (float 1e-6) 0.0 (Nx.item [ 0; 0; 0; 0 ] tr);
  equal ~msg:"transpose[0,0,0,1]" (float 1e-6) 60.0 (Nx.item [ 0; 0; 0; 1 ] tr)

let test_transpose_axes () =
  let t = Nx.create Nx.float32 [| 2; 3; 4 |] (Array.init 24 float_of_int) in
  let tr = Nx.transpose ~axes:[ 1; 2; 0 ] t in
  check_shape "transpose custom axes" [| 3; 4; 2 |] tr;
  equal ~msg:"transpose[0,0,0]" (float 1e-6) 0.0 (Nx.item [ 0; 0; 0 ] tr);
  equal ~msg:"transpose[0,0,1]" (float 1e-6) 12.0 (Nx.item [ 0; 0; 1 ] tr)

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
  equal ~msg:"transpose view modified" (float 1e-6) 99.0 (Nx.item [ 1; 0 ] tr)

(* ───── Concatenate Tests ───── *)

let test_concat_axis_1 () =
  let t1 = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let t2 = Nx.create Nx.float32 [| 2; 2 |] [| 7.; 8.; 9.; 10. |] in
  let c = Nx.concatenate ~axis:1 [ t1; t2 ] in
  check_t "concat axis 1" [| 2; 5 |]
    [| 1.; 2.; 3.; 7.; 8.; 4.; 5.; 6.; 9.; 10. |]
    c

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
  equal ~msg:"concat is new array" (float 1e-6) 1.0 (Nx.item [ 0 ] c)

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
  equal ~msg:"stack is new array" (float 1e-6) 1.0 (Nx.item [ 0; 0 ] s)

(* ───── Split Tests ───── *)

let test_split_equal () =
  let t = Nx.create Nx.float32 [| 12 |] (Array.init 12 float_of_int) in
  let parts = Nx.split ~axis:0 3 t in
  equal ~msg:"split count" int 3 (List.length parts);
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
  equal ~msg:"split axis 1 count" int 2 (List.length parts);
  check_shape "split axis 1 shape" [| 4; 3 |] (List.nth parts 0)

let test_split_one () =
  let t = Nx.create Nx.float32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let parts = Nx.split ~axis:0 1 t in
  equal ~msg:"split into 1 count" int 1 (List.length parts);
  check_t "split into 1 part" [| 6 |]
    [| 1.; 2.; 3.; 4.; 5.; 6. |]
    (List.nth parts 0)

let test_split_views () =
  let t = Nx.create Nx.float32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let parts = Nx.split ~axis:0 2 t in
  let p1 = List.nth parts 0 in
  Nx.set_item [ 0 ] 99.0 p1;
  equal ~msg:"split view modified" (float 1e-6) 99.0 (Nx.item [ 0 ] t)

(* ───── Array Split Tests ───── *)

let test_array_split_equal () =
  let t = Nx.create Nx.float32 [| 6 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let parts = Nx.array_split ~axis:0 (`Count 3) t in
  equal ~msg:"array_split equal count" int 3 (List.length parts);
  check_t "array_split equal part 0" [| 2 |] [| 1.0; 2.0 |] (List.nth parts 0)

let test_array_split_unequal () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let parts = Nx.array_split ~axis:0 (`Count 3) t in
  equal ~msg:"array_split unequal count" int 3 (List.length parts);
  check_t "array_split unequal part 0" [| 2 |] [| 1.0; 2.0 |] (List.nth parts 0);
  check_t "array_split unequal part 1" [| 2 |] [| 3.0; 4.0 |] (List.nth parts 1);
  check_t "array_split unequal part 2" [| 1 |] [| 5.0 |] (List.nth parts 2)

let test_array_split_views () =
  let t = Nx.create Nx.float32 [| 5 |] [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let parts = Nx.array_split ~axis:0 (`Count 2) t in
  let p1 = List.nth parts 0 in
  Nx.set_item [ 0 ] 99.0 p1;
  equal ~msg:"array_split view modified" (float 1e-6) 99.0 (Nx.item [ 0 ] t)

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
  equal ~msg:"broadcast[0,0]" (float 1e-6) 1.0 (Nx.item [ 0; 0 ] b);
  equal ~msg:"broadcast[0,3]" (float 1e-6) 1.0 (Nx.item [ 0; 3 ] b);
  equal ~msg:"broadcast[2,2]" (float 1e-6) 3.0 (Nx.item [ 2; 2 ] b)

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
  equal ~msg:"broadcast scalar value" (float 1e-6) 5.0 (Nx.item [ 2; 3; 4 ] b)

let test_broadcast_arrays_compatible () =
  let t1 = Nx.create Nx.float32 [| 3; 1 |] [| 1.0; 2.0; 3.0 |] in
  let t2 = Nx.create Nx.float32 [| 1; 4 |] [| 10.0; 20.0; 30.0; 40.0 |] in
  let broadcasted = Nx.broadcast_arrays [ t1; t2 ] in
  equal ~msg:"broadcast_arrays count" int 2 (List.length broadcasted);
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
  equal ~msg:"broadcast array view modified" (float 1e-6) 99.0
    (Nx.item [ 0; 0 ] b1)

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
  equal ~msg:"flatten view modified" (float 1e-6) 99.0 (Nx.item [ 0 ] flat)

let test_ravel_contiguous_view () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let r = Nx.ravel t in
  Nx.set_item [ 0; 0 ] 99.0 t;
  equal ~msg:"ravel view modified" (float 1e-6) 99.0 (Nx.item [ 0 ] r)

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
  equal ~msg:"flip view modified" (float 1e-6) 99.0 (Nx.item [ 1; 1 ] f)

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
  equal ~msg:"moveaxis view modified" (float 1e-6) 99.0 (Nx.item [ 0; 0 ] m)

let test_moveaxis_invalid () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  check_invalid_arg "moveaxis invalid"
    "moveaxis: invalid source 2 or destination 0 (out of bounds for shape \
     [2,2])" (fun () -> Nx.moveaxis 2 0 t)

let test_swapaxes_view () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let s = Nx.swapaxes 0 1 t in
  Nx.set_item [ 0; 1 ] 99.0 t;
  equal ~msg:"swapaxes view modified" (float 1e-6) 99.0 (Nx.item [ 1; 0 ] s)

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

(* Test Suite Organization *)

let reshape_tests =
  [
    test "reshape minus one" test_reshape_minus_one;
    test "reshape multiple minus one" test_reshape_multiple_minus_one;
    test "reshape wrong size" test_reshape_wrong_size;
    test "reshape 0d to 1d" test_reshape_0d_to_1d;
    test "reshape to 0d" test_reshape_to_0d;
    test "reshape empty" test_reshape_empty;
    test "reshape view when contiguous" test_reshape_view_when_contiguous;
    test "reshape copy when not contiguous"
      test_reshape_copy_when_not_contiguous;
  ]

let transpose_tests =
  [
    test "transpose 1d" test_transpose_1d;
    test "transpose 0d" test_transpose_0d;
    test "transpose high d" test_transpose_high_d;
    test "transpose axes" test_transpose_axes;
    test "transpose invalid axes" test_transpose_invalid_axes;
    test "transpose view" test_transpose_view;
  ]

let concatenate_tests =
  [
    test "concat axis 1" test_concat_axis_1;
    test "concat empty list" test_concat_empty_list;
    test "concat different dtypes" test_concat_different_dtypes;
    test "concat with empty" test_concat_with_empty;
    test "concat shape mismatch" test_concat_shape_mismatch;
    test "concat new array" test_concat_new_array;
  ]

let stack_tests =
  [
    test "stack new axis" test_stack_new_axis;
    test "stack shape mismatch" test_stack_shape_mismatch;
    test "stack new array" test_stack_new_array;
  ]

let split_tests =
  [
    test "split equal" test_split_equal;
    test "split unequal" test_split_unequal;
    test "split axis" test_split_axis;
    test "split one" test_split_one;
    test "split views" test_split_views;
    test "array split equal" test_array_split_equal;
    test "array split unequal" test_array_split_unequal;
    test "array split views" test_array_split_views;
  ]

let squeeze_expand_tests =
  [
    test "squeeze all" test_squeeze_all;
    test "squeeze specific" test_squeeze_specific;
    test "squeeze no ones" test_squeeze_no_ones;
    test "squeeze invalid axis" test_squeeze_invalid_axis;
    test "expand dims various" test_expand_dims_various;
    test "expand dims invalid axis" test_expand_dims_invalid_axis;
  ]

let broadcast_tests =
  [
    test "broadcast to valid" test_broadcast_to_valid;
    test "broadcast to invalid" test_broadcast_to_invalid;
    test "broadcast to same" test_broadcast_to_same;
    test "broadcast scalar" test_broadcast_scalar;
    test "broadcast arrays compatible" test_broadcast_arrays_compatible;
    test "broadcast arrays views" test_broadcast_arrays_views;
    test "broadcast arrays invalid" test_broadcast_arrays_invalid;
  ]

let tile_repeat_tests =
  [
    test "tile 1d" test_tile_1d;
    test "tile 2d" test_tile_2d;
    test "tile broadcast" test_tile_broadcast;
    test "tile invalid" test_tile_invalid;
    test "repeat axis" test_repeat_axis;
    test "repeat no axis" test_repeat_no_axis;
    test "repeat invalid" test_repeat_invalid;
  ]

let other_manipulation_tests =
  [
    test "flatten view" test_flatten_view;
    test "ravel contiguous view" test_ravel_contiguous_view;
    test "ravel non-contiguous copy" test_ravel_non_contiguous_copy;
    test "pad 2d" test_pad_2d;
    test "pad invalid" test_pad_invalid;
    test "flip axis" test_flip_axis;
    test "flip view" test_flip_view;
    test "roll no axis" test_roll_no_axis;
    test "roll negative" test_roll_negative;
    test "moveaxis view" test_moveaxis_view;
    test "moveaxis invalid" test_moveaxis_invalid;
    test "swapaxes view" test_swapaxes_view;
    test "swapaxes invalid" test_swapaxes_invalid;
    test "dstack 1d" test_dstack_1d;
    test "vstack invalid" test_vstack_invalid;
    test "hstack invalid" test_hstack_invalid;
    test "dstack invalid" test_dstack_invalid;
  ]

let () =
  run "Nx Manipulation"
    [
      group "Reshape" reshape_tests;
      group "Transpose" transpose_tests;
      group "Concatenate" concatenate_tests;
      group "Stack" stack_tests;
      group "Split" split_tests;
      group "Squeeze/Expand" squeeze_expand_tests;
      group "Broadcasting" broadcast_tests;
      group "Tile/Repeat" tile_repeat_tests;
      group "Other Manipulation" other_manipulation_tests;
    ]
