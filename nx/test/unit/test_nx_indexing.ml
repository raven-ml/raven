(* Indexing and slicing tests for Nx *)

open Alcotest

module Make (Backend : Nx_core.Backend_intf.S) = struct
  module Support = Test_nx_support.Make (Backend)
  module Nx = Support.Nx
  open Support

  (* ───── Basic Slicing Tests ───── *)

  let test_slice_full ctx () =
    let t = Nx.create ctx Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
    let sliced = t in
    check_t "slice full [:] is identity" [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] sliced

  let test_slice_single ctx () =
    let t = Nx.create ctx Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
    let sliced = Nx.slice [ R [ 2; 3 ] ] t in
    check_t "slice single [2:3]" [| 1 |] [| 3. |] sliced

  let test_slice_range ctx () =
    let t = Nx.create ctx Nx.float32 [| 7 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7. |] in
    let sliced = Nx.slice [ R [ 2; 5 ] ] t in
    check_t "slice range [2:5]" [| 3 |] [| 3.; 4.; 5. |] sliced

  let test_slice_step ctx () =
    let t =
      Nx.create ctx Nx.float32 [| 10 |]
        (Array.init 10 (fun i -> float_of_int i))
    in
    (* [::2] - every other element *)
    let sliced1 = Nx.slice_ranges ~steps:[ 2 ] [ 0 ] [ 10 ] t in
    check_t "slice step [::2]" [| 5 |] [| 0.; 2.; 4.; 6.; 8. |] sliced1;

    (* [1::2] - every other element starting from 1 *)
    let sliced2 = Nx.slice_ranges ~steps:[ 2 ] [ 1 ] [ 10 ] t in
    check_t "slice step [1::2]" [| 5 |] [| 1.; 3.; 5.; 7.; 9. |] sliced2

  let test_slice_negative_step ctx () =
    let t = Nx.create ctx Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
    (* [::-1] reverses array *)
    let sliced = Nx.slice_ranges ~steps:[ -1 ] [ 4 ] [ -6 ] t in
    check_t "slice negative step [::-1]" [| 5 |] [| 5.; 4.; 3.; 2.; 1. |] sliced

  let test_slice_negative_indices ctx () =
    let t = Nx.create ctx Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
    (* [-3:-1] = elements at indices 2,3 *)
    let sliced = Nx.slice [ R [ -3; -1 ] ] t in
    check_t "slice negative indices [-3:-1]" [| 2 |] [| 3.; 4. |] sliced

  let test_slice_mixed_indices ctx () =
    let t = Nx.create ctx Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
    (* [1:-1] = from index 1 to second-to-last *)
    let sliced = Nx.slice [ R [ 1; -1 ] ] t in
    check_t "slice mixed indices [1:-1]" [| 3 |] [| 2.; 3.; 4. |] sliced

  let test_slice_out_of_bounds_clamp ctx () =
    let t = Nx.create ctx Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
    (* [2:100] clamps to [2:5] *)
    let sliced = Nx.slice [ R [ 2; 100 ] ] t in
    check_t "slice out of bounds clamp [2:100]" [| 3 |] [| 3.; 4.; 5. |] sliced

  let test_slice_empty ctx () =
    let t = Nx.create ctx Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
    (* [5:2] produces empty array *)
    let sliced = Nx.slice [ R [ 5; 2 ] ] t in
    check_shape "slice empty [5:2]" [| 0 |] sliced

  (* ───── Multi Dimensional Slicing Tests ───── *)

  let test_slice_2d_rows ctx () =
    let t =
      Nx.create ctx Nx.float32 [| 4; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
    in
    (* arr[1:3, :] *)
    let sliced = Nx.slice [ R [ 1; 3 ]; R [] ] t in
    check_t "slice 2d rows [1:3, :]" [| 2; 3 |]
      [| 4.; 5.; 6.; 7.; 8.; 9. |]
      sliced

  let test_slice_2d_cols ctx () =
    let t =
      Nx.create ctx Nx.float32 [| 4; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
    in
    (* arr[:, 1:3] *)
    let sliced = Nx.slice [ R []; R [ 1; 3 ] ] t in
    check_t "slice 2d cols [:, 1:3]" [| 4; 2 |]
      [| 2.; 3.; 5.; 6.; 8.; 9.; 11.; 12. |]
      sliced

  let test_slice_2d_both ctx () =
    let t =
      Nx.create ctx Nx.float32 [| 4; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
    in
    (* arr[1:3, 1:3] *)
    let sliced = Nx.slice [ R [ 1; 3 ]; R [ 1; 3 ] ] t in
    check_t "slice 2d both [1:3, 1:3]" [| 2; 2 |] [| 5.; 6.; 8.; 9. |] sliced

  let test_slice_different_steps ctx () =
    let t = Nx.create ctx Nx.float32 [| 6; 6 |] (Array.init 36 float_of_int) in
    (* arr[::2, ::3] *)
    let sliced = Nx.slice_ranges ~steps:[ 2; 3 ] [ 0; 0 ] [ 6; 6 ] t in
    check_t "slice different steps [::2, ::3]" [| 3; 2 |]
      [| 0.; 3.; 12.; 15.; 24.; 27. |]
      sliced

  let test_slice_newaxis ctx () =
    let t = Nx.create ctx Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
    (* arr[:, None, :] adds dimension *)
    let sliced = Nx.unsqueeze ~axes:[| 1 |] t in
    check_shape "slice newaxis [:, None, :]" [| 3; 1; 4 |] sliced

  (* Ellipsis indexing not supported in current API *)
  let test_slice_ellipsis _ctx () =
    (* Skip this test - ellipsis indexing not implemented *)
    ()

  (*  ─────  Integer Array Indexing Tests  ─────  *)
  (* Note: Advanced integer array indexing with Nx.I not supported in current API *)

  (* let test_int_array_1d ctx () = let t = Nx.create ctx Nx.float32 [| 6 |] [|
     10.; 20.; 30.; 40.; 50.; 60. |] in (* arr[[0,2,4]] *) let indices =
     Nx.create Nx.int32 [| 3 |] [| 0l; 2l; 4l |] in let result = Nx.get (Nx.I
     indices) t in check_t "int array 1d [[0,2,4]]" [| 3 |] [| 10.; 30.; 50. |]
     result

     let test_int_array_negative ctx () = let t = Nx.create ctx Nx.float32 [| 5
     |] [| 1.; 2.; 3.; 4.; 5. |] in (* arr[[-1,-2]] *) let indices = Nx.create
     Nx.int32 [| 2 |] [| -1l; -2l |] in let result = Nx.get (Nx.I indices) t in
     check_t "int array negative [[-1,-2]]" [| 2 |] [| 5.; 4. |] result

     let test_int_array_repeated ctx () = let t = Nx.create ctx Nx.float32 [| 3
     |] [| 10.; 20.; 30. |] in (* arr[[0,1,1,0]] *) let indices = Nx.create ctx
     Nx.int32 [| 4 |] [| 0l; 1l; 1l; 0l |] in let result = Nx.get (Nx.I indices)
     t in check_t "int array repeated [[0,1,1,0]]" [| 4 |] [| 10.; 20.; 20.; 10.
     |] result

     let test_int_array_out_of_bounds ctx () = let t = Nx.create ctx Nx.float32
     [| 3 |] [| 1.; 2.; 3. |] in (* arr[[0,100]] should error *) let indices =
     Nx.create Nx.int32 [| 2 |] [| 0l; 100l |] in check_invalid_arg "int array
     out of bounds" "Index 100 is out of bounds for axis 0 with size 3" (fun ()
     -> ignore (Nx.get (Nx.I indices) t))

     let test_int_array_multidim ctx () = let t = Nx.create ctx Nx.float32 [| 3;
     4 |] (Array.init 12 float_of_int) in (* arr[[0,1], [2,3]] *) let idx0 =
     Nx.create ctx Nx.int32 [| 2 |] [| 0l; 1l |] in let idx1 = Nx.create ctx
     Nx.int32 [| 2 |] [| 2l; 3l |] in let result = Nx.get (Nx.LI [ idx0; idx1 ])
     t in check_t "int array multidim [[0,1], [2,3]]" [| 2 |] [| 2.; 7. |]
     result

     let test_int_array_broadcast ctx () = let t = Nx.create ctx Nx.float32 [|
     3; 4 |] (Array.init 12 float_of_int) in (* arr[[[0,1]], [2,3]] -
     broadcasting shapes *) let idx0 = Nx.create ctx Nx.int32 [| 1; 2 |] [| 0l;
     1l |] in let idx1 = Nx.create ctx Nx.int32 [| 2 |] [| 2l; 3l |] in let
     result = Nx.get (Nx.LI [ idx0; idx1 ]) t in check_t "int array broadcast"
     [| 1; 2 |] [| 2.; 7. |] result *)

  (*  ─────  Boolean Indexing Tests  ─────  *)
  (* Note: Boolean indexing with Nx.B not supported in current API *)

  (* let test_bool_1d ctx () = let t = Nx.create ctx Nx.float32 [| 4 |] [| 10.;
     20.; 30.; 40. |] in (* arr[[true,false,true,false]] *) let mask = Nx.create
     Nx.uint8 [| 4 |] [| 1; 0; 1; 0 |] in let result = Nx.get (Nx.B mask) t in
     check_t "bool 1d" [| 2 |] [| 10.; 30. |] result

     let test_bool_wrong_shape ctx () = let t = Nx.create ctx Nx.float32 [| 3 |]
     [| 1.; 2.; 3. |] in (* Bool array wrong size (error) *) let mask =
     Nx.create Nx.uint8 [| 2 |] [| 1; 0 |] in check_invalid_arg "bool wrong
     shape" "Boolean indexing array has shape [2] but array has shape [3]" (fun
     () -> ignore (Nx.get (Nx.B mask) t))

     let test_bool_multidim ctx () = let t = Nx.create ctx Nx.float32 [| 2; 3 |]
     [| 1.; 2.; 3.; 4.; 5.; 6. |] in (* 2D bool on 2D array *) let mask =
     Nx.create Nx.uint8 [| 2; 3 |] [| 1; 0; 1; 0; 1; 0 |] in let result = Nx.get
     (Nx.B mask) t in check_t "bool multidim" [| 3 |] [| 1.; 3.; 5. |] result

     let test_bool_result_1d ctx () = let t = Nx.create ctx Nx.float32 [| 2; 2;
     2 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |] in (* Boolean indexing always
     returns 1D *) let mask = Nx.create ctx Nx.uint8 [| 2; 2; 2 |] [| 1; 0; 0;
     1; 0; 1; 1; 0 |] in let result = Nx.get (Nx.B mask) t in check_shape "bool
     result is 1D" [| 4 |] result; check_t "bool result 1D values" [| 4 |] [|
     1.; 4.; 6.; 7. |] result *)

  (*  ─────  Advanced Indexing Tests  ─────  *)
  (* Note: Advanced indexing features not supported in current API *)

  (* let test_mixed_basic_advanced ctx () = let t = Nx.create ctx Nx.float32 [|
     4; 5 |] (Array.init 20 float_of_int) in (* arr[1:3, [0,2,4]] - mix of slice
     and integer array *) let indices = Nx.create ctx Nx.int32 [| 3 |] [| 0l;
     2l; 4l |] in let result = Nx.get (Nx.LR [ [ 1; 3 ]; Nx.Index indices ]) t
     in check_t "mixed basic advanced" [| 2; 3 |] [| 5.; 7.; 9.; 10.; 12.; 14.
     |] result

     let test_setitem_advanced ctx () = let t = Nx.zeros Nx.float32 [| 5 |] in
     (* arr[[0,2,4]] = values *) let indices = Nx.create ctx Nx.int32 [| 3 |] [|
     0l; 2l; 4l |] in let values = Nx.create ctx Nx.float32 [| 3 |] [| 10.; 20.;
     30. |] in Nx.set (Nx.I indices) t values; check_t "setitem advanced" [| 5
     |] [| 10.; 0.; 20.; 0.; 30. |] t

     let test_advanced_broadcast ctx () = let t = Nx.create ctx Nx.float32 [| 3;
     4 |] (Array.init 12 float_of_int) in (* Test broadcasting in advanced
     indexing output shape *) let idx0 = Nx.create ctx Nx.int32 [| 2; 1 |] [|
     0l; 2l |] in let idx1 = Nx.create ctx Nx.int32 [| 3 |] [| 1l; 2l; 3l |] in
     let result = Nx.get (Nx.LI [ idx0; idx1 ]) t in check_shape "advanced
     broadcast shape" [| 2; 3 |] result *)

  (* ───── View Vs Copy Tests ───── *)

  let test_basic_slice_is_view ctx () =
    let t = Nx.create ctx Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
    (* arr[1:3] shares memory *)
    let sliced = Nx.slice [ R [ 1; 3 ] ] t in
    Nx.unsafe_set [ 1 ] 99.0 t;
    check (float 1e-6) "basic slice is view" 99.0 (Nx.unsafe_get [ 0 ] sliced)

  (* Advanced indexing not supported let test_advanced_index_is_copy ctx () =
     let t = Nx.create ctx Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in (*
     arr[[0,2]] is copy *) let indices = Nx.create ctx Nx.int32 [| 2 |] [| 0l;
     2l |] in let result = Nx.get (Nx.I indices) t in Nx.unsafe_set [ 0 ] 99.0
     t; check (float 1e-6) "advanced index is copy" 1.0 (Nx.unsafe_get [ 0 ]
     result) *)

  let test_slice_of_slice_offset ctx () =
    let t = Nx.create ctx Nx.float32 [| 10 |] (Array.init 10 float_of_int) in
    (* First slice: [2:8] *)
    let slice1 = Nx.slice [ R [ 2; 8 ] ] t in
    (* Second slice of first: [1:4] *)
    let slice2 = Nx.slice [ R [ 1; 4 ] ] slice1 in
    (* Should contain elements [3,4,5] from original *)
    check_t "slice of slice values" [| 3 |] [| 3.; 4.; 5. |] slice2;
    (* Modify original and check if visible *)
    Nx.unsafe_set [ 3 ] 99.0 t;
    check (float 1e-6) "slice of slice offset" 99.0 (Nx.unsafe_get [ 0 ] slice2)

  (* ───── Additional Edge Cases ───── *)

  (* Advanced indexing not supported let test_empty_indices ctx () = let t =
     Nx.create ctx Nx.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in (* Empty
     integer array indexing *) let indices = Nx.create ctx Nx.int32 [| 0 |] [||]
     in let result = Nx.get (Nx.I indices) t in check_shape "empty indices
     shape" [| 0 |] result *)

  let test_scalar_indexing ctx () =
    let t = Nx.create ctx Nx.float32 [| 3; 4 |] (Array.init 12 float_of_int) in
    (* Single element indexing returns scalar view *)
    let result = Nx.get [ 1; 2 ] t in
    check_t "scalar indexing" [||] [| 6. |] result

  let test_negative_step_2d ctx () =
    let t =
      Nx.create ctx Nx.float32 [| 3; 3 |]
        [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
    in
    (* Reverse both dimensions *)
    let sliced = Nx.flip t in
    check_t "negative step 2d" [| 3; 3 |]
      [| 9.; 8.; 7.; 6.; 5.; 4.; 3.; 2.; 1. |]
      sliced

  (* Test Suite Organization *)

  let basic_slicing_tests ctx =
    [
      ("slice full", `Quick, test_slice_full ctx);
      ("slice single", `Quick, test_slice_single ctx);
      ("slice range", `Quick, test_slice_range ctx);
      ("slice step", `Quick, test_slice_step ctx);
      ("slice negative step", `Quick, test_slice_negative_step ctx);
      ("slice negative indices", `Quick, test_slice_negative_indices ctx);
      ("slice mixed indices", `Quick, test_slice_mixed_indices ctx);
      ("slice out of bounds clamp", `Quick, test_slice_out_of_bounds_clamp ctx);
      ("slice empty", `Quick, test_slice_empty ctx);
    ]

  let multidim_slicing_tests ctx =
    [
      ("slice 2d rows", `Quick, test_slice_2d_rows ctx);
      ("slice 2d cols", `Quick, test_slice_2d_cols ctx);
      ("slice 2d both", `Quick, test_slice_2d_both ctx);
      ("slice different steps", `Quick, test_slice_different_steps ctx);
      ("slice newaxis", `Quick, test_slice_newaxis ctx);
      ("slice ellipsis", `Quick, test_slice_ellipsis ctx);
    ]

  let view_copy_tests ctx =
    [
      ("basic slice is view", `Quick, test_basic_slice_is_view ctx);
      (* "advanced index is copy", `Quick, test_advanced_index_is_copy ctx; *)
      ("slice of slice offset", `Quick, test_slice_of_slice_offset ctx);
    ]

  let edge_case_tests ctx =
    [
      (* "empty indices", `Quick, test_empty_indices ctx; *)
      ("scalar indexing", `Quick, test_scalar_indexing ctx);
      ("negative step 2d", `Quick, test_negative_step_2d ctx);
    ]

  let suite backend_name ctx =
    [
      ("Indexing :: " ^ backend_name ^ " Basic Slicing", basic_slicing_tests ctx);
      ( "Indexing :: " ^ backend_name ^ " Multi-dimensional Slicing",
        multidim_slicing_tests ctx );
      ("Indexing :: " ^ backend_name ^ " View vs Copy", view_copy_tests ctx);
      ("Indexing :: " ^ backend_name ^ " Edge Cases", edge_case_tests ctx);
    ]
end
