(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Sorting and searching tests for Nx *)

open Windtrap
open Test_nx_support

(* ───── Where Tests ───── *)

let test_where_1d () =
  let mask = Nx.create Nx.bool [| 3 |] [| true; false; true |] in
  let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Nx.create Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
  let result = Nx.where mask a b in
  check_t "where 1D" [| 3 |] [| 1.; 5.; 3. |] result

let test_where_broadcast () =
  let mask = Nx.create Nx.bool [| 2; 1 |] [| true; false |] in
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = Nx.create Nx.float32 [| 2; 3 |] [| 7.; 8.; 9.; 10.; 11.; 12. |] in
  let result = Nx.where mask a b in
  check_t "where with broadcasting" [| 2; 3 |]
    [| 1.; 2.; 3.; 10.; 11.; 12. |]
    result

let test_where_scalar_inputs () =
  let mask =
    Nx.create Nx.bool [| 2; 3 |] [| true; false; true; false; true; false |]
  in
  let a = Nx.scalar Nx.float32 5.0 in
  let b = Nx.scalar Nx.float32 10.0 in
  let result = Nx.where mask a b in
  check_t "where with scalar inputs" [| 2; 3 |]
    [| 5.0; 10.0; 5.0; 10.0; 5.0; 10.0 |]
    result

let test_where_invalid_shapes () =
  let mask = Nx.create Nx.bool [| 2 |] [| true; false |] in
  let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Nx.create Nx.float32 [| 2 |] [| 4.; 5. |] in
  raises ~msg:"where invalid shapes"
    (Invalid_argument
       "broadcast: cannot broadcast [3] to [2] (dim 0: 3\226\137\1602)\n\
        hint: broadcasting requires dimensions to be either equal or 1")
    (fun () -> ignore (Nx.where mask a b))

(* ───── Sort Tests ───── *)

let test_sort_2d_axis0 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 4.; 1.; 3.; 2.; 5.; 6. |] in
  let result, indices = Nx.sort ~axis:0 t in
  check_t "sort 2D axis 0 values" [| 2; 3 |] [| 2.; 1.; 3.; 4.; 5.; 6. |] result;
  check_t "sort 2D axis 0 indices" [| 2; 3 |]
    [| 1l; 0l; 0l; 0l; 1l; 1l |]
    indices

let test_sort_2d_axis1 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 4.; 1.; 3.; 2.; 5.; 6. |] in
  let result, indices = Nx.sort ~axis:1 t in
  check_t "sort 2D axis 1 values" [| 2; 3 |] [| 1.; 3.; 4.; 2.; 5.; 6. |] result;
  check_t "sort 2D axis 1 indices" [| 2; 3 |]
    [| 1l; 2l; 0l; 0l; 1l; 2l |]
    indices

let test_sort_invalid_axis () =
  let t = Nx.create Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  check_invalid_arg "sort invalid axis"
    "sort: invalid axis 2 (out of bounds for 2D array)" (fun () ->
      Nx.sort ~axis:2 t)

let test_sort_nan_handling () =
  let t = Nx.create Nx.float32 [| 5 |] [| 3.; nan; 1.; 2.; nan |] in
  let result, _ = Nx.sort t in
  (* NaN values should be sorted to the end *)
  let first_three = Nx.slice [ Nx.R (0, 3) ] result in
  check_t "sort NaN handling - non-NaN values" [| 3 |] [| 1.; 2.; 3. |]
    first_three;
  (* Check that last two values are NaN *)
  equal ~msg:"sort NaN handling - NaN at end" bool true
    (Float.is_nan (Nx.item [ 3 ] result) && Float.is_nan (Nx.item [ 4 ] result))

let test_sort_stable () =
  (* Test sort stability with repeated values *)
  let t = Nx.create Nx.float32 [| 6 |] [| 3.; 1.; 2.; 1.; 3.; 2. |] in
  let _, indices = Nx.sort t in
  (* For stable sort, original order should be preserved for equal elements *)
  check_t "sort stable indices" [| 6 |] [| 1l; 3l; 2l; 5l; 0l; 4l |] indices

(* ───── Argsort Tests ───── *)

let test_argsort_1d () =
  let t = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 4.; 1.; 5. |] in
  let result = Nx.argsort t in
  check_t "argsort 1D" [| 5 |] [| 1l; 3l; 0l; 2l; 4l |] result

let test_argsort_2d_axis0 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 4.; 1.; 3.; 2.; 5.; 6. |] in
  let result = Nx.argsort ~axis:0 t in
  check_t "argsort 2D axis 0" [| 2; 3 |] [| 1l; 0l; 0l; 0l; 1l; 1l |] result

let test_argsort_2d_axis1 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 4.; 1.; 3.; 2.; 5.; 6. |] in
  let result = Nx.argsort ~axis:1 t in
  check_t "argsort 2D axis 1" [| 2; 3 |] [| 1l; 2l; 0l; 0l; 1l; 2l |] result

let test_argsort_empty () =
  let t = Nx.create Nx.float32 [| 0 |] [||] in
  let result = Nx.argsort t in
  check_t "argsort empty" [| 0 |] [||] result

(* ───── Argmax Tests ───── *)

let test_argmax_1d () =
  let t = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 4.; 1.; 5. |] in
  let result = Nx.argmax t in
  check_t "argmax 1D" [||] [| 4l |] result

let test_argmax_2d_axis0 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let result = Nx.argmax ~axis:0 t in
  check_t "argmax 2D axis 0" [| 3 |] [| 1l; 1l; 1l |] result

let test_argmax_2d_axis1 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let result = Nx.argmax ~axis:1 t in
  check_t "argmax 2D axis 1" [| 2 |] [| 2l; 2l |] result

let test_argmax_keepdims () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let result = Nx.argmax ~axis:1 ~keepdims:true t in
  check_shape "argmax keepdims shape" [| 2; 1 |] result;
  check_t "argmax keepdims values" [| 2; 1 |] [| 2l; 2l |] result

let test_argmax_nan () =
  let t = Nx.create Nx.float32 [| 4 |] [| 1.; nan; 3.; 2. |] in
  let result = Nx.argmax t in
  (* NaN handling may vary - just check it doesn't crash *)
  check_shape "argmax with NaN" [||] result

(* ───── Argmin Tests ───── *)

let test_argmin_1d () =
  let t = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 4.; 1.; 5. |] in
  let result = Nx.argmin t in
  check_t "argmin 1D" [||] [| 1l |] result

let test_argmin_2d_axis0 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let result = Nx.argmin ~axis:0 t in
  check_t "argmin 2D axis 0" [| 3 |] [| 0l; 0l; 0l |] result

let test_argmin_2d_axis1 () =
  let t = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let result = Nx.argmin ~axis:1 t in
  check_t "argmin 2D axis 1" [| 2 |] [| 0l; 0l |] result

let test_argmin_ties () =
  let t = Nx.create Nx.float32 [| 5 |] [| 3.; 1.; 2.; 1.; 3. |] in
  let result = Nx.argmin t in
  (* Should return first occurrence *)
  check_t "argmin ties" [||] [| 1l |] result

(* ───── Sort Regression Tests ───── *)

let test_sort_large_1d () =
  (* Regression: bitonic sort breaks for n >= 129.
     The sort produces duplicate values instead of a correct permutation. *)
  let n = 150 in
  let t = Nx.arange Nx.float32 0 n 1 in
  (* Reverse so it's not already sorted *)
  let t = Nx.flip ~axes:[ 0 ] t in
  let sorted_vals, sorted_indices = Nx.sort t in
  (* Check sorted values are 0, 1, 2, ..., n-1 *)
  let expected_vals = Nx.arange Nx.float32 0 n 1 in
  check_nx "sort large 1D values" expected_vals sorted_vals;
  (* Check indices map back to original positions *)
  let expected_indices = Nx.arange Nx.int32 (n - 1) (-1) (-1) in
  check_nx "sort large 1D indices" expected_indices sorted_indices

let test_sort_power_of_two () =
  (* n=256 is a power of two (no padding needed) but still breaks *)
  let n = 256 in
  let t = Nx.arange Nx.float32 0 n 1 in
  let t = Nx.flip ~axes:[ 0 ] t in
  let sorted_vals, _ = Nx.sort t in
  let expected_vals = Nx.arange Nx.float32 0 n 1 in
  check_nx "sort power-of-two values" expected_vals sorted_vals

let test_sort_128_boundary () =
  (* n=128 works, n=129 does not *)
  let t128 = Nx.flip ~axes:[ 0 ] (Nx.arange Nx.float32 0 128 1) in
  let sorted128, _ = Nx.sort t128 in
  check_nx "sort n=128 values" (Nx.arange Nx.float32 0 128 1) sorted128;
  let t129 = Nx.flip ~axes:[ 0 ] (Nx.arange Nx.float32 0 129 1) in
  let sorted129, _ = Nx.sort t129 in
  check_nx "sort n=129 values" (Nx.arange Nx.float32 0 129 1) sorted129

(* Test Suite Organization *)

let where_tests =
  [
    test "where 1D" test_where_1d;
    test "where broadcast" test_where_broadcast;
    test "where scalar inputs" test_where_scalar_inputs;
    test "where invalid shapes" test_where_invalid_shapes;
  ]

let sort_tests =
  [
    test "sort 2D axis 0" test_sort_2d_axis0;
    test "sort 2D axis 1" test_sort_2d_axis1;
    test "sort invalid axis" test_sort_invalid_axis;
    test "sort NaN handling" test_sort_nan_handling;
    test "sort stable" test_sort_stable;
  ]

let sort_regression_tests =
  [
    test "sort large 1D (n=150)" test_sort_large_1d;
    test "sort power of two (n=256)" test_sort_power_of_two;
    test "sort 128 boundary" test_sort_128_boundary;
  ]

let argsort_tests =
  [
    test "argsort 1D" test_argsort_1d;
    test "argsort 2D axis 0" test_argsort_2d_axis0;
    test "argsort 2D axis 1" test_argsort_2d_axis1;
    test "argsort empty" test_argsort_empty;
  ]

let argmax_tests =
  [
    test "argmax 1D" test_argmax_1d;
    test "argmax 2D axis 0" test_argmax_2d_axis0;
    test "argmax 2D axis 1" test_argmax_2d_axis1;
    test "argmax keepdims" test_argmax_keepdims;
    test "argmax NaN" test_argmax_nan;
  ]

let argmin_tests =
  [
    test "argmin 1D" test_argmin_1d;
    test "argmin 2D axis 0" test_argmin_2d_axis0;
    test "argmin 2D axis 1" test_argmin_2d_axis1;
    test "argmin ties" test_argmin_ties;
  ]

let suite =
  [
    group "Where" where_tests;
    group "Sort" sort_tests;
    group "Sort Regression" sort_regression_tests;
    group "Argsort" argsort_tests;
    group "Argmax" argmax_tests;
    group "Argmin" argmin_tests;
  ]

let () = run "Nx Sorting" suite
