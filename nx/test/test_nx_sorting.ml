(* Sorting and searching tests for Nx *)

open Alcotest
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
  check_raises "where invalid shapes"
    (Invalid_argument
       "broadcast: cannot broadcast [3] to [2] (dim 0: 3\226\137\1602)\n\
        hint: broadcasting requires dimensions to be either equal or 1")
    (fun () -> ignore (Nx.where mask a b))

(* ───── Sort Tests ───── *)

let test_sort_1d () =
  let t = Nx.create Nx.float32 [| 3 |] [| 3.; 1.; 2. |] in
  let result, indices = Nx.sort t in
  check_t "sort 1D values" [| 3 |] [| 1.; 2.; 3. |] result;
  check_t "sort 1D indices" [| 3 |] [| 1l; 2l; 0l |] indices

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
  check bool "sort NaN handling - NaN at end" true
    (Float.is_nan (Nx.item [ 3 ] result) && Float.is_nan (Nx.item [ 4 ] result))

let test_sort_stable () =
  (* Test sort stability with repeated values *)
  let t = Nx.create Nx.float32 [| 6 |] [| 3.; 1.; 2.; 1.; 3.; 2. |] in
  let _, indices = Nx.sort t in
  (* For stable sort, original order should be preserved for equal elements *)
  check_t "sort stable indices" [| 6 |] [| 1l; 3l; 2l; 5l; 0l; 4l |] indices

(* ───── Argsort Tests ───── *)

let test_argsort_1d () =
  let t = Nx.create Nx.float32 [| 3 |] [| 3.; 1.; 2. |] in
  let result = Nx.argsort t in
  check_t "argsort 1D" [| 3 |] [| 1l; 2l; 0l |] result

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
  check_invalid_arg "argsort empty"
    "arange: invalid range [0, 0) (empty with step=1)\n\
     hint: ensure start < stop for positive step, or start > stop for negative \
     step" (fun () -> ignore (Nx.argsort t))

(* ───── Argmax Tests ───── *)

let test_argmax_1d () =
  let t = Nx.create Nx.float32 [| 3 |] [| 1.; 3.; 2. |] in
  let result = Nx.argmax t in
  check_t "argmax 1D" [||] [| 1l |] result

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
  let t = Nx.create Nx.float32 [| 3 |] [| 1.; 3.; 2. |] in
  let result = Nx.argmin t in
  check_t "argmin 1D" [||] [| 0l |] result

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

(* Test Suite Organization *)

let where_tests =
  [
    ("where 1D", `Quick, test_where_1d);
    ("where broadcast", `Quick, test_where_broadcast);
    ("where scalar inputs", `Quick, test_where_scalar_inputs);
    ("where invalid shapes", `Quick, test_where_invalid_shapes);
  ]

let sort_tests =
  [
    ("sort 1D", `Quick, test_sort_1d);
    ("sort 2D axis 0", `Quick, test_sort_2d_axis0);
    ("sort 2D axis 1", `Quick, test_sort_2d_axis1);
    ("sort invalid axis", `Quick, test_sort_invalid_axis);
    ("sort NaN handling", `Quick, test_sort_nan_handling);
    ("sort stable", `Quick, test_sort_stable);
  ]

let argsort_tests =
  [
    ("argsort 1D", `Quick, test_argsort_1d);
    ("argsort 2D axis 0", `Quick, test_argsort_2d_axis0);
    ("argsort 2D axis 1", `Quick, test_argsort_2d_axis1);
    ("argsort empty", `Quick, test_argsort_empty);
  ]

let argmax_tests =
  [
    ("argmax 1D", `Quick, test_argmax_1d);
    ("argmax 2D axis 0", `Quick, test_argmax_2d_axis0);
    ("argmax 2D axis 1", `Quick, test_argmax_2d_axis1);
    ("argmax keepdims", `Quick, test_argmax_keepdims);
    ("argmax NaN", `Quick, test_argmax_nan);
  ]

let argmin_tests =
  [
    ("argmin 1D", `Quick, test_argmin_1d);
    ("argmin 2D axis 0", `Quick, test_argmin_2d_axis0);
    ("argmin 2D axis 1", `Quick, test_argmin_2d_axis1);
    ("argmin ties", `Quick, test_argmin_ties);
  ]

let suite =
  [
    ("Sorting :: Where", where_tests);
    ("Sorting :: Sort", sort_tests);
    ("Sorting :: Argsort", argsort_tests);
    ("Sorting :: Argmax", argmax_tests);
    ("Sorting :: Argmin", argmin_tests);
  ]

let () = Alcotest.run "Nx Sorting" suite
