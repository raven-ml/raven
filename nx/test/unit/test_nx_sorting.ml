(* Sorting and searching tests for Nx *)

open Alcotest

module Make (Backend : Nx_core.Backend_intf.S) = struct
  module Support = Test_nx_support.Make (Backend)
  module Nx = Support.Nx
  open Support

  (* ───── Where Tests ───── *)

  let test_where_1d ctx () =
    let mask = Nx.create ctx Nx.uint8 [| 3 |] [| 1; 0; 1 |] in
    let a = Nx.create ctx Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
    let b = Nx.create ctx Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
    let result = Nx.where mask a b in
    check_t "where 1D" [| 3 |] [| 1.; 5.; 3. |] result

  let test_where_broadcast ctx () =
    let mask = Nx.create ctx Nx.uint8 [| 2; 1 |] [| 1; 0 |] in
    let a = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let b =
      Nx.create ctx Nx.float32 [| 2; 3 |] [| 7.; 8.; 9.; 10.; 11.; 12. |]
    in
    let result = Nx.where mask a b in
    check_t "where with broadcasting" [| 2; 3 |]
      [| 1.; 2.; 3.; 10.; 11.; 12. |]
      result

  let test_where_scalar_inputs ctx () =
    let mask = Nx.create ctx Nx.uint8 [| 2; 3 |] [| 1; 0; 1; 0; 1; 0 |] in
    let a = Nx.scalar ctx Nx.float32 5.0 in
    let b = Nx.scalar ctx Nx.float32 10.0 in
    let result = Nx.where mask a b in
    check_t "where with scalar inputs" [| 2; 3 |]
      [| 5.0; 10.0; 5.0; 10.0; 5.0; 10.0 |]
      result

  let test_where_invalid_shapes ctx () =
    let mask = Nx.create ctx Nx.uint8 [| 2 |] [| 1; 0 |] in
    let a = Nx.create ctx Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
    let b = Nx.create ctx Nx.float32 [| 2 |] [| 4.; 5. |] in
    check_raises "where invalid shapes"
      (Invalid_argument
         "broadcast: cannot broadcast [3] to [2] (dim 0: 3\226\137\1602)\n\
          hint: broadcasting requires dimensions to be either equal or 1")
      (fun () -> ignore (Nx.where mask a b))

  (* ───── Sort Tests ───── *)

  let test_sort_1d ctx () =
    let t = Nx.create ctx Nx.float32 [| 3 |] [| 3.; 1.; 2. |] in
    let result, indices = Nx.sort t in
    check_t "sort 1D values" [| 3 |] [| 1.; 2.; 3. |] result;
    check_t "sort 1D indices" [| 3 |] [| 1l; 2l; 0l |] indices

  let test_sort_2d_axis0 ctx () =
    let t = Nx.create ctx Nx.float32 [| 2; 3 |] [| 4.; 1.; 3.; 2.; 5.; 6. |] in
    let result, indices = Nx.sort ~axis:0 t in
    check_t "sort 2D axis 0 values" [| 2; 3 |]
      [| 2.; 1.; 3.; 4.; 5.; 6. |]
      result;
    check_t "sort 2D axis 0 indices" [| 2; 3 |]
      [| 1l; 0l; 0l; 0l; 1l; 1l |]
      indices

  let test_sort_2d_axis1 ctx () =
    let t = Nx.create ctx Nx.float32 [| 2; 3 |] [| 4.; 1.; 3.; 2.; 5.; 6. |] in
    let result, indices = Nx.sort ~axis:1 t in
    check_t "sort 2D axis 1 values" [| 2; 3 |]
      [| 1.; 3.; 4.; 2.; 5.; 6. |]
      result;
    check_t "sort 2D axis 1 indices" [| 2; 3 |]
      [| 1l; 2l; 0l; 0l; 1l; 2l |]
      indices

  let test_sort_invalid_axis ctx () =
    let t = Nx.create ctx Nx.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
    check_invalid_arg "sort invalid axis"
      "sort: invalid axis 2 (out of bounds for 2D array)" (fun () ->
        Nx.sort ~axis:2 t)

  let test_sort_nan_handling ctx () =
    let t = Nx.create ctx Nx.float32 [| 5 |] [| 3.; nan; 1.; 2.; nan |] in
    let result, _ = Nx.sort t in
    (* NaN values should be sorted to the end *)
    let first_three = Nx.slice [ Nx.R (0, 3) ] result in
    check_t "sort NaN handling - non-NaN values" [| 3 |] [| 1.; 2.; 3. |]
      first_three;
    (* Check that last two values are NaN *)
    check bool "sort NaN handling - NaN at end" true
      (Float.is_nan (Nx.item [ 3 ] result)
      && Float.is_nan (Nx.item [ 4 ] result))

  let test_sort_stable ctx () =
    (* Test sort stability with repeated values *)
    let t = Nx.create ctx Nx.float32 [| 6 |] [| 3.; 1.; 2.; 1.; 3.; 2. |] in
    let _, indices = Nx.sort t in
    (* For stable sort, original order should be preserved for equal elements *)
    check_t "sort stable indices" [| 6 |] [| 1l; 3l; 2l; 5l; 0l; 4l |] indices

  (* ───── Argsort Tests ───── *)

  let test_argsort_1d ctx () =
    let t = Nx.create ctx Nx.float32 [| 3 |] [| 3.; 1.; 2. |] in
    let result = Nx.argsort t in
    check_t "argsort 1D" [| 3 |] [| 1l; 2l; 0l |] result

  let test_argsort_2d_axis0 ctx () =
    let t = Nx.create ctx Nx.float32 [| 2; 3 |] [| 4.; 1.; 3.; 2.; 5.; 6. |] in
    let result = Nx.argsort ~axis:0 t in
    check_t "argsort 2D axis 0" [| 2; 3 |] [| 1l; 0l; 0l; 0l; 1l; 1l |] result

  let test_argsort_2d_axis1 ctx () =
    let t = Nx.create ctx Nx.float32 [| 2; 3 |] [| 4.; 1.; 3.; 2.; 5.; 6. |] in
    let result = Nx.argsort ~axis:1 t in
    check_t "argsort 2D axis 1" [| 2; 3 |] [| 1l; 2l; 0l; 0l; 1l; 2l |] result

  let test_argsort_empty ctx () =
    let t = Nx.create ctx Nx.float32 [| 0 |] [||] in
    check_invalid_arg "argsort empty"
      "arange: invalid range [0, 0) (empty with step=1)\n\
       hint: ensure start < stop for positive step, or start > stop for \
       negative step" (fun () -> ignore (Nx.argsort t))

  (* ───── Argmax Tests ───── *)

  let test_argmax_1d ctx () =
    let t = Nx.create ctx Nx.float32 [| 3 |] [| 1.; 3.; 2. |] in
    let result = Nx.argmax t in
    check_t "argmax 1D" [||] [| 1l |] result

  let test_argmax_2d_axis0 ctx () =
    let t = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let result = Nx.argmax ~axis:0 t in
    check_t "argmax 2D axis 0" [| 3 |] [| 1l; 1l; 1l |] result

  let test_argmax_2d_axis1 ctx () =
    let t = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let result = Nx.argmax ~axis:1 t in
    check_t "argmax 2D axis 1" [| 2 |] [| 2l; 2l |] result

  let test_argmax_keepdims ctx () =
    let t = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let result = Nx.argmax ~axis:1 ~keepdims:true t in
    check_shape "argmax keepdims shape" [| 2; 1 |] result;
    check_t "argmax keepdims values" [| 2; 1 |] [| 2l; 2l |] result

  let test_argmax_nan ctx () =
    let t = Nx.create ctx Nx.float32 [| 4 |] [| 1.; nan; 3.; 2. |] in
    let result = Nx.argmax t in
    (* NaN handling may vary - just check it doesn't crash *)
    check_shape "argmax with NaN" [||] result

  (* ───── Argmin Tests ───── *)

  let test_argmin_1d ctx () =
    let t = Nx.create ctx Nx.float32 [| 3 |] [| 1.; 3.; 2. |] in
    let result = Nx.argmin t in
    check_t "argmin 1D" [||] [| 0l |] result

  let test_argmin_2d_axis0 ctx () =
    let t = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let result = Nx.argmin ~axis:0 t in
    check_t "argmin 2D axis 0" [| 3 |] [| 0l; 0l; 0l |] result

  let test_argmin_2d_axis1 ctx () =
    let t = Nx.create ctx Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
    let result = Nx.argmin ~axis:1 t in
    check_t "argmin 2D axis 1" [| 2 |] [| 0l; 0l |] result

  let test_argmin_ties ctx () =
    let t = Nx.create ctx Nx.float32 [| 5 |] [| 3.; 1.; 2.; 1.; 3. |] in
    let result = Nx.argmin t in
    (* Should return first occurrence *)
    check_t "argmin ties" [||] [| 1l |] result

  (* Test Suite Organization *)

  let where_tests ctx =
    [
      ("where 1D", `Quick, test_where_1d ctx);
      ("where broadcast", `Quick, test_where_broadcast ctx);
      ("where scalar inputs", `Quick, test_where_scalar_inputs ctx);
      ("where invalid shapes", `Quick, test_where_invalid_shapes ctx);
    ]

  let sort_tests ctx =
    [
      ("sort 1D", `Quick, test_sort_1d ctx);
      ("sort 2D axis 0", `Quick, test_sort_2d_axis0 ctx);
      ("sort 2D axis 1", `Quick, test_sort_2d_axis1 ctx);
      ("sort invalid axis", `Quick, test_sort_invalid_axis ctx);
      ("sort NaN handling", `Quick, test_sort_nan_handling ctx);
      ("sort stable", `Quick, test_sort_stable ctx);
    ]

  let argsort_tests ctx =
    [
      ("argsort 1D", `Quick, test_argsort_1d ctx);
      ("argsort 2D axis 0", `Quick, test_argsort_2d_axis0 ctx);
      ("argsort 2D axis 1", `Quick, test_argsort_2d_axis1 ctx);
      ("argsort empty", `Quick, test_argsort_empty ctx);
    ]

  let argmax_tests ctx =
    [
      ("argmax 1D", `Quick, test_argmax_1d ctx);
      ("argmax 2D axis 0", `Quick, test_argmax_2d_axis0 ctx);
      ("argmax 2D axis 1", `Quick, test_argmax_2d_axis1 ctx);
      ("argmax keepdims", `Quick, test_argmax_keepdims ctx);
      ("argmax NaN", `Quick, test_argmax_nan ctx);
    ]

  let argmin_tests ctx =
    [
      ("argmin 1D", `Quick, test_argmin_1d ctx);
      ("argmin 2D axis 0", `Quick, test_argmin_2d_axis0 ctx);
      ("argmin 2D axis 1", `Quick, test_argmin_2d_axis1 ctx);
      ("argmin ties", `Quick, test_argmin_ties ctx);
    ]

  let suite backend_name ctx =
    [
      ("Sorting :: " ^ backend_name ^ " Where", where_tests ctx);
      ("Sorting :: " ^ backend_name ^ " Sort", sort_tests ctx);
      ("Sorting :: " ^ backend_name ^ " Argsort", argsort_tests ctx);
      ("Sorting :: " ^ backend_name ^ " Argmax", argmax_tests ctx);
      ("Sorting :: " ^ backend_name ^ " Argmin", argmin_tests ctx);
    ]
end
