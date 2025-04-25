open Alcotest
module Nd = Ndarray

(* Testables *)
let ndarray_float32 : (float, Nd.float32_elt) Nd.t testable =
  Alcotest.testable Ndarray.pp Ndarray.array_equal

let ndarray_int64 : (int64, Nd.int64_elt) Nd.t testable =
  Alcotest.testable Ndarray.pp Ndarray.array_equal

let test_where_1d () =
  let mask = Nd.create Nd.uint8 [| 3 |] [| 1; 0; 1 |] in
  let a = Nd.create Nd.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Nd.create Nd.float32 [| 3 |] [| 4.; 5.; 6. |] in
  let result = Nd.where mask a b in
  let expected = Nd.create Nd.float32 [| 3 |] [| 1.; 5.; 3. |] in
  check ndarray_float32 "where 1D" expected result

let test_where_broadcast () =
  let mask = Nd.create Nd.uint8 [| 2; 1 |] [| 1; 0 |] in
  let a = Nd.create Nd.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = Nd.create Nd.float32 [| 2; 3 |] [| 7.; 8.; 9.; 10.; 11.; 12. |] in
  let result = Nd.where mask a b in
  let expected =
    Nd.create Nd.float32 [| 2; 3 |] [| 1.; 2.; 3.; 10.; 11.; 12. |]
  in
  check ndarray_float32 "where with broadcasting" expected result

let test_where_scalar_inputs () =
  let mask = Nd.create Nd.uint8 [| 2; 3 |] [| 1; 0; 1; 0; 1; 0 |] in
  let a = Nd.create Nd.float32 [||] [| 5.0 |] in
  let b = Nd.create Nd.float32 [||] [| 10.0 |] in
  let result = Nd.where mask a b in
  let expected =
    Nd.create Nd.float32 [| 2; 3 |] [| 5.0; 10.0; 5.0; 10.0; 5.0; 10.0 |]
  in
  check ndarray_float32 "where with scalar inputs" expected result

let test_where_invalid_shapes () =
  let mask = Nd.create Nd.uint8 [| 2 |] [| 1; 0 |] in
  let a = Nd.create Nd.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Nd.create Nd.float32 [| 2 |] [| 4.; 5. |] in
  check_raises "where invalid shapes"
    (Invalid_argument
       "broadcast_shapes: shapes [3] and [2] cannot be broadcast together")
    (fun () -> ignore (Nd.where mask a b))

let test_sort_1d () =
  let t = Nd.create Nd.float32 [| 3 |] [| 3.; 1.; 2. |] in
  let result = Nd.sort t in
  let expected = Nd.create Nd.float32 [| 3 |] [| 1.; 2.; 3. |] in
  check ndarray_float32 "sort 1D" expected result

let test_sort_2d_axis0 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 4.; 1.; 3.; 2.; 5.; 6. |] in
  let result = Nd.sort ~axis:0 t in
  let expected = Nd.create Nd.float32 [| 2; 3 |] [| 2.; 1.; 3.; 4.; 5.; 6. |] in
  check ndarray_float32 "sort 2D axis 0" expected result

let test_sort_2d_axis1 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 4.; 1.; 3.; 2.; 5.; 6. |] in
  let result = Nd.sort ~axis:1 t in
  let expected = Nd.create Nd.float32 [| 2; 3 |] [| 1.; 3.; 4.; 2.; 5.; 6. |] in
  check ndarray_float32 "sort 2D axis 1" expected result

let test_sort_invalid_axis () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  check_raises "sort invalid axis" (Invalid_argument "sort: axis out of bounds")
    (fun () -> ignore (Nd.sort ~axis:2 t))

let test_argsort_1d () =
  let t = Nd.create Nd.float32 [| 3 |] [| 3.; 1.; 2. |] in
  let result = Nd.argsort t in
  let expected = Nd.create Nd.int64 [| 3 |] [| 1L; 2L; 0L |] in
  check ndarray_int64 "argsort 1D" expected result

let test_argsort_2d_axis0 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 4.; 1.; 3.; 2.; 5.; 6. |] in
  let result = Nd.argsort ~axis:0 t in
  let expected = Nd.create Nd.int64 [| 2; 3 |] [| 1L; 0L; 0L; 0L; 1L; 1L |] in
  check ndarray_int64 "argsort 2D axis 0" expected result

let test_argsort_2d_axis1 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 4.; 1.; 3.; 2.; 5.; 6. |] in
  let result = Nd.argsort ~axis:1 t in
  let expected = Nd.create Nd.int64 [| 2; 3 |] [| 1L; 2L; 0L; 0L; 1L; 2L |] in
  check ndarray_int64 "argsort 2D axis 1" expected result

let test_argmax_1d () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.; 3.; 2. |] in
  let result = Nd.argmax t in
  let expected = Nd.create Nd.int64 [||] [| 1L |] in
  check ndarray_int64 "argmax 1D" expected result

let test_argmax_2d_axis0 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let result = Nd.argmax ~axis:0 t in
  let expected = Nd.create Nd.int64 [| 3 |] [| 1L; 1L; 1L |] in
  check ndarray_int64 "argmax 2D axis 0" expected result

let test_argmax_2d_axis1 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let result = Nd.argmax ~axis:1 t in
  let expected = Nd.create Nd.int64 [| 2 |] [| 2L; 2L |] in
  check ndarray_int64 "argmax 2D axis 1" expected result

let test_argmin_1d () =
  let t = Nd.create Nd.float32 [| 3 |] [| 1.; 3.; 2. |] in
  let result = Nd.argmin t in
  let expected = Nd.create Nd.int64 [||] [| 0L |] in
  check ndarray_int64 "argmin 1D" expected result

let test_argmin_2d_axis0 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let result = Nd.argmin ~axis:0 t in
  let expected = Nd.create Nd.int64 [| 3 |] [| 0L; 0L; 0L |] in
  check ndarray_int64 "argmin 2D axis 0" expected result

let test_argmin_2d_axis1 () =
  let t = Nd.create Nd.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let result = Nd.argmin ~axis:1 t in
  let expected = Nd.create Nd.int64 [| 2 |] [| 0L; 0L |] in
  check ndarray_int64 "argmin 2D axis 1" expected result

let test_unique_1d () =
  let t = Nd.create Nd.float32 [| 5 |] [| 3.; 1.; 2.; 3.; 2. |] in
  let result = Nd.unique t in
  let expected = Nd.create Nd.float32 [| 3 |] [| 1.; 2.; 3. |] in
  check ndarray_float32 "unique 1D" expected result

let test_unique_2d () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 1.; 2.; 2.; 3. |] in
  let result = Nd.unique t in
  let expected = Nd.create Nd.float32 [| 3 |] [| 1.; 2.; 3. |] in
  check ndarray_float32 "unique 2D" expected result

let test_unique_empty () =
  let t = Nd.create Nd.float32 [| 0 |] [||] in
  let result = Nd.unique t in
  let expected = Nd.create Nd.float32 [| 0 |] [||] in
  check ndarray_float32 "unique empty" expected result

let test_nonzero_1d () =
  let t = Nd.create Nd.float32 [| 4 |] [| 0.; 1.; 0.; 2. |] in
  let result = Nd.nonzero t in
  let expected = Nd.create Nd.int64 [| 1; 2 |] [| 1L; 3L |] in
  check ndarray_int64 "nonzero 1D" expected result

let test_nonzero_2d () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 0.; 1.; 2.; 0. |] in
  let result = Nd.nonzero t in
  let expected = Nd.create Nd.int64 [| 2; 2 |] [| 0L; 1L; 1L; 0L |] in
  check ndarray_int64 "nonzero 2D" expected result

let test_nonzero_all_zeros () =
  let t = Nd.create Nd.float32 [| 2; 2 |] [| 0.; 0.; 0.; 0. |] in
  let result = Nd.nonzero t in
  let expected = Nd.create Nd.int64 [| 2; 0 |] [||] in
  check ndarray_int64 "nonzero all zeros" expected result

(* Test list *)
let sorting_searching_unique_tests =
  [
    ("where 1D", `Quick, test_where_1d);
    ("where with broadcasting", `Quick, test_where_broadcast);
    ("where with scalar inputs", `Quick, test_where_scalar_inputs);
    ("where invalid shapes", `Quick, test_where_invalid_shapes);
    ("sort 1D", `Quick, test_sort_1d);
    ("sort 2D axis 0", `Quick, test_sort_2d_axis0);
    ("sort 2D axis 1", `Quick, test_sort_2d_axis1);
    ("sort invalid axis", `Quick, test_sort_invalid_axis);
    ("argsort 1D", `Quick, test_argsort_1d);
    ("argsort 2D axis 0", `Quick, test_argsort_2d_axis0);
    ("argsort 2D axis 1", `Quick, test_argsort_2d_axis1);
    ("argmax 1D", `Quick, test_argmax_1d);
    ("argmax 2D axis 0", `Quick, test_argmax_2d_axis0);
    ("argmax 2D axis 1", `Quick, test_argmax_2d_axis1);
    ("argmin 1D", `Quick, test_argmin_1d);
    ("argmin 2D axis 0", `Quick, test_argmin_2d_axis0);
    ("argmin 2D axis 1", `Quick, test_argmin_2d_axis1);
    ("unique 1D", `Quick, test_unique_1d);
    ("unique 2D", `Quick, test_unique_2d);
    ("unique empty", `Quick, test_unique_empty);
    ("nonzero 1D", `Quick, test_nonzero_1d);
    ("nonzero 2D", `Quick, test_nonzero_2d);
    ("nonzero all zeros", `Quick, test_nonzero_all_zeros);
  ]

(* Run the tests *)
let () =
  Printexc.record_backtrace true;
  Alcotest.run "Ndarray Sorting Searching Unique"
    [ ("Tests", sorting_searching_unique_tests) ]
