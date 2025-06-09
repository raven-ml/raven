(* Test for Metal reduce operations missing stride/offset support *)
module Nx = Nx_core.Make_frontend (Nx_metal)
open Nx

let ctx = Nx_metal.create_context ()

let test_metal_reduce_stride () =
  Printf.printf "Testing Metal reduce operations with non-contiguous views\n";

  (* Create a matrix and take a transposed view (non-contiguous) *)
  let matrix =
    create ctx float32 [| 3; 4 |]
      [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
  in
  let transposed = transpose matrix in

  Printf.printf "Original matrix shape: [3, 4]\n";
  Printf.printf "Transposed view shape: [4, 3]\n";
  Printf.printf "Transposed is contiguous: %b\n" (is_c_contiguous transposed);

  (* Sum along axis 0 of the transposed view *)
  (* This should sum columns of the original matrix *)
  let sum_result = sum transposed ~axes:[| 0 |] ~keepdims:false in

  Printf.printf "\nSum along axis 0 of transposed view:\n";
  print sum_result;

  (* Expected result: [18, 22, 26] (sum of each column) *)
  let expected = create ctx float32 [| 3 |] [| 18.; 22.; 26. |] in

  (* Check if the result is correct *)
  let correct = ref true in
  for i = 0 to 2 do
    let expected_val = unsafe_get [ i ] expected in
    let actual_val = unsafe_get [ i ] sum_result in
    if abs_float (expected_val -. actual_val) > 0.001 then (
      Printf.printf "FAIL: Index %d: expected %.1f, got %.1f\n" i expected_val
        actual_val;
      correct := false)
  done;

  if !correct then
    Printf.printf
      "\nPASS: Reduce operation handles non-contiguous views correctly\n"
  else
    Printf.printf
      "\n\
       FAIL: Reduce operation produces incorrect results on non-contiguous views\n";
  Printf.printf
    "This is a Metal-specific bug - reduce kernels don't handle strides/offsets\n"

let () =
  (* Only run this test on Metal backend *)
  test_metal_reduce_stride ()
