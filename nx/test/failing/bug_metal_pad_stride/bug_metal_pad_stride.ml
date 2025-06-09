(* Test for Metal pad operation missing stride support *)
module Nx = Nx_core.Make_frontend (Nx_metal)
open Nx

let ctx = Nx_metal.create_context ()

let test_metal_pad_stride () =
  Printf.printf "Testing Metal pad operation with non-contiguous views\n";
  
  (* Create a matrix and take a transposed view (non-contiguous) *)
  let matrix = create ctx float32 [|3; 2|] [|
    1.; 2.;
    3.; 4.;
    5.; 6.
  |] in
  let transposed = transpose matrix in
  
  Printf.printf "Original matrix shape: [3, 2]\n";
  Printf.printf "Transposed view shape: [2, 3]\n";
  Printf.printf "Transposed is contiguous: %b\n" (is_c_contiguous transposed);
  
  (* Pad the transposed view *)
  (* Pad 1 element on each side for both dimensions *)
  let padded = pad [|(1, 1); (1, 1)|] 0. transposed in
  
  Printf.printf "\nPadded transposed view:\n";
  print padded;
  
  (* Expected result:
     Transposed matrix is:
     [[1, 3, 5],
      [2, 4, 6]]
     
     After padding with 1 on each side:
     [[0, 0, 0, 0, 0],
      [0, 1, 3, 5, 0],
      [0, 2, 4, 6, 0],
      [0, 0, 0, 0, 0]]
  *)
  let expected = create ctx float32 [|4; 5|] [|
    0.; 0.; 0.; 0.; 0.;
    0.; 1.; 3.; 5.; 0.;
    0.; 2.; 4.; 6.; 0.;
    0.; 0.; 0.; 0.; 0.
  |] in
  
  (* Check if the result is correct *)
  let correct = ref true in
  for i = 0 to 3 do
    for j = 0 to 4 do
      let expected_val = unsafe_get [i; j] expected in
      let actual_val = unsafe_get [i; j] padded in
      if abs_float (expected_val -. actual_val) > 0.001 then (
        Printf.printf "FAIL: Index [%d,%d]: expected %.1f, got %.1f\n" i j expected_val actual_val;
        correct := false
      )
    done
  done;
  
  if !correct then
    Printf.printf "\nPASS: Pad operation handles non-contiguous views correctly\n"
  else (
    Printf.printf "\nFAIL: Pad operation produces incorrect results on non-contiguous views\n";
    Printf.printf "This is a Metal-specific bug - pad kernel doesn't handle strides\n"
  )

let () = test_metal_pad_stride ()