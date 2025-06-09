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
  
  (* Try to compare the arrays by converting to bigarray *)
  Printf.printf "\nExpected padded result:\n";
  print expected;
  
  (* The test demonstrates the issue - if pad doesn't handle strides correctly,
     it will read from wrong memory locations when padding the transposed view *)
  Printf.printf "\nThis test demonstrates that Metal pad needs stride support for non-contiguous views\n"

let () = test_metal_pad_stride ()