(* Test for Metal cast operation with complex types *)
module Nx = Nx_core.Make_frontend (Nx_metal)
open Nx

let ctx = Nx_metal.create_context ()

let test_metal_cast_complex () =
  Printf.printf "Testing Metal cast operation with complex types...\n";
  
  (* Create a float tensor *)
  let float_tensor = create ctx float32 [|2; 2|] [|1.0; 2.0; 3.0; 4.0|] in
  Printf.printf "Created float32 tensor:\n";
  print float_tensor;
  
  (* Try to cast to complex64 *)
  Printf.printf "\nTrying to cast float32 to complex64...\n";
  try
    let complex_tensor = astype complex64 float_tensor in
    Printf.printf "Cast succeeded:\n";
    print complex_tensor;
    Printf.printf "FAIL: Metal should not support complex types\n"
  with Failure msg ->
    print_endline ("Expected failure: " ^ msg);
  
  (* Also try creating a complex tensor directly *)
  Printf.printf "\nTrying to create complex64 tensor directly...\n";
  try
    let complex_vals = Array.init 4 (fun i -> {Complex.re = float_of_int i; im = 0.0}) in
    let complex_tensor = create ctx complex64 [|2; 2|] complex_vals in
    Printf.printf "Created complex tensor:\n";
    print complex_tensor;
    Printf.printf "FAIL: Metal should not support complex types\n"
  with e ->
    Printf.printf "PASS: Metal backend correctly rejects complex types: %s\n" (Printexc.to_string e)

let () = test_metal_cast_complex ()