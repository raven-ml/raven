(* Test for Metal backend unsafe_get operation *)
module Nx = Nx_core.Make_frontend (Nx_metal)
open Nx

let ctx = Nx_metal.create_context ()

let test_metal_unsafe_get () =
  Printf.printf "Testing Metal unsafe_get operation...\n";

  (* Create a simple 2x2 matrix *)
  let matrix = create ctx float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  Printf.printf "Created matrix:\n";
  print matrix;

  (* Try to get element at [0, 1] *)
  Printf.printf "\nTrying to get element at [0, 1]...\n";
  try
    let value = unsafe_get [ 0; 1 ] matrix in
    Printf.printf "Got value: %f\n" value;
    if abs_float (value -. 2.0) < 0.001 then
      Printf.printf "PASS: unsafe_get returns correct value\n"
    else Printf.printf "FAIL: Expected 2.0, got %f\n" value
  with e ->
    Printf.printf "FAIL: %s\n" (Printexc.to_string e);
    Printf.printf "Metal backend missing implementation for get/unsafe_get\n"

let () = test_metal_unsafe_get ()
