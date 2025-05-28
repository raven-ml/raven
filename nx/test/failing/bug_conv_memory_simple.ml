open Nx

(* Simplified version of the convolution memory corruption bug

   This is a minimal test case that triggers memory corruption with conv2d. The
   issue appears to be related to the pool function creating many intermediate
   tensors that cause memory management issues. *)

let () =
  Printf.printf "Simple convolution memory corruption test...\n";
  flush stdout;

  (* This specific size combination reliably triggers the crash *)
  let x = randn float32 ~seed:42 [| 1; 8; 32; 32 |] in
  let k = randn float32 ~seed:43 [| 16; 8; 3; 3 |] in

  Printf.printf "Input shape: [1, 8, 32, 32]\n";
  Printf.printf "Kernel shape: [16, 8, 3, 3]\n";
  Printf.printf "Running convolution...\n";
  flush stdout;

  (* Even a single convolution can trigger the crash *)
  let result = convolve2d ~padding_mode:`Valid x k in

  Printf.printf "Result shape: %s\n"
    (result |> shape |> Array.to_list |> List.map string_of_int
   |> String.concat ", ");

  (* Force GC to potentially trigger the issue *)
  Gc.full_major ();

  Printf.printf "Test completed!\n"
