(* Test that matmul computes gradients correctly *)
open Rune

let test_matmul_gradients () =
  (* Create simple matrices *)
  let a = create cpu float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = create cpu float32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  
  (* Define loss function that uses matmul *)
  let loss_fn a =
    let c = matmul a b in
    (* Sum all elements to get scalar loss *)
    sum c
  in
  
  (* Compute gradients *)
  let loss, grads = value_and_grad loss_fn a in
  
  (* Check that gradients are not all zeros *)
  let grad_sum = sum grads |> unsafe_get [] in
  
  Printf.printf "Loss: %.6f\n" (unsafe_get [] loss);
  Printf.printf "Gradient sum: %.6f\n" grad_sum;
  
  (* The gradient should not be zero *)
  if grad_sum = 0.0 then (
    Printf.printf "FAIL: Matmul gradients are all zero!\n";
    exit 1
  ) else
    Printf.printf "PASS: Matmul computes non-zero gradients\n"

let () = test_matmul_gradients ()