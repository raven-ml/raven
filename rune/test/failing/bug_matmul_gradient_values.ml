(* Test to verify correct matmul gradient values *)
open Rune

let () =
  (* Input matrices *)
  let a = create cpu float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = create cpu float32 [| 3; 2 |] [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6 |] in

  Printf.printf "Testing matmul gradient computation\n";
  Printf.printf "a = \n";
  print a;
  Printf.printf "\nb = \n";
  print b;

  (* Compute c = a @ b *)
  let c = matmul a b in
  Printf.printf "\nc = a @ b = \n";
  print c;

  (* Define loss as sum of all elements *)
  let f_a a = sum (matmul a b) in
  let f_b b = sum (matmul a b) in

  (* Compute gradients *)
  let grad_a = grad f_a a in
  let grad_b = grad f_b b in

  Printf.printf "\nGradients:\n";
  Printf.printf "grad_a = \n";
  print grad_a;
  Printf.printf "\ngrad_b = \n";
  print grad_b;

  (* Manual computation: For f(A,B) = sum(A @ B), the gradients are: - ∂f/∂A = 1
     @ B^T (where 1 is a matrix of ones with shape of A @ B) - ∂f/∂B = A^T @ 1

     Since c has shape [2, 2], the "1" matrix is [[1, 1], [1, 1]]

     grad_a should be: [[1, 1], [1, 1]] @ B^T B^T = [[0.1, 0.3, 0.5], [0.2, 0.4,
     0.6]]

     grad_a = [[0.1+0.2, 0.3+0.4, 0.5+0.6], [0.1+0.2, 0.3+0.4, 0.5+0.6]] =
     [[0.3, 0.7, 1.1], [0.3, 0.7, 1.1]] *)
  Printf.printf
    "\nExpected grad_a (from test): [[0.3, 0.7, 1.1], [0.3, 0.7, 1.1]]\n";
  Printf.printf "This matches the manual calculation!\n";

  Printf.printf "\nBut we got: ";
  print grad_a;
  Printf.printf
    "\n\
     This suggests the gradient computation might be using a different formula.\n"
