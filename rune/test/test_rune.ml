open Rune

let test _test_name operation_desc f =
  Printf.printf "> %s\n" operation_desc;
  try
    let result = f () in
    print result;
    Printf.printf "\n"
  with
  | Invalid_argument msg -> Printf.printf "Error: %s\n\n" msg
  | e -> Printf.printf "Error: %s\n\n" (Printexc.to_string e)

let test_value_and_grad _test_name operation_desc f =
  Printf.printf "> %s\n" operation_desc;
  try
    let value, grad = f () in
    Printf.printf "Value: ";
    print value;
    Printf.printf "\nGradient: ";
    print grad;
    Printf.printf "\n"
  with
  | Invalid_argument msg -> Printf.printf "Error: %s\n\n" msg
  | e -> Printf.printf "Error: %s\n\n" (Printexc.to_string e)

let _test_custom _test_name operation_desc f =
  Printf.printf "> %s\n" operation_desc;
  f ();
  Printf.printf "\n"

let test_f _test_name operation_desc f =
  Printf.printf "> %s\n" operation_desc;
  try
    let _ = f () in
    Printf.printf "Completed without error.\n\n"
  with
  | Invalid_argument msg -> Printf.printf "Error: %s\n\n" msg
  | e -> Printf.printf "Error: %s\n\n" (Printexc.to_string e)

let () = Printexc.record_backtrace true

let () =
  (* Creation Tests *)
  test "Create 2x2 float32 Rune tensor"
    "create float32 [2; 2] [1.0; 2.0; 3.0; 4.0]" (fun () ->
      create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]);

  test "Create 1D int32 Rune tensor" "create int32 [3] [1; 2; 3]" (fun () ->
      create int32 [| 3 |] [| 1l; 2l; 3l |]);

  (* Evaluation Tests *)
  test "Evaluate addition of two tensors"
    "eval (fun x -> add x b) where x = ones [2; 2], b = [[1; 2]; [3; 4]]"
    (fun () ->
      let x = ones float32 [| 2; 2 |] in
      let b = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      eval (fun x -> add x b) x);

  (* Expected: [[2; 3]; [4; 5]] *)
  test "Evaluate multiplication with scalar"
    "eval (fun x -> mul x a) where x = scalar 2.0, a = [[1; 2]; [3; 4]]"
    (fun () ->
      let x = scalar float32 2.0 in
      let a = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      eval (fun x -> mul x a) x);

  (* Expected: [[2; 4]; [6; 8]] *)

  (* Gradient Tests *)
  test "Gradient of sum(add(x, a))"
    "grad (fun x -> sum (add x a)) at x = ones [2; 2], a = [[1; 2]; [3; 4]]"
    (fun () ->
      let a = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      let f x = sum (add x a) in
      let x0 = ones float32 [| 2; 2 |] in
      grad f x0);

  (* Expected: [[1; 1]; [1; 1]] since df/dx_i = 1 *)
  test "Gradient of sum(mul(x, a))"
    "grad (fun x -> sum (mul x a)) at x = ones [2; 2], a = [[1; 2]; [3; 4]]"
    (fun () ->
      let a = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      let f x = sum (mul x a) in
      let x0 = ones float32 [| 2; 2 |] in
      grad f x0);

  (* Expected: [[1; 2]; [3; 4]] since df/dx_i = a_i *)
  test "Gradient of sum(sin(x))"
    "grad (fun x -> sum (sin x)) at x = [[0; pi/2]; [pi; 3pi/2]]" (fun () ->
      let f x = sum (sin x) in
      let x0 = create float32 [| 2; 2 |] [| 0.0; 1.5708; 3.1416; 4.7124 |] in
      grad f x0);

  (* Expected: cos(x0) ≈ [[1; 0]; [-1; 0]] *)
  test "Gradient of sum(matmul(x, a))"
    "grad (fun x -> sum (matmul x a)) at x = [1; 2], a = [[3; 4]; [5; 6]]"
    (fun () ->
      let a = create float32 [| 2; 2 |] [| 3.0; 4.0; 5.0; 6.0 |] in
      let f x = sum (matmul x a) in
      let x0 = create float32 [| 1; 2 |] [| 1.0; 2.0 |] in
      grad f x0);

  (* Value and Gradient Tests *)
  (* Expected: [7; 11] since df/dx_k = sum_j a_{k,j} *)
  test_value_and_grad "Value and gradient of sum(mul(x, a))"
    "value_and_grad (fun x -> sum (mul x a)) at x = ones [2; 2], a = [[1; 2]; \
     [3; 4]]" (fun () ->
      let a = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      let f x = sum (mul x a) in
      let x0 = ones float32 [| 2; 2 |] in
      value_and_grad f x0);

  (* Expected: [[1; 2]; [3; 4]] *)

  (* Error Handling Tests *)
  test_f "Error on incompatible shapes for add"
    "add a b where a = [2; 2], b = [2; 3]" (fun () ->
      let a = create float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      let b = create float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
      add a b)
