module T = Rune
open Nx_core

let test_basic_operations () =
  (* Test basic tensor creation and operations *)
  let x = T.full T.cpu Dtype.float32 [| 3; 3 |] 2.0 in
  let y = T.full T.cpu Dtype.float32 [| 3; 3 |] 3.0 in
  let z = T.add x y in

  (* Verify the result *)
  let z_data = T.unsafe_data z in
  let expected = 5.0 in
  for i = 0 to 8 do
    assert (Bigarray.Array1.get z_data i = expected)
  done;
  print_endline "test_basic_operations: PASSED"

let test_grad_simple () =
  (* Test gradient of f(x) = x^2 at x = 3.0 *)
  (* Gradient should be 2*x = 6.0 *)
  let f x = T.mul x x in

  let x = T.scalar T.cpu Dtype.float32 3.0 in
  let dx = T.grad f x in

  let dx_data = T.unsafe_data dx in
  let expected_grad = 6.0 in
  let actual_grad = Bigarray.Array1.get dx_data 0 in
  Printf.printf "Expected gradient: %f, Actual gradient: %f\n" expected_grad
    actual_grad;
  assert (actual_grad = expected_grad);
  print_endline "test_grad_simple: PASSED"

let test_grad_multi_input () =
  (* Test gradient of f(x, y) = x * y at x = 2.0, y = 3.0 *)
  (* df/dx = y = 3.0, df/dy = x = 2.0 *)
  let f x y = T.mul x y in

  let x = T.scalar T.cpu Dtype.float32 2.0 in
  let y = T.scalar T.cpu Dtype.float32 3.0 in

  let dx = T.grad (fun x -> f x y) x in
  let dy = T.grad (fun y -> f x y) y in

  let dx_data = T.unsafe_data dx in
  let dy_data = T.unsafe_data dy in

  assert (Bigarray.Array1.get dx_data 0 = 3.0);
  assert (Bigarray.Array1.get dy_data 0 = 2.0);
  print_endline "test_grad_multi_input: PASSED"

let test_jit_simple () =
  (* Test JIT compilation of simple addition *)
  let f x y = T.add x y in

  let x = T.full T.cpu Dtype.float32 [| 2; 2 |] 1.0 in
  let y = T.full T.cpu Dtype.float32 [| 2; 2 |] 2.0 in

  (* JIT expects single input, so we need to create a wrapper *)
  let f_single x = f x y in
  let jit_f_single = T.jit f_single in
  let z = jit_f_single x in

  let z_data = T.unsafe_data z in
  let expected = 3.0 in
  for i = 0 to 3 do
    assert (Bigarray.Array1.get z_data i = expected)
  done;
  print_endline "test_jit_simple: PASSED"

let test_jit_with_grad () =
  (* Test composition of JIT and gradient *)
  let f x =
    let y = T.mul x x in
    T.add y x
  in

  (* JIT compile the gradient function *)
  let grad_f x = T.grad f x in
  let jit_grad_f = T.jit grad_f in

  let x = T.scalar T.cpu Dtype.float32 2.0 in
  let dx = jit_grad_f x in

  (* f(x) = x^2 + x, so f'(x) = 2x + 1 *)
  (* At x = 2.0, f'(2) = 4 + 1 = 5.0 *)
  let dx_data = T.unsafe_data dx in
  assert (Bigarray.Array1.get dx_data 0 = 5.0);
  print_endline "test_jit_with_grad: PASSED"

let () =
  print_endline "Running Rune tests...";
  test_basic_operations ();
  test_grad_simple ();
  test_grad_multi_input ();
  test_jit_simple ();
  test_jit_with_grad ();
  print_endline "All tests passed!"
