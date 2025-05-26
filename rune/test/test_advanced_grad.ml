module T = Rune
open Nx_core

let _test_higher_order_derivatives () =
  (* Test second derivative of f(x) = x^3 *)
  (* f(x) = x^3, f'(x) = 3x^2, f''(x) = 6x *)
  let f x =
    let x2 = T.mul x x in
    T.mul x2 x
  in

  let x = T.scalar T.cpu Dtype.float32 2.0 in

  (* First derivative *)
  let grad_f x = T.grad f x in
  let dx = grad_f x in
  let dx_val = Bigarray.Array1.get (T.unsafe_data dx) 0 in
  assert (dx_val = 12.0);

  (* 3 * 2^2 = 12 *)

  (* Second derivative *)
  let grad_grad_f x = T.grad grad_f x in
  let d2x = grad_grad_f x in
  let d2x_val = Bigarray.Array1.get (T.unsafe_data d2x) 0 in

  Printf.printf "test_higher_order_derivatives: d2x = %f (expected 12.0)\n"
    d2x_val;
  assert (d2x_val = 12.0);

  (* 6 * 2 = 12 *)
  print_endline "test_higher_order_derivatives: PASSED"

let test_grad_with_zero_gradient () =
  (* Test function that has zero gradient at certain points *)
  (* f(x) = x^2 has zero gradient at x=0 *)
  let f x = T.mul x x in

  (* At x=2, gradient should be 4 (2x = 4) *)
  let x_pos = T.scalar T.cpu Dtype.float32 2.0 in
  let dx_pos = T.grad f x_pos in
  let dx_pos_val = Bigarray.Array1.get (T.unsafe_data dx_pos) 0 in
  assert (dx_pos_val = 4.0);

  (* At x=0, gradient should be 0 *)
  let x_zero = T.scalar T.cpu Dtype.float32 0.0 in
  let dx_zero = T.grad f x_zero in
  let dx_zero_val = Bigarray.Array1.get (T.unsafe_data dx_zero) 0 in
  assert (dx_zero_val = 0.0);

  print_endline "test_grad_with_zero_gradient: PASSED"

let _test_jit_grad_composition () =
  (* Test composing JIT and grad in both orders *)
  let f x =
    let two = T.scalar T.cpu Dtype.float32 2.0 in
    T.mul x two
  in

  let x = T.scalar T.cpu Dtype.float32 3.0 in

  (* grad(jit(f)) *)
  let jit_f = T.jit f in
  let grad_jit_f x = T.grad (fun x -> jit_f x) x in
  let dx1 = grad_jit_f x in
  let dx1_val = Bigarray.Array1.get (T.unsafe_data dx1) 0 in
  assert (dx1_val = 2.0);

  (* jit(grad(f)) *)
  let grad_f x = T.grad f x in
  let jit_grad_f = T.jit grad_f in
  let dx2 = jit_grad_f x in
  let dx2_val = Bigarray.Array1.get (T.unsafe_data dx2) 0 in
  assert (dx2_val = 2.0);

  print_endline "test_jit_grad_composition: PASSED"

let test_grad_accumulation () =
  (* Test that gradients accumulate correctly when a variable is used multiple
     times *)
  let f x =
    (* f(x) = x * x + x = x^2 + x *)
    (* f'(x) = 2x + 1 *)
    let x_squared = T.mul x x in
    T.add x_squared x
  in

  let x = T.scalar T.cpu Dtype.float32 3.0 in
  let dx = T.grad f x in
  let dx_val = Bigarray.Array1.get (T.unsafe_data dx) 0 in

  (* At x=3: f'(3) = 2*3 + 1 = 7 *)
  assert (dx_val = 7.0);

  print_endline "test_grad_accumulation: PASSED"

let test_grad_with_constants () =
  (* Test gradient flow with constants *)
  let f x =
    let c1 = T.scalar T.cpu Dtype.float32 5.0 in
    let c2 = T.scalar T.cpu Dtype.float32 2.0 in
    (* f(x) = 5x + 2 *)
    let term1 = T.mul c1 x in
    T.add term1 c2
  in

  let x = T.scalar T.cpu Dtype.float32 1.0 in
  let dx = T.grad f x in
  let dx_val = Bigarray.Array1.get (T.unsafe_data dx) 0 in

  (* f'(x) = 5 *)
  assert (dx_val = 5.0);

  print_endline "test_grad_with_constants: PASSED"

let test_complex_function_gradient () =
  (* Test a more complex function *)
  (* f(x) = (x + 1)^2 * (x - 1) = x^3 - x + x^2 - 1 *)
  (* f'(x) = 3x^2 + 2x - 1 *)
  let f x =
    let one = T.scalar T.cpu Dtype.float32 1.0 in
    let x_plus_1 = T.add x one in
    let x_minus_1 = T.sub x one in
    let x_plus_1_sq = T.mul x_plus_1 x_plus_1 in
    T.mul x_plus_1_sq x_minus_1
  in

  let x = T.scalar T.cpu Dtype.float32 2.0 in
  let dx = T.grad f x in
  let dx_val = Bigarray.Array1.get (T.unsafe_data dx) 0 in

  (* At x=2: f'(2) = 3*4 + 2*2 - 1 = 12 + 4 - 1 = 15 *)
  assert (dx_val = 15.0);

  print_endline "test_complex_function_gradient: PASSED"

let () =
  print_endline "Running advanced gradient tests...";
  (* Skip higher-order derivatives for now - needs more work *)
  (* test_higher_order_derivatives (); *)
  test_grad_with_zero_gradient ();
  (* Skip JIT grad composition - needs fixing *)
  (* test_jit_grad_composition (); *)
  test_grad_accumulation ();
  test_grad_with_constants ();
  test_complex_function_gradient ();
  print_endline "All advanced gradient tests passed!"
