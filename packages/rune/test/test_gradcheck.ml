(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Test_rune_support

module T = struct
  include Nx
  include Rune
end

(* ───── Test Finite Differences ───── *)

let test_finite_diff_simple () =
  let x = T.scalar T.float32 2.0 in
  let f x = T.mul x x in
  let grad_fd = T.finite_diff f x in
  check_scalar ~eps:1e-2 "finite_diff(x²) at x=2" 4.0 (scalar_value grad_fd)

let test_finite_diff_polynomial () =
  let x = T.scalar T.float32 3.0 in
  let f x =
    let x2 = T.mul x x in
    let x3 = T.mul x2 x in
    T.add x3 (T.mul (T.scalar T.float32 2.0) x2)
  in
  let grad_fd = T.finite_diff f x in
  (* Derivative of x³ + 2x² is 3x² + 4x = 3*9 + 4*3 = 27 + 12 = 39 *)
  check_scalar ~eps:0.1 "finite_diff(x³ + 2x²) at x=3" 39.0
    (scalar_value grad_fd)

let test_finite_diff_vector () =
  let x = T.create T.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let f x = T.sum (T.mul x x) in
  let grad_fd = T.finite_diff f x in
  let expected = T.create T.float32 [| 3 |] [| 2.; 4.; 6. |] in
  check_rune ~eps:1e-2 "finite_diff vector gradient" expected grad_fd

let test_finite_diff_methods () =
  let x = T.scalar T.float32 1.0 in
  let f = T.exp in

  let grad_central = T.finite_diff ~method_:`Central f x in
  let grad_forward = T.finite_diff ~method_:`Forward f x in
  let grad_backward = T.finite_diff ~method_:`Backward f x in

  let exp_1 = exp 1.0 in
  check_scalar ~eps:1e-3 "central difference exp'(1)" exp_1
    (scalar_value grad_central);
  check_scalar ~eps:1e-2 "forward difference exp'(1)" exp_1
    (scalar_value grad_forward);
  check_scalar ~eps:1e-2 "backward difference exp'(1)" exp_1
    (scalar_value grad_backward)

(* ───── Test Gradient Checking ───── *)

let test_check_gradient_pass () =
  let x = T.create T.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let f x = T.sum (T.mul x x) in

  match T.check_gradient ~verbose:false f x with
  | `Pass result ->
      equal ~msg:"gradient check passed" bool true result.passed;
      equal ~msg:"no failed indices" bool true (result.failed_indices = [])
  | `Fail _ -> fail "Expected gradient check to pass"

let test_check_gradient_fail () =
  let x = T.scalar T.float32 2.0 in
  let f x =
    let wrong_grad = T.mul x (T.scalar T.float32 3.0) in
    wrong_grad
  in

  let _grad_with_bug _f _x = T.scalar T.float32 2.0 in

  let autodiff_grad = T.grad f x in
  let finite_diff_grad = T.finite_diff f x in

  check_scalar ~eps:1e-3 "autodiff gives 3.0" 3.0 (scalar_value autodiff_grad);
  check_scalar ~eps:5e-3 "finite_diff gives 3.0" 3.0
    (scalar_value finite_diff_grad)

let test_check_gradient_tolerances () =
  let x = T.scalar T.float32 1.0 in
  let f x = T.sin x in

  match T.check_gradient ~rtol:1e-4 ~atol:1e-5 f x with
  | `Pass result ->
      equal ~msg:"gradient check with tight tolerances" bool true result.passed
  | `Fail result ->
      Printf.printf "Max abs error: %.2e, Max rel error: %.2e\n"
        result.max_abs_error result.max_rel_error;
      fail "Gradient check failed unexpectedly"

let test_check_gradient_complex () =
  let x = T.create T.float32 [| 2 |] [| 0.5; 1.5 |] in
  let f x =
    let exp_x = T.exp x in
    let sin_x = T.sin x in
    let prod = T.mul exp_x sin_x in
    T.sum prod
  in

  match T.check_gradient ~verbose:false f x with
  | `Pass result ->
      equal ~msg:"complex function gradient check" bool true result.passed;
      equal ~msg:"low relative error" bool true (result.max_rel_error < 1e-3)
  | `Fail result ->
      Printf.printf "Failed: max_rel_error = %.2e\n" result.max_rel_error;
      fail "Complex gradient check failed"

let test_check_gradients_multiple () =
  let x1 = T.scalar T.float32 2.0 in
  let x2 = T.scalar T.float32 3.0 in
  let f xs =
    match xs with [ a; b ] -> T.mul a b | _ -> failwith "Expected 2 inputs"
  in

  match T.check_gradients ~verbose:false f [ x1; x2 ] with
  | `Pass results ->
      equal ~msg:"number of results" int 2 (List.length results);
      List.iter
        (fun r -> equal ~msg:"each gradient passed" bool true r.T.passed)
        results
  | `Fail _ -> fail "Expected multiple gradients check to pass"

let test_check_gradient_matrix () =
  let x = T.create T.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let f x =
    let xt = T.transpose x in
    let result = T.matmul x xt in
    T.sum result
  in

  (* Matrix operations need looser tolerance due to float32 precision *)
  match T.check_gradient ~verbose:false ~rtol:1e-2 ~atol:1e-2 f x with
  | `Pass result ->
      equal ~msg:"matrix operation gradient check" bool true result.passed
  | `Fail result ->
      Printf.printf "Matrix gradient check failed: max_rel_error = %.2e\n"
        result.max_rel_error;
      fail "Matrix gradient check failed"

let test_finite_diff_jacobian () =
  let x = T.create T.float32 [| 2 |] [| 1.; 2. |] in
  let f x =
    (* Simple function that returns a 2-element vector *)
    (* f(x) = [x1 + x2, x1 * x2] where x = [x1, x2] *)
    let x1 = T.get [ 0 ] x in
    let x2 = T.get [ 1 ] x in
    let sum = T.add x1 x2 in
    let prod = T.mul x1 x2 in
    (* Create result manually without stack *)
    let result = T.zeros T.float32 [| 2 |] in
    T.set [ 0 ] result sum;
    T.set [ 1 ] result prod;
    result
  in

  let jacobian = T.finite_diff_jacobian f x in
  let expected_shape = [| 2; 2 |] in
  equal ~msg:"jacobian shape" (array int) expected_shape (T.shape jacobian)

(* ───── Test Suite ───── *)

let () =
  run "Gradient Checking"
    [
      group "finite_diff"
        [
          test "simple quadratic" test_finite_diff_simple;
          test "polynomial" test_finite_diff_polynomial;
          test "vector gradient" test_finite_diff_vector;
          test "different methods" test_finite_diff_methods;
          test "jacobian" test_finite_diff_jacobian;
        ];
      group "check_gradient"
        [
          test "passing check" test_check_gradient_pass;
          test "verify correctness" test_check_gradient_fail;
          test "tolerance settings" test_check_gradient_tolerances;
          test "complex function" test_check_gradient_complex;
          test "multiple inputs" test_check_gradients_multiple;
          test "matrix operations" test_check_gradient_matrix;
        ];
    ]
