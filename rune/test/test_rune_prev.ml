open Alcotest
module Rn = Rune

let rune_float32 : (float, Rn.float32_elt, [ `cpu ]) Rn.t testable =
  Alcotest.testable Rune.pp Rune.array_equal

let float = float 1e-10

let rune_float32_approx =
  Alcotest.testable Rn.pp (fun a b ->
      let diff = Rn.sub a b in
      let abs_diff = Rn.abs diff in
      let max_diff = Rn.max abs_diff in
      let max_diff_val = Rn.get [||] max_diff in
      max_diff_val < 1e-4)

let test_create_2x2_float32 () =
  let t = Rn.create Rn.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  check (array int) "Shape" [| 2; 2 |] (Rn.shape t);
  check float "Element [0,0]" 1.0 (Rn.get [| 0; 0 |] t);
  check float "Element [0,1]" 2.0 (Rn.get [| 0; 1 |] t);
  check float "Element [1,0]" 3.0 (Rn.get [| 1; 0 |] t);
  check float "Element [1,1]" 4.0 (Rn.get [| 1; 1 |] t)

let test_create_1d_int32 () =
  let t = Rn.create Rn.int32 [| 3 |] [| 1l; 2l; 3l |] in
  check (array int) "Shape" [| 3 |] (Rn.shape t);
  check int32 "Element [0]" 1l (Rn.get [| 0 |] t);
  check int32 "Element [1]" 2l (Rn.get [| 1 |] t);
  check int32 "Element [2]" 3l (Rn.get [| 2 |] t)

let test_eval_add () =
  let result =
    Rn.eval
      (fun () ->
        let x = Rn.ones Rn.float32 [| 2; 2 |] in
        let b = Rn.create Rn.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
        Rn.add x b)
      ()
  in
  let expected = Rn.create Rn.float32 [| 2; 2 |] [| 2.0; 3.0; 4.0; 5.0 |] in
  check rune_float32 "same tensor" expected result

let test_eval_mul_scalar () =
  let result =
    Rn.eval
      (fun () ->
        let x = Rn.scalar Rn.float32 2.0 in
        let a = Rn.create Rn.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
        Rn.mul x a)
      ()
  in
  let expected = Rn.create Rn.float32 [| 2; 2 |] [| 2.0; 4.0; 6.0; 8.0 |] in
  check rune_float32 "same tensor" expected result

let test_grad_sum_add () =
  let a = Rn.create Rn.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let f x = Rn.sum (Rn.add x a) in
  let x0 = Rn.ones Rn.float32 [| 2; 2 |] in
  let grad_result = Rn.grad f x0 in
  let expected_grad =
    Rn.create Rn.float32 [| 2; 2 |] [| 1.0; 1.0; 1.0; 1.0 |]
  in
  check rune_float32 "same gradient" expected_grad grad_result

let test_grad_sum_mul () =
  let a = Rn.create Rn.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let f x = Rn.sum (Rn.mul x a) in
  let x0 = Rn.ones Rn.float32 [| 2; 2 |] in
  let grad_result = Rn.grad f x0 in
  let expected_grad =
    Rn.create Rn.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
  in
  check rune_float32 "same gradient" expected_grad grad_result

let test_grad_sum_sin () =
  let f x = Rn.sum (Rn.sin x) in
  let x0 = Rn.create Rn.float32 [| 2; 2 |] [| 0.0; 1.5708; 3.1416; 4.7124 |] in
  let grad_result = Rn.grad f x0 in
  let expected_grad =
    Rn.create Rn.float32 [| 2; 2 |] [| 1.0; 0.0; -1.0; 0.0 |]
  in
  check rune_float32_approx "same gradient" expected_grad grad_result

let test_grad_sum_matmul () =
  let a = Rn.create Rn.float32 [| 2; 2 |] [| 3.0; 4.0; 5.0; 6.0 |] in
  let f x = Rn.sum (Rn.matmul x a) in
  let x0 = Rn.create Rn.float32 [| 1; 2 |] [| 1.0; 2.0 |] in
  let grad_result = Rn.grad f x0 in
  let expected_grad = Rn.create Rn.float32 [| 1; 2 |] [| 7.0; 11.0 |] in
  check rune_float32 "same gradient" expected_grad grad_result

let test_grad_sum_with_axes () =
  let x = Rn.create Rn.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let f x =
    let s = Rn.sum x ~axes:[| 1 |] in
    (* Shape [2]: [6.0; 15.0] *)
    let s_squared = Rn.mul s s in
    (* Shape [2]: [36.0; 225.0] *)
    Rn.sum s_squared (* Scalar: 261.0 *)
  in
  let grad_result = Rn.grad f x in
  (* Gradient derivation: f = sum((sum(x, axis=1))^2), ∂f/∂x_{i,j} = 2 * (sum of
     row i), so row 0: 2*6, row 1: 2*15 *)
  let expected_grad =
    Rn.create Rn.float32 [| 2; 3 |] [| 12.0; 12.0; 12.0; 30.0; 30.0; 30.0 |]
  in
  check rune_float32 "same gradient" expected_grad grad_result

let test_grad_log () =
  let x = Rn.create Rn.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let f x = Rn.sum (Rn.log x) in
  let grad_result = Rn.grad f x in
  (* Gradient of sum(log(x)) = 1/x element-wise *)
  let expected_grad =
    Rn.create Rn.float32 [| 3 |] [| 1.0 /. 1.0; 1.0 /. 2.0; 1.0 /. 3.0 |]
  in
  check rune_float32_approx "same gradient" expected_grad grad_result

let test_grad_mean () =
  let x = Rn.create Rn.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let f x = Rn.mean x in
  (* Mean = 21.0 / 6 = 3.5 *)
  let grad_result = Rn.grad f x in
  (* Gradient of mean: 1 / number of elements = 1/6 *)
  let expected_grad =
    Rn.create Rn.float32 [| 2; 3 |] (Array.make 6 (1.0 /. 6.0))
  in
  check rune_float32_approx "same gradient" expected_grad grad_result

let test_value_and_grad_sum_mul () =
  let a = Rn.create Rn.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let f x = Rn.sum (Rn.mul x a) in
  let x0 = Rn.ones Rn.float32 [| 2; 2 |] in
  let value, grad = Rn.value_and_grad f x0 in
  check float "same value" 10.0 (Rn.get [||] value);
  let expected_grad =
    Rn.create Rn.float32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |]
  in
  check rune_float32 "same gradient" expected_grad grad

let test_value_and_grads_multiple_params () =
  let x = Rn.ones Rn.float32 [| 2; 2 |] in
  let w = Rn.ones Rn.float32 [| 2; 2 |] in
  let b = Rn.zeros Rn.float32 [| 2; 2 |] in
  let f params =
    match params with
    | [ w; b ] ->
        Rn.sum (Rn.add (Rn.mul w x) b) (* Linear layer: sum(w*x + b) *)
    | _ -> failwith "invalid"
  in
  let value, grads = Rn.value_and_grads f [ w; b ] in
  let expected_value = Rn.sum (Rn.add (Rn.mul w x) b) in
  (* 4.0 *)
  let expected_grad_w = x in
  (* ∂/∂w = x *)
  let expected_grad_b = Rn.ones_like b in
  (* ∂/∂b = ones *)
  check rune_float32 "same value" expected_value value;
  check rune_float32 "same grad w" expected_grad_w (List.nth grads 0);
  check rune_float32 "same grad b" expected_grad_b (List.nth grads 1)

let test_tanh () =
  let x = Rn.create Rn.float32 [| 3 |] [| -1.0; 0.0; 1.0 |] in
  let result = Rn.tanh x in
  let expected =
    Rn.create Rn.float32 [| 3 |]
      [| -0.7615941559557649; 0.0; 0.7615941559557649 |]
  in
  check rune_float32_approx "tanh" expected result

let test_relu () =
  let x = Rn.create Rn.float32 [| 3 |] [| -1.0; 0.0; 1.0 |] in
  let result = Rn.relu x in
  let expected = Rn.create Rn.float32 [| 3 |] [| 0.0; 0.0; 1.0 |] in
  check rune_float32 "relu" expected result

let test_softmax () =
  let x = Rn.create Rn.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let result = Rn.softmax ~axes:[| 1 |] x in
  let row1 = [| 1.0; 2.0; 3.0 |] in
  let exp_row1 = Array.map exp row1 in
  let sum_exp1 = Array.fold_left ( +. ) 0.0 exp_row1 in
  let softmax_row1 = Array.map (fun e -> e /. sum_exp1) exp_row1 in
  let row2 = [| 4.0; 5.0; 6.0 |] in
  let exp_row2 = Array.map exp row2 in
  let sum_exp2 = Array.fold_left ( +. ) 0.0 exp_row2 in
  let softmax_row2 = Array.map (fun e -> e /. sum_exp2) exp_row2 in
  let expected_data = Array.concat [ softmax_row1; softmax_row2 ] in
  let expected = Rn.create Rn.float32 [| 2; 3 |] expected_data in
  check rune_float32_approx "softmax" expected result

let test_max_keepdims_broadcast () =
  let x = Rn.create Rn.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let max_x = Rn.max ~axes:[| 1 |] ~keepdims:true x in
  let expected_max_shape = [| 2; 1 |] in
  check (array int) "Shape of max_x with keepdims=true" expected_max_shape
    (Rn.shape max_x);
  let x_shifted = Rn.sub x max_x in
  let expected_shifted_shape = [| 2; 3 |] in
  check (array int) "Shape of x_shifted" expected_shifted_shape
    (Rn.shape x_shifted);
  let expected_shifted_data = [| -2.0; -1.0; 0.0; -2.0; -1.0; 0.0 |] in
  let expected_shifted =
    Rn.create Rn.float32 [| 2; 3 |] expected_shifted_data
  in
  check rune_float32_approx "Values of x_shifted" expected_shifted x_shifted

let test_softmax_extended () =
  let x = Rn.create Rn.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let max_x = Rn.max ~axes:[| 1 |] ~keepdims:true x in
  let x_shifted = Rn.sub x max_x in
  let exp_x = Rn.exp x_shifted in
  let sum_exp = Rn.sum exp_x ~axes:[| 1 |] ~keepdims:true in
  let probs = Rn.div exp_x sum_exp in
  let expected_shape = [| 2; 3 |] in
  check (array int) "Shape of softmax output" expected_shape (Rn.shape probs);
  let result = Rn.softmax ~axes:[| 1 |] x in
  check rune_float32_approx "Softmax values match manual computation" probs
    result

let test_grad_max_keepdims () =
  let x = Rn.create Rn.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let f x =
    let max_x = Rn.max ~axes:[| 1 |] ~keepdims:true x in
    let x_shifted = Rn.sub x max_x in
    Rn.sum x_shifted
  in
  let grad_result = Rn.grad f x in
  let expected_shape = [| 2; 3 |] in
  check (array int) "Shape of gradient" expected_shape (Rn.shape grad_result);
  let expected_grad =
    Rn.create Rn.float32 [| 2; 3 |] [| 1.0; 1.0; -2.0; 1.0; 1.0; -2.0 |]
  in
  check rune_float32_approx "Gradient values" expected_grad grad_result

let test_one_hot () =
  let indices = Rn.create Rn.int32 [| 3 |] [| 0l; 1l; 2l |] in
  let result = Rn.one_hot Rn.float32 indices 3 in
  let expected =
    Rn.create Rn.float32 [| 3; 3 |]
      [| 1.0; 0.0; 0.0; 0.0; 1.0; 0.0; 0.0; 0.0; 1.0 |]
  in
  check rune_float32 "one_hot" expected result

let test_vmap_sum () =
  let x = Rn.create Rn.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let fn xs = Rn.sum (List.hd xs) in
  let result = Rn.vmap fn ~in_axes:[ Some 0 ] [ x ] in
  let expected = Rn.create Rn.float32 [| 2 |] [| 6.0; 15.0 |] in
  check rune_float32 "vmap sum" expected result

let test_vmap_add_sum () =
  let a = Rn.create Rn.float32 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let b = Rn.create Rn.float32 [| 2; 3 |] [| 0.5; 1.5; 2.5; 3.5; 4.5; 5.5 |] in
  let fn xs =
    match xs with [ x; y ] -> Rn.sum (Rn.add x y) | _ -> failwith "invalid"
  in
  let result = Rn.vmap fn ~in_axes:[ Some 0; Some 0 ] [ a; b ] in
  let expected = Rn.create Rn.float32 [| 2 |] [| 10.5; 28.5 |] in
  check rune_float32 "vmap add sum" expected result

let test_grad_tanh () =
  let x = Rn.create Rn.float32 [| 3 |] [| -1.0; 0.0; 1.0 |] in
  let f x = Rn.sum (Rn.tanh x) in
  let grad_result = Rn.grad f x in
  let tanh_x = Rn.tanh x in
  let expected_grad = Rn.sub (Rn.ones_like tanh_x) (Rn.mul tanh_x tanh_x) in
  check rune_float32_approx "grad tanh" expected_grad grad_result

let test_grad_relu () =
  let x = Rn.create Rn.float32 [| 3 |] [| -1.0; 0.0; 1.0 |] in
  let f x = Rn.sum (Rn.relu x) in
  let grad_result = Rn.grad f x in
  let expected_grad = Rn.create Rn.float32 [| 3 |] [| 0.0; 0.0; 1.0 |] in
  check rune_float32 "grad relu" expected_grad grad_result

let test_value_and_grad_tanh () =
  let x = Rn.create Rn.float32 [| 3 |] [| -1.0; 0.0; 1.0 |] in
  let f x = Rn.sum (Rn.tanh x) in
  let value, grad = Rn.value_and_grad f x in
  let expected_value = Rn.sum (Rn.tanh x) in
  let tanh_x = Rn.tanh x in
  let expected_grad = Rn.sub (Rn.ones_like tanh_x) (Rn.mul tanh_x tanh_x) in
  check rune_float32_approx "value tanh" expected_value value;
  check rune_float32_approx "grad tanh" expected_grad grad

(* Test Groups *)
let creation_tests =
  [
    ("Create 2x2 float32 Rune tensor", `Quick, test_create_2x2_float32);
    ("Create 1D int32 Rune tensor", `Quick, test_create_1d_int32);
  ]

let evaluation_tests =
  [
    ("Evaluate addition of two tensors", `Quick, test_eval_add);
    ("Evaluate multiplication with scalar", `Quick, test_eval_mul_scalar);
  ]

let gradient_tests =
  [
    ("Gradient of sum(add(x, a))", `Quick, test_grad_sum_add);
    ("Gradient of sum(mul(x, a))", `Quick, test_grad_sum_mul);
    ("Gradient of sum(sin(x))", `Quick, test_grad_sum_sin);
    ("Gradient of sum(matmul(x, a))", `Quick, test_grad_sum_matmul);
    ("Gradient of sum(tanh(x))", `Quick, test_grad_tanh);
    ("Gradient of sum(relu(x))", `Quick, test_grad_relu);
    ("Gradient of sum with axes", `Quick, test_grad_sum_with_axes);
    ("Gradient of mean", `Quick, test_grad_mean);
    ("Gradient of log", `Quick, test_grad_log);
  ]

let broadcasting_softmax_tests =
  [
    ("Max with keepdims and broadcasting", `Quick, test_max_keepdims_broadcast);
    ("Softmax extended with intermediates", `Quick, test_softmax_extended);
    ("Gradient with max and keepdims", `Quick, test_grad_max_keepdims);
  ]

let value_and_grad_tests =
  [
    ("Value and gradient of sum(mul(x, a))", `Quick, test_value_and_grad_sum_mul);
    ("Value and gradient of sum(tanh(x))", `Quick, test_value_and_grad_tanh);
    ( "Value and gradients with multiple params",
      `Quick,
      test_value_and_grads_multiple_params );
  ]

let neural_network_tests =
  [
    ("Tanh", `Quick, test_tanh);
    ("Relu", `Quick, test_relu);
    ("Softmax", `Quick, test_softmax);
    ("One Hot", `Quick, test_one_hot);
  ]

let vmap_tests =
  [
    ("Vmap Sum", `Quick, test_vmap_sum);
    ("Vmap Add Sum", `Quick, test_vmap_add_sum);
  ]

let () =
  Alcotest.run "Rune Tests"
    [
      ("Creation", creation_tests);
      ("Evaluation", evaluation_tests);
      ("Gradient", gradient_tests);
      ("Value and Gradient", value_and_grad_tests);
      ("Neural Network Operations", neural_network_tests);
      ("Vmap", vmap_tests);
      ("Broadcasting and Softmax", broadcasting_softmax_tests);
    ]
