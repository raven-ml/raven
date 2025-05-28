open Alcotest
open Test_rune_support
module T = Rune

let ctx = T.cpu (* Default CPU context *)
let eps = 1e-6

(* Single variable gradients *)
let test_grad_simple () =
  let x = T.scalar ctx T.float32 2.0 in
  let f x = T.mul x x in
  let g = T.grad f x in
  check_scalar ~eps "grad(x²) at x=2" 4.0 (scalar_value g)

let test_grad_constant () =
  let x = T.scalar ctx T.float32 3.0 in
  let c = T.scalar ctx T.float32 5.0 in
  let f x = T.add x c in
  let g = T.grad f x in
  check_scalar ~eps "grad(x + 5) at x=3" 1.0 (scalar_value g)

let test_grad_linear () =
  let x = T.scalar ctx T.float32 2.0 in
  let a = T.scalar ctx T.float32 3.0 in
  let f x = T.mul a x in
  let g = T.grad f x in
  check_scalar ~eps "grad(3x) at x=2" 3.0 (scalar_value g)

let test_grad_polynomial () =
  let x = T.scalar ctx T.float32 2.0 in
  let f x =
    let x2 = T.mul x x in
    let x3 = T.mul x2 x in
    T.add (T.add x3 (T.mul x2 (T.scalar ctx T.float32 2.0))) x
  in
  let g = T.grad f x in
  (* y = x³ + 2x² + x, dy/dx = 3x² + 4x + 1 = 12 + 8 + 1 = 21 at x=2 *)
  check_scalar ~eps "grad(x³ + 2x² + x) at x=2" 21.0 (scalar_value g)

(* Unary operation gradients *)
let test_grad_neg () =
  let x = T.scalar ctx T.float32 3.0 in
  let f x = T.neg x in
  let g = T.grad f x in
  check_scalar ~eps "grad(-x) at x=3" (-1.0) (scalar_value g)

let test_grad_exp () =
  let x = T.scalar ctx T.float32 0.0 in
  let f x = T.exp x in
  let g = T.grad f x in
  check_scalar ~eps "grad(exp(x)) at x=0" 1.0 (scalar_value g)

let test_grad_log () =
  let x = T.scalar ctx T.float32 2.0 in
  let f x = T.log x in
  let g = T.grad f x in
  check_scalar ~eps "grad(log(x)) at x=2" 0.5 (scalar_value g)

let test_grad_sin () =
  let x = T.scalar ctx T.float32 0.0 in
  let f x = T.sin x in
  let g = T.grad f x in
  check_scalar ~eps "grad(sin(x)) at x=0" 1.0 (scalar_value g)

let test_grad_cos () =
  let x = T.scalar ctx T.float32 0.0 in
  let f x = T.cos x in
  let g = T.grad f x in
  check_scalar ~eps "grad(cos(x)) at x=0" 0.0 (scalar_value g)

let test_grad_tanh () =
  let x = T.scalar ctx T.float32 0.5 in
  let f x = T.tanh x in
  let actual_grad = T.grad f x in

  (* tanh'(x) = 1 - tanh²(x) = 1 - tanh²(0.5) ≈ 0.786 *)
  let tanh_val = T.tanh x |> scalar_value in
  let expected_grad_val = 1.0 -. (tanh_val *. tanh_val) in
  check_scalar ~eps:1e-4 "tanh gradient" expected_grad_val
    (scalar_value actual_grad)

let test_grad_relu () =
  let x = T.create ctx T.float32 [| 5 |] [| -2.; -1.; 0.; 1.; 2. |] in
  let f x = T.sum (T.relu x) in
  let grad = T.grad f x in
  let expected = T.create ctx T.float32 [| 5 |] [| 0.; 0.; 0.; 1.; 1. |] in
  check_rune "relu gradient" expected grad

let test_grad_sqrt () =
  let x = T.scalar ctx T.float32 4.0 in
  let f x = T.sqrt x in
  let g = T.grad f x in
  check_scalar ~eps "grad(sqrt(x)) at x=4" 0.25 (scalar_value g)

(* Reduction operation gradients *)
let test_grad_sum () =
  let x = T.create ctx T.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let f x = T.sum x in
  let grad = T.grad f x in
  let expected = T.ones ctx T.float32 [| 2; 2 |] in
  check_rune "sum gradient" expected grad

let test_grad_sum_axis () =
  let x = T.create ctx T.float32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let f x = T.sum (T.sum x ~axes:[| 1 |]) in
  let grad = T.grad f x in
  let expected = T.ones ctx T.float32 [| 2; 3 |] in
  check_rune "sum with axis gradient" expected grad

let test_grad_mean () =
  let x = T.create ctx T.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let f x = T.mean x in
  let grad = T.grad f x in
  let expected = T.full ctx T.float32 [| 2; 2 |] 0.25 in
  check_rune "mean gradient" expected grad

let test_grad_max () =
  let x = T.create ctx T.float32 [| 2; 2 |] [| 1.; 3.; 2.; 4. |] in
  let f x = T.max x in
  let grad = T.grad f x in
  let expected = T.create ctx T.float32 [| 2; 2 |] [| 0.; 0.; 0.; 1. |] in
  check_rune "max gradient" expected grad

(* Broadcasting and shape manipulation gradients *)
let test_grad_broadcast () =
  let x = T.create ctx T.float32 [| 1; 3 |] [| 1.; 2.; 3. |] in
  let y = T.create ctx T.float32 [| 2; 1 |] [| 4.; 5. |] in

  let f_x x = T.sum (T.mul x y) in
  let f_y y = T.sum (T.mul x y) in

  let grad_x = T.grad f_x x in
  let grad_y = T.grad f_y y in

  let expected_x = T.create ctx T.float32 [| 1; 3 |] [| 9.; 9.; 9. |] in
  (* sum of y values *)
  let expected_y = T.create ctx T.float32 [| 2; 1 |] [| 6.; 6. |] in
  (* sum of x values *)

  check_rune "broadcast grad wrt x" expected_x grad_x;
  check_rune "broadcast grad wrt y" expected_y grad_y

let test_grad_squeeze () =
  let x = T.create ctx T.float32 [| 1; 3; 1 |] [| 1.; 2.; 3. |] in
  let f x = T.sum (T.squeeze x) in
  let grad = T.grad f x in
  let expected = T.ones ctx T.float32 [| 1; 3; 1 |] in
  check_rune "squeeze gradient" expected grad

let test_grad_reshape () =
  let x = T.create ctx T.float32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let f x = T.sum (T.reshape [| 3; 2 |] x) in
  let grad = T.grad f x in
  let expected = T.ones ctx T.float32 [| 2; 3 |] in
  check_rune "reshape gradient" expected grad

let test_grad_transpose () =
  let x = T.create ctx T.float32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let f x = T.sum (T.transpose x) in
  let grad = T.grad f x in
  let expected = T.ones ctx T.float32 [| 2; 3 |] in
  check_rune "transpose gradient" expected grad

(* Neural network operation gradients *)
let test_grad_matmul () =
  let a = T.create ctx T.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b =
    T.create ctx T.float32 [| 3; 2 |] [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6 |]
  in

  let f_a a = T.sum (T.matmul a b) in
  let f_b b = T.sum (T.matmul a b) in

  let grad_a = T.grad f_a a in
  let grad_b = T.grad f_b b in

  (* Expected gradients computed by hand or using NumPy *)
  let expected_a =
    T.create ctx T.float32 [| 2; 3 |] [| 0.3; 0.7; 1.1; 0.3; 0.7; 1.1 |]
  in
  let expected_b =
    T.create ctx T.float32 [| 3; 2 |] [| 5.; 5.; 7.; 7.; 9.; 9. |]
  in

  check_rune ~eps "matmul grad wrt a" expected_a grad_a;
  check_rune ~eps "matmul grad wrt b" expected_b grad_b

let test_grad_softmax () =
  (* For softmax, just test the gradient of sum which should be zero due to
     normalization *)
  let x = T.create ctx T.float32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let f x = T.sum (T.softmax x ~axes:[| 0 |]) in
  let actual_grad = T.grad f x in

  (* The sum of softmax is always 1, so gradient should be zero *)
  let expected = T.zeros ctx T.float32 [| 3 |] in
  check_rune ~eps:1e-4 "softmax gradient of sum" expected actual_grad

let test_grad_cross_entropy () =
  let logits =
    T.create ctx T.float32 [| 2; 3 |] [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6 |]
  in
  let targets =
    T.create ctx T.float32 [| 2; 3 |] [| 1.; 0.; 0.; 0.; 0.; 1. |]
  in

  (* Cross-entropy: -sum(targets * log(softmax(logits))) *)
  let f logits =
    let probs = T.softmax logits ~axes:[| 1 |] in
    let log_probs = T.log probs in
    T.neg (T.sum (T.mul targets log_probs))
  in

  let actual_grad = T.grad f logits in

  (* Expected gradient is softmax(logits) - targets *)
  let probs = T.softmax logits ~axes:[| 1 |] in
  let expected_grad = T.sub probs targets in
  check_rune ~eps:1e-5 "cross-entropy gradient" expected_grad actual_grad

(* Multiple variables and composition *)
let test_grad_multi_var () =
  let x = T.scalar ctx T.float32 2.0 in
  let y = T.scalar ctx T.float32 3.0 in

  let f_x x = T.add (T.mul x x) (T.mul y y) in
  (* z = x² + y² *)
  let f_y y = T.add (T.mul x x) (T.mul y y) in

  let grad_x = T.grad f_x x in
  let grad_y = T.grad f_y y in

  check_scalar ~eps "grad(x² + y²) wrt x at x=2" 4.0 (scalar_value grad_x);
  check_scalar ~eps "grad(x² + y²) wrt y at y=3" 6.0 (scalar_value grad_y)

let test_grad_chain_rule () =
  let x = T.scalar ctx T.float32 2.0 in
  let f x =
    let y = T.mul x x in
    (* y = x² *)
    T.mul y y (* z = y² = x⁴ *)
  in
  let g = T.grad f x in
  (* dz/dx = 4x³ = 32 at x=2 *)
  check_scalar ~eps "chain rule: grad(x⁴) at x=2" 32.0 (scalar_value g)

(* Higher-order derivatives *)
let test_grad_second_order () =
  let x = T.scalar ctx T.float32 2.0 in
  let f x = T.mul x (T.mul x x) in
  (* y = x³ *)
  let grad_f x = T.grad f x in
  let second_deriv = T.grad grad_f x in
  check_scalar ~eps "second derivative of x³ at x=2" 12.0
    (scalar_value second_deriv)

(* Complex computational graphs *)
let test_grad_diamond () =
  (* x -> a, b -> c pattern *)
  let x = T.scalar ctx T.float32 2.0 in
  let f x =
    let a = T.mul x (T.scalar ctx T.float32 3.0) in
    let b = T.add x (T.scalar ctx T.float32 1.0) in
    T.mul a b
  in
  let grad = T.grad f x in
  (* c = (3x)(x+1) = 3x² + 3x, dc/dx = 6x + 3 = 15 at x=2 *)
  check_scalar ~eps "diamond pattern gradient" 15.0 (scalar_value grad)

let test_grad_shared_subexpression () =
  let x = T.scalar ctx T.float32 2.0 in
  let f x =
    let a = T.mul x x in
    T.add a a (* Reuse a twice *)
  in
  let grad = T.grad f x in
  (* b = 2x², db/dx = 4x = 8 at x=2 *)
  check_scalar ~eps "shared subexpression gradient" 8.0 (scalar_value grad)

(* Value and gradient *)
let test_value_and_grad () =
  let x = T.scalar ctx T.float32 3.0 in
  let f x = T.mul x x in
  let value, grad = T.value_and_grad f x in
  check_scalar ~eps "value_and_grad value" 9.0 (scalar_value value);
  check_scalar ~eps "value_and_grad grad" 6.0 (scalar_value grad)

(* Edge cases *)
let test_grad_zero () =
  let x = T.scalar ctx T.float32 0.0 in
  let f x = T.mul x x in
  let g = T.grad f x in
  check_scalar ~eps "grad(x²) at x=0" 0.0 (scalar_value g)

(* TODO: Implement when detach and no_grad are available let test_grad_detached
   () = let x = T.scalar ctx T.float32 2.0 in let f x = let y = T.detach x in
   T.mul y x in let g = T.grad f x in (* Since y is detached, only the direct x
   contributes to gradient *) check_scalar ~eps "grad with detached variable"
   2.0 (scalar_value g)

   let test_grad_no_grad_scope () = let x = T.scalar ctx T.float32 2.0 in let f
   x = let y = T.no_grad (fun () -> T.mul x x) in T.add y x in let g = T.grad f
   x in (* Only the direct x in z = y + x contributes since y was computed in
   no_grad *) check_scalar ~eps "grad with no_grad scope" 1.0 (scalar_value
   g) *)

(* Test suite *)
let () =
  run "Rune Autodiff Tests"
    [
      ( "basic gradients",
        [
          test_case "simple x²" `Quick test_grad_simple;
          test_case "constant addition" `Quick test_grad_constant;
          test_case "linear function" `Quick test_grad_linear;
          test_case "polynomial" `Quick test_grad_polynomial;
        ] );
      ( "unary operations",
        [
          test_case "negation" `Quick test_grad_neg;
          test_case "exponential" `Quick test_grad_exp;
          test_case "logarithm" `Quick test_grad_log;
          test_case "sine" `Quick test_grad_sin;
          test_case "cosine" `Quick test_grad_cos;
          test_case "tanh" `Quick test_grad_tanh;
          test_case "relu" `Quick test_grad_relu;
          test_case "sqrt" `Quick test_grad_sqrt;
        ] );
      ( "reduction operations",
        [
          test_case "sum" `Quick test_grad_sum;
          test_case "sum with axis" `Quick test_grad_sum_axis;
          test_case "mean" `Quick test_grad_mean;
          test_case "max" `Quick test_grad_max;
        ] );
      ( "broadcasting and reshaping",
        [
          test_case "broadcast" `Quick test_grad_broadcast;
          test_case "squeeze" `Quick test_grad_squeeze;
          test_case "reshape" `Quick test_grad_reshape;
          test_case "transpose" `Quick test_grad_transpose;
        ] );
      ( "neural network operations",
        [
          test_case "matmul" `Quick test_grad_matmul;
          test_case "softmax" `Quick test_grad_softmax;
          test_case "cross-entropy" `Quick test_grad_cross_entropy;
        ] );
      ( "composition and multi-variable",
        [
          test_case "multiple variables" `Quick test_grad_multi_var;
          test_case "chain rule" `Quick test_grad_chain_rule;
          test_case "diamond pattern" `Quick test_grad_diamond;
          test_case "shared subexpression" `Quick test_grad_shared_subexpression;
        ] );
      ( "advanced features",
        [
          test_case "second-order derivative" `Quick test_grad_second_order;
          test_case "value_and_grad" `Quick test_value_and_grad;
        ] );
      ( "edge cases",
        [
          test_case "gradient at zero" `Quick test_grad_zero;
          (* test_case "detached tensors" `Quick test_grad_detached; *)
          (* test_case "no_grad scope" `Quick test_grad_no_grad_scope; *)
        ] );
    ]
