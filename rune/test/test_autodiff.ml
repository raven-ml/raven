open Alcotest
open Test_rune_support
module T = Rune

let ctx = T.c
let eps = 1e-6

(* ───── basic gradients ───── *)

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
  check_scalar ~eps "grad(x³ + 2x² + x) at x=2" 21.0 (scalar_value g)

(* ───── unary operations ───── *)

let test_grad_unary_ops () =
  (* Test multiple unary ops in one test to reduce redundancy *)
  let x = T.scalar ctx T.float32 1.0 in

  (* exp: d/dx e^x = e^x *)
  let grad_exp = T.grad T.exp (T.scalar ctx T.float32 0.0) in
  check_scalar ~eps "grad(exp(x)) at x=0" 1.0 (scalar_value grad_exp);

  (* log: d/dx ln(x) = 1/x *)
  let grad_log = T.grad T.log (T.scalar ctx T.float32 2.0) in
  check_scalar ~eps "grad(log(x)) at x=2" 0.5 (scalar_value grad_log);

  (* sin/cos at x=0 *)
  let grad_sin = T.grad T.sin (T.scalar ctx T.float32 0.0) in
  let grad_cos = T.grad T.cos (T.scalar ctx T.float32 0.0) in
  check_scalar ~eps "grad(sin(x)) at x=0" 1.0 (scalar_value grad_sin);
  check_scalar ~eps "grad(cos(x)) at x=0" 0.0 (scalar_value grad_cos);

  (* sqrt: d/dx √x = 1/(2√x) *)
  let grad_sqrt = T.grad T.sqrt (T.scalar ctx T.float32 4.0) in
  check_scalar ~eps "grad(sqrt(x)) at x=4" 0.25 (scalar_value grad_sqrt);

  (* neg: d/dx (-x) = -1 *)
  let grad_neg = T.grad T.neg x in
  check_scalar ~eps "grad(-x)" (-1.0) (scalar_value grad_neg)

let test_grad_activation_functions () =
  (* Group activation functions together *)
  let x = T.create ctx T.float32 [| 4 |] [| -2.; -1.; 1.; 2. |] in

  (* ReLU gradient *)
  let f_relu x = T.sum (T.relu x) in
  let grad_relu = T.grad f_relu x in
  let expected_relu = T.create ctx T.float32 [| 4 |] [| 0.; 0.; 1.; 1. |] in
  check_rune ~eps "relu gradient" expected_relu grad_relu;

  (* Tanh gradient *)
  let x_tanh = T.scalar ctx T.float32 0.5 in
  let grad_tanh = T.grad T.tanh x_tanh in
  let tanh_val = T.tanh x_tanh |> scalar_value in
  let expected_tanh = 1.0 -. (tanh_val *. tanh_val) in
  check_scalar ~eps:1e-4 "tanh gradient" expected_tanh (scalar_value grad_tanh);

  (* Abs gradient *)
  let f_abs x = T.sum (T.abs x) in
  let grad_abs = T.grad f_abs x in
  let expected_abs = T.create ctx T.float32 [| 4 |] [| -1.; -1.; 1.; 1. |] in
  check_rune ~eps "abs gradient" expected_abs grad_abs

(* ───── reduction operations ───── *)

let test_grad_reductions () =
  let x = T.create ctx T.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in

  (* Sum gradient *)
  let grad_sum = T.grad T.sum x in
  let expected_sum = T.ones ctx T.float32 [| 2; 2 |] in
  check_rune ~eps "sum gradient" expected_sum grad_sum;

  (* Mean gradient *)
  let grad_mean = T.grad T.mean x in
  let expected_mean = T.full ctx T.float32 [| 2; 2 |] 0.25 in
  check_rune ~eps "mean gradient" expected_mean grad_mean;

  (* Max gradient *)
  let x_max = T.create ctx T.float32 [| 2; 2 |] [| 1.; 3.; 2.; 4. |] in
  let grad_max = T.grad T.max x_max in
  let expected_max = T.create ctx T.float32 [| 2; 2 |] [| 0.; 0.; 0.; 1. |] in
  check_rune ~eps "max gradient" expected_max grad_max;

  (* Sum with axis *)
  let x_axis = T.create ctx T.float32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let f_axis x = T.sum (T.sum x ~axes:[| 1 |]) in
  let grad_axis = T.grad f_axis x_axis in
  let expected_axis = T.ones ctx T.float32 [| 2; 3 |] in
  check_rune ~eps "sum with axis gradient" expected_axis grad_axis

(* ───── broadcasting gradients ───── *)

let test_grad_broadcast_binary_ops () =
  (* Comprehensive test for all binary ops with broadcasting *)
  let x = T.create ctx T.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let bias = T.create ctx T.float32 [| 3 |] [| 0.1; 0.2; 0.3 |] in
  let scalar = T.scalar ctx T.float32 2.0 in

  (* Addition: [2,3] + [3] *)
  let _, grads_add =
    T.value_and_grads
      (fun inputs ->
        match inputs with
        | [ a; b ] -> T.sum (T.add a b)
        | _ -> failwith "Expected 2 inputs")
      [ x; bias ]
  in
  let grad_bias_add = List.nth grads_add 1 in
  check_rune ~eps "add broadcast: bias gradient"
    (T.full ctx T.float32 [| 3 |] 2.0)
    grad_bias_add;

  (* Multiplication: [2,3] * [3] *)
  let scale = T.create ctx T.float32 [| 3 |] [| 2.; 3.; 4. |] in
  let _, grads_mul =
    T.value_and_grads
      (fun inputs ->
        match inputs with
        | [ a; s ] -> T.sum (T.mul a s)
        | _ -> failwith "Expected 2 inputs")
      [ x; scale ]
  in
  let grad_scale_mul = List.nth grads_mul 1 in
  let expected_mul = T.create ctx T.float32 [| 3 |] [| 5.; 7.; 9. |] in
  check_rune ~eps "mul broadcast: scale gradient" expected_mul grad_scale_mul;

  (* Scalar broadcasting *)
  let grad_scalar_add = T.grad (fun s -> T.sum (T.add x s)) scalar in
  check_scalar ~eps "scalar add broadcast" 6.0 (scalar_value grad_scalar_add);

  let grad_scalar_mul = T.grad (fun s -> T.sum (T.mul x s)) scalar in
  check_scalar ~eps "scalar mul broadcast" 21.0 (scalar_value grad_scalar_mul)

let test_grad_broadcast_special_cases () =
  (* Test expand, where, and complex broadcasting patterns *)

  (* Expand: scalar to vector *)
  let scalar = T.scalar ctx T.float32 5.0 in
  let f_expand s = T.sum (T.expand [| 3 |] s) in
  let grad_expand = T.grad f_expand scalar in
  check_scalar ~eps "expand scalar gradient" 3.0 (scalar_value grad_expand);

  (* Expand: vector to matrix (bias pattern) *)
  let vec = T.create ctx T.float32 [| 3 |] [| 10.; 20.; 30. |] in
  let f_expand_vec v = T.sum (T.expand [| 2; 3 |] v) in
  let grad_expand_vec = T.grad f_expand_vec vec in
  check_rune ~eps "expand vector gradient"
    (T.full ctx T.float32 [| 3 |] 2.0)
    grad_expand_vec;

  (* Where with broadcasting *)
  let cond = T.create ctx T.uint8 [| 2; 3 |] [| 1; 0; 1; 0; 1; 0 |] in
  let x = T.create ctx T.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y_scalar = T.scalar ctx T.float32 10.0 in
  let _, grads_where =
    T.value_and_grads
      (fun inputs ->
        match inputs with
        | [ a; b ] -> T.sum (T.where cond a b)
        | _ -> failwith "Expected 2 inputs")
      [ x; y_scalar ]
  in
  let grad_y_where = List.nth grads_where 1 in
  check_scalar ~eps "where scalar broadcast gradient" 3.0
    (scalar_value grad_y_where)

(* ───── shape manipulation ───── *)

let test_grad_shape_ops () =
  let x = T.create ctx T.float32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in

  (* Reshape gradient *)
  let f_reshape x = T.sum (T.reshape [| 3; 2 |] x) in
  let grad_reshape = T.grad f_reshape x in
  check_rune ~eps "reshape gradient" (T.ones_like x) grad_reshape;

  (* Transpose gradient *)
  let grad_transpose = T.grad (fun x -> T.sum (T.transpose x)) x in
  check_rune ~eps "transpose gradient" (T.ones_like x) grad_transpose;

  (* Squeeze gradient *)
  let x_squeeze = T.create ctx T.float32 [| 1; 3; 1 |] [| 1.; 2.; 3. |] in
  let grad_squeeze = T.grad (fun x -> T.sum (T.squeeze x)) x_squeeze in
  check_rune ~eps "squeeze gradient" (T.ones_like x_squeeze) grad_squeeze

(* ───── neural network operations ───── *)

let test_grad_matmul () =
  let a = T.create ctx T.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b =
    T.create ctx T.float32 [| 3; 2 |] [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6 |]
  in

  let f_a a = T.sum (T.matmul a b) in
  let f_b b = T.sum (T.matmul a b) in

  let grad_a = T.grad f_a a in
  let grad_b = T.grad f_b b in

  let expected_a =
    T.create ctx T.float32 [| 2; 3 |] [| 0.3; 0.7; 1.1; 0.3; 0.7; 1.1 |]
  in
  let expected_b =
    T.create ctx T.float32 [| 3; 2 |] [| 5.; 5.; 7.; 7.; 9.; 9. |]
  in

  check_rune ~eps "matmul grad wrt a" expected_a grad_a;
  check_rune ~eps "matmul grad wrt b" expected_b grad_b

let test_grad_linear_layer () =
  (* Combined matmul + bias pattern *)
  let batch = 4 in
  let in_dim = 2 in
  let out_dim = 8 in

  let x =
    T.create ctx T.float32 [| batch; in_dim |]
      (Array.init (batch * in_dim) float_of_int)
  in
  let w =
    T.create ctx T.float32 [| in_dim; out_dim |]
      (Array.init (in_dim * out_dim) (fun i -> float_of_int i *. 0.1))
  in
  let b =
    T.create ctx T.float32 [| out_dim |]
      (Array.init out_dim (fun i -> float_of_int i *. 0.01))
  in

  let linear w b x = T.add (T.matmul x w) b in
  let loss x w b = T.sum (linear w b x) in

  let grad_b = T.grad (fun b -> loss x w b) b in
  let grad_w = T.grad (fun w -> loss x w b) w in

  check_rune ~eps "linear layer bias gradient"
    (T.full ctx T.float32 [| out_dim |] (float_of_int batch))
    grad_b;

  let ones_out = T.ones ctx T.float32 [| batch; out_dim |] in
  let expected_w = T.matmul (T.transpose x) ones_out in
  check_rune ~eps "linear layer weight gradient" expected_w grad_w

let test_grad_loss_functions () =
  (* Softmax + Cross-entropy *)
  let logits =
    T.create ctx T.float32 [| 2; 3 |] [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6 |]
  in
  let targets =
    T.create ctx T.float32 [| 2; 3 |] [| 1.; 0.; 0.; 0.; 0.; 1. |]
  in

  let f_ce logits =
    let probs = T.softmax logits ~axes:[| 1 |] in
    let log_probs = T.log probs in
    T.neg (T.sum (T.mul targets log_probs))
  in

  let grad_ce = T.grad f_ce logits in
  let expected_ce = T.sub (T.softmax logits ~axes:[| 1 |]) targets in
  check_rune ~eps:1e-5 "cross-entropy gradient" expected_ce grad_ce;

  (* Binary cross-entropy with sigmoid *)
  let logits_bce = T.create ctx T.float32 [| 4; 1 |] [| -1.; 0.5; 0.5; -1. |] in
  let targets_bce = T.create ctx T.float32 [| 4; 1 |] [| 0.; 1.; 1.; 0. |] in

  let f_bce logits =
    let sigmoid_logits = T.sigmoid logits in
    let one = T.ones_like targets_bce in
    let one_minus_targets = T.sub one targets_bce in
    let one_minus_sigmoid = T.sub one sigmoid_logits in
    let term1 = T.mul targets_bce (T.log sigmoid_logits) in
    let term2 = T.mul one_minus_targets (T.log one_minus_sigmoid) in
    T.mean (T.neg (T.add term1 term2))
  in

  let grad_bce = T.grad f_bce logits_bce in
  let diff = T.sub (T.sigmoid logits_bce) targets_bce in
  let n = float_of_int (Array.fold_left ( * ) 1 (T.shape logits_bce)) in
  let expected_bce = T.div diff (T.scalar ctx T.float32 n) in
  check_rune ~eps:1e-5 "sigmoid BCE gradient" expected_bce grad_bce

let test_grad_pooling () =
  let x =
    T.create ctx T.float32 [| 1; 1; 4; 4 |]
      [|
        1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12.; 13.; 14.; 15.; 16.;
      |]
  in

  let f x = T.sum (T.avg_pool2d x ~kernel_size:(2, 2) ~stride:(2, 2)) in
  let grad = T.grad f x in

  let expected = T.full ctx T.float32 [| 1; 1; 4; 4 |] 0.25 in
  check_rune ~eps "avg_pool2d gradient" expected grad

let test_grad_conv2d () =
  (* Test 1: Simple conv2d gradient with Valid padding *)
  let x =
    T.create ctx T.float32 [| 1; 1; 4; 4 |]
      [|
        1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12.; 13.; 14.; 15.; 16.;
      |]
  in

  let kernel =
    T.create ctx T.float32 [| 1; 1; 2; 2 |] [| 1.; 0.; 0.; 1. |]
    (* Identity-like kernel *)
  in

  (* Test gradient w.r.t input *)
  let f_x x = T.sum (T.convolve2d x kernel ~padding_mode:`Valid) in
  let grad_x = T.grad f_x x in

  (* With Valid padding, only inner elements get gradients *)
  let expected_x =
    T.create ctx T.float32 [| 1; 1; 4; 4 |]
      [| 1.; 1.; 1.; 0.; 1.; 2.; 2.; 1.; 1.; 2.; 2.; 1.; 0.; 1.; 1.; 1. |]
  in
  check_rune ~eps "conv2d gradient w.r.t input (Valid)" expected_x grad_x;

  (* Test gradient w.r.t kernel *)
  let f_k k = T.sum (T.convolve2d x k ~padding_mode:`Valid) in
  let grad_k = T.grad f_k kernel in

  (* Expected gradient: sum of all valid windows *)
  (* Note: The kernel gradient appears in reverse order due to correlation vs convolution *)
  let expected_k =
    T.create ctx T.float32 [| 1; 1; 2; 2 |] [| 99.; 90.; 63.; 54. |]
  in
  check_rune ~eps "conv2d gradient w.r.t kernel" expected_k grad_k;

  (* Test 2: Conv2d with Same padding *)
  let f_same x = T.sum (T.convolve2d x kernel ~padding_mode:`Same) in
  let grad_same = T.grad f_same x in

  (* With Same padding, output has same size as input *)
  (* Expected gradients depend on how kernel overlaps at each position *)
  (* Note: Due to correlation vs convolution, the pattern appears flipped *)
  let expected_same =
    T.create ctx T.float32 [| 1; 1; 4; 4 |]
      [| 2.; 2.; 2.; 1.; 2.; 2.; 2.; 1.; 2.; 2.; 2.; 1.; 1.; 1.; 1.; 1. |]
  in
  check_rune ~eps "conv2d gradient with Same padding" expected_same grad_same

let test_grad_avg_pool_overlapping () =
  let eps = 1e-4 in
  let x =
    T.create ctx T.float32 [| 1; 1; 4; 4 |]
      [|
        1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12.; 13.; 14.; 15.; 16.;
      |]
  in

  (* THE ONLY CHANGE IS THE STRIDE *)
  let f x = T.sum (T.avg_pool2d x ~kernel_size:(2, 2) ~stride:(1, 1)) in
  let grad = T.grad f x in

  (* With overlapping windows, the gradients accumulate. Each of the 9 output
     elements contributes 0.25 to its 4 input pixels. *)
  let expected =
    T.create ctx T.float32 [| 1; 1; 4; 4 |]
      [|
        0.25;
        0.5;
        0.5;
        0.25;
        0.5;
        1.0;
        1.0;
        0.5;
        0.5;
        1.0;
        1.0;
        0.5;
        0.25;
        0.5;
        0.5;
        0.25;
      |]
  in

  (* Your code will produce a result 4x larger than this. *)
  check_rune ~eps "avg_pool2d gradient with overlap" expected grad

(* ───── composition and higher-order ───── *)

let test_grad_composition () =
  let x = T.scalar ctx T.float32 2.0 in
  let y = T.scalar ctx T.float32 3.0 in

  (* Multi-variable gradient *)
  let f_x x = T.add (T.mul x x) (T.mul y y) in
  let f_y y = T.add (T.mul x x) (T.mul y y) in
  let grad_x = T.grad f_x x in
  let grad_y = T.grad f_y y in
  check_scalar ~eps "multi-var grad wrt x" 4.0 (scalar_value grad_x);
  check_scalar ~eps "multi-var grad wrt y" 6.0 (scalar_value grad_y);

  (* Chain rule *)
  let f_chain x =
    let y = T.mul x x in
    T.mul y y
  in
  let grad_chain = T.grad f_chain x in
  check_scalar ~eps "chain rule: grad(x⁴) at x=2" 32.0 (scalar_value grad_chain);

  (* Shared subexpression *)
  let f_shared x =
    let a = T.mul x x in
    T.add a a
  in
  let grad_shared = T.grad f_shared x in
  check_scalar ~eps "shared subexpression gradient" 8.0
    (scalar_value grad_shared);

  (* Second-order derivative *)
  let f x = T.mul x (T.mul x x) in
  let grad_f x = T.grad f x in
  let second_deriv = T.grad grad_f x in
  check_scalar ~eps "second derivative of x³ at x=2" 12.0
    (scalar_value second_deriv)

(* ───── api functions ───── *)

let test_grad_api_functions () =
  let x = T.scalar ctx T.float32 2.0 in
  let y = T.scalar ctx T.float32 3.0 in
  let z = T.scalar ctx T.float32 1.0 in

  (* value_and_grad *)
  let f x = T.mul x x in
  let value, grad = T.value_and_grad f x in
  check_scalar ~eps "value_and_grad value" 4.0 (scalar_value value);
  check_scalar ~eps "value_and_grad grad" 4.0 (scalar_value grad);

  (* grads and value_and_grads *)
  let f_multi inputs =
    match inputs with
    | [ a; b; c ] -> T.mul c (T.add (T.mul a a) (T.mul b b))
    | _ -> failwith "Expected 3 inputs"
  in

  let value, grads = T.value_and_grads f_multi [ x; y; z ] in
  check_scalar ~eps "value_and_grads value" 13.0 (scalar_value value);
  check_scalar ~eps "value_and_grads grad x" 4.0
    (scalar_value (List.nth grads 0));
  check_scalar ~eps "value_and_grads grad y" 6.0
    (scalar_value (List.nth grads 1));
  check_scalar ~eps "value_and_grads grad z" 13.0
    (scalar_value (List.nth grads 2))

(* ───── edge cases ───── *)

let test_grad_edge_cases () =
  (* Gradient at zero *)
  let x = T.scalar ctx T.float32 0.0 in
  let f x = T.mul x x in
  let g = T.grad f x in
  check_scalar ~eps "grad(x²) at x=0" 0.0 (scalar_value g)

(* ───── forward mode AD (JVP) tests ───── *)

let test_jvp_simple () =
  (* Test basic JVP computation *)
  let x = T.scalar ctx T.float32 2.0 in
  let v = T.scalar ctx T.float32 1.0 in
  
  (* f(x) = x² *)
  let f x = T.mul x x in
  let primal, tangent = T.jvp f x v in
  
  check_scalar ~eps "jvp(x²) primal at x=2" 4.0 (scalar_value primal);
  check_scalar ~eps "jvp(x²) tangent at x=2" 4.0 (scalar_value tangent);
  
  (* Test with different tangent vector *)
  let v2 = T.scalar ctx T.float32 3.0 in
  let _, tangent2 = T.jvp f x v2 in
  check_scalar ~eps "jvp(x²) tangent with v=3" 12.0 (scalar_value tangent2)

let test_jvp_unary_ops () =
  let x = T.scalar ctx T.float32 1.0 in
  let v = T.scalar ctx T.float32 1.0 in
  
  (* sin/cos *)
  let primal_sin, tangent_sin = T.jvp T.sin (T.scalar ctx T.float32 0.0) v in
  check_scalar ~eps "jvp(sin) primal at x=0" 0.0 (scalar_value primal_sin);
  check_scalar ~eps "jvp(sin) tangent at x=0" 1.0 (scalar_value tangent_sin);
  
  (* exp *)
  let primal_exp, tangent_exp = T.jvp T.exp (T.scalar ctx T.float32 0.0) v in
  check_scalar ~eps "jvp(exp) primal at x=0" 1.0 (scalar_value primal_exp);
  check_scalar ~eps "jvp(exp) tangent at x=0" 1.0 (scalar_value tangent_exp);
  
  (* sqrt *)
  let x_sqrt = T.scalar ctx T.float32 4.0 in
  let primal_sqrt, tangent_sqrt = T.jvp T.sqrt x_sqrt v in
  check_scalar ~eps "jvp(sqrt) primal at x=4" 2.0 (scalar_value primal_sqrt);
  check_scalar ~eps "jvp(sqrt) tangent at x=4" 0.25 (scalar_value tangent_sqrt);
  
  (* neg *)
  let primal_neg, tangent_neg = T.jvp T.neg x v in
  check_scalar ~eps "jvp(neg) primal" (-1.0) (scalar_value primal_neg);
  check_scalar ~eps "jvp(neg) tangent" (-1.0) (scalar_value tangent_neg)

let test_jvp_binary_ops () =
  let x = T.scalar ctx T.float32 3.0 in
  let y = T.scalar ctx T.float32 2.0 in
  let vx = T.scalar ctx T.float32 1.0 in
  let vy = T.scalar ctx T.float32 0.5 in
  
  (* Test addition with multiple inputs *)
  let f inputs =
    match inputs with
    | [a; b] -> T.add a b
    | _ -> failwith "Expected 2 inputs"
  in
  let primal_add, tangent_add = T.jvps f [x; y] [vx; vy] in
  check_scalar ~eps "jvp(add) primal" 5.0 (scalar_value primal_add);
  check_scalar ~eps "jvp(add) tangent" 1.5 (scalar_value tangent_add);
  
  (* Test multiplication *)
  let f_mul inputs =
    match inputs with
    | [a; b] -> T.mul a b
    | _ -> failwith "Expected 2 inputs"
  in
  let primal_mul, tangent_mul = T.jvps f_mul [x; y] [vx; vy] in
  check_scalar ~eps "jvp(mul) primal" 6.0 (scalar_value primal_mul);
  (* d(xy) = y*dx + x*dy = 2*1 + 3*0.5 = 3.5 *)
  check_scalar ~eps "jvp(mul) tangent" 3.5 (scalar_value tangent_mul);
  
  (* Test division *)
  let f_div inputs =
    match inputs with
    | [a; b] -> T.div a b
    | _ -> failwith "Expected 2 inputs"
  in
  let primal_div, tangent_div = T.jvps f_div [x; y] [vx; vy] in
  check_scalar ~eps "jvp(div) primal" 1.5 (scalar_value primal_div);
  (* d(x/y) = dx/y - x*dy/y² = 1/2 - 3*0.5/4 = 0.5 - 0.375 = 0.125 *)
  check_scalar ~eps "jvp(div) tangent" 0.125 (scalar_value tangent_div)

let test_jvp_shape_ops () =
  let x = T.create ctx T.float32 [|2; 3|] [|1.; 2.; 3.; 4.; 5.; 6.|] in
  let v = T.ones_like x in
  
  (* Reshape *)
  let f_reshape x = T.reshape [|3; 2|] x in
  let primal_reshape, tangent_reshape = T.jvp f_reshape x v in
  check_rune ~eps "jvp(reshape) primal" (T.reshape [|3; 2|] x) primal_reshape;
  check_rune ~eps "jvp(reshape) tangent" (T.reshape [|3; 2|] v) tangent_reshape;
  
  (* Sum reduction *)
  let f_sum x = T.sum x in
  let primal_sum, tangent_sum = T.jvp f_sum x v in
  check_scalar ~eps "jvp(sum) primal" 21.0 (scalar_value primal_sum);
  check_scalar ~eps "jvp(sum) tangent" 6.0 (scalar_value tangent_sum);
  
  (* Transpose *)
  let f_transpose x = T.transpose x in
  let primal_transpose, tangent_transpose = T.jvp f_transpose x v in
  check_rune ~eps "jvp(transpose) primal" (T.transpose x) primal_transpose;
  check_rune ~eps "jvp(transpose) tangent" (T.transpose v) tangent_transpose

let test_jvp_matmul () =
  let a = T.create ctx T.float32 [|2; 3|] [|1.; 2.; 3.; 4.; 5.; 6.|] in
  let b = T.create ctx T.float32 [|3; 2|] [|0.1; 0.2; 0.3; 0.4; 0.5; 0.6|] in
  let va = T.ones_like a in
  let vb = T.zeros_like b in
  
  (* Test matmul with tangent only in first argument *)
  let f inputs =
    match inputs with
    | [a; b] -> T.matmul a b
    | _ -> failwith "Expected 2 inputs"
  in
  let primal, tangent = T.jvps f [a; b] [va; vb] in
  
  (* Expected tangent: va @ b + a @ vb = ones @ b + a @ zeros = ones @ b *)
  let expected_tangent = T.matmul va b in
  check_rune ~eps "jvp(matmul) primal" (T.matmul a b) primal;
  check_rune ~eps "jvp(matmul) tangent" expected_tangent tangent;
  
  (* Test with tangent in both arguments *)
  let vb2 = T.ones_like b in
  let _, tangent2 = T.jvps f [a; b] [va; vb2] in
  let expected_tangent2 = T.add (T.matmul va b) (T.matmul a vb2) in
  check_rune ~eps "jvp(matmul) tangent both" expected_tangent2 tangent2

let test_jvp_multiple_inputs () =
  let x = T.scalar ctx T.float32 2.0 in
  let y = T.scalar ctx T.float32 3.0 in
  let z = T.scalar ctx T.float32 4.0 in
  let vx = T.scalar ctx T.float32 1.0 in
  let vy = T.scalar ctx T.float32 0.5 in
  let vz = T.scalar ctx T.float32 0.25 in
  
  (* f(x,y,z) = x*y + y*z + x*z *)
  let f inputs =
    match inputs with
    | [x; y; z] ->
        let xy = T.mul x y in
        let yz = T.mul y z in
        let xz = T.mul x z in
        T.add (T.add xy yz) xz
    | _ -> failwith "Expected 3 inputs"
  in
  
  let primal, tangent = T.jvps f [x; y; z] [vx; vy; vz] in
  
  (* f(2,3,4) = 6 + 12 + 8 = 26 *)
  check_scalar ~eps "jvp multi-input primal" 26.0 (scalar_value primal);
  
  (* df = (y+z)*dx + (x+z)*dy + (x+y)*dz = 7*1 + 6*0.5 + 5*0.25 = 11.25 *)
  check_scalar ~eps "jvp multi-input tangent" 11.25 (scalar_value tangent)

let test_jvp_with_aux () =
  let x = T.scalar ctx T.float32 3.0 in
  let v = T.scalar ctx T.float32 1.0 in
  
  (* Function that returns both result and auxiliary data *)
  let f_with_aux x =
    let y = T.mul x x in
    let aux = T.shape y in  (* auxiliary output *)
    (y, aux)
  in
  
  let primal, tangent, aux = T.jvp_aux f_with_aux x v in
  
  check_scalar ~eps "jvp_aux primal" 9.0 (scalar_value primal);
  check_scalar ~eps "jvp_aux tangent" 6.0 (scalar_value tangent);
  check (array int) "jvp_aux auxiliary" [||] aux

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
          test_case "unary ops" `Quick test_grad_unary_ops;
          test_case "activation functions" `Quick test_grad_activation_functions;
        ] );
      ( "reduction operations",
        [ test_case "reductions" `Quick test_grad_reductions ] );
      ( "broadcasting",
        [
          test_case "binary ops broadcasting" `Quick
            test_grad_broadcast_binary_ops;
          test_case "special broadcasting cases" `Quick
            test_grad_broadcast_special_cases;
        ] );
      ( "shape manipulation",
        [ test_case "shape operations" `Quick test_grad_shape_ops ] );
      ( "neural network operations",
        [
          test_case "matmul" `Quick test_grad_matmul;
          test_case "linear layer" `Quick test_grad_linear_layer;
          test_case "loss functions" `Quick test_grad_loss_functions;
          test_case "pooling" `Quick test_grad_pooling;
          test_case "conv2d" `Quick test_grad_conv2d;
          test_case "avg_pool2d overlapping" `Quick
            test_grad_avg_pool_overlapping;
        ] );
      ( "composition and higher-order",
        [ test_case "composition patterns" `Quick test_grad_composition ] );
      ( "api functions",
        [ test_case "gradient APIs" `Quick test_grad_api_functions ] );
      ("edge cases", [ test_case "edge cases" `Quick test_grad_edge_cases ]);
      ( "forward mode (jvp)",
        [
          test_case "jvp simple" `Quick test_jvp_simple;
          test_case "jvp unary ops" `Quick test_jvp_unary_ops;
          test_case "jvp binary ops" `Quick test_jvp_binary_ops;
          test_case "jvp shape ops" `Quick test_jvp_shape_ops;
          test_case "jvp matmul" `Quick test_jvp_matmul;
          test_case "jvp multiple inputs" `Quick test_jvp_multiple_inputs;
          test_case "jvp with aux" `Quick test_jvp_with_aux;
        ] );
    ]
