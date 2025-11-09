open Alcotest
open Test_rune_support
module T = Rune

let eps = 1e-6

(* ───── binary operations ───── *)

let test_grad_add () =
  let x = T.scalar T.float32 2.0 in
  let y = T.scalar T.float32 3.0 in
  let grad_x = T.grad (fun x -> T.add x y) x in
  let grad_y = T.grad (fun y -> T.add x y) y in
  check_scalar ~eps "add grad wrt x" 1.0 (scalar_value grad_x);
  check_scalar ~eps "add grad wrt y" 1.0 (scalar_value grad_y)

let test_grad_mul () =
  let x = T.scalar T.float32 2.0 in
  let y = T.scalar T.float32 3.0 in
  let grad_x = T.grad (fun x -> T.mul x y) x in
  let grad_y = T.grad (fun y -> T.mul x y) y in
  check_scalar ~eps "mul grad wrt x" 3.0 (scalar_value grad_x);
  check_scalar ~eps "mul grad wrt y" 2.0 (scalar_value grad_y)

let test_grad_sub () =
  let x = T.scalar T.float32 5.0 in
  let y = T.scalar T.float32 3.0 in
  let grad_x = T.grad (fun x -> T.sub x y) x in
  let grad_y = T.grad (fun y -> T.sub x y) y in
  check_scalar ~eps "sub grad wrt x" 1.0 (scalar_value grad_x);
  check_scalar ~eps "sub grad wrt y" (-1.0) (scalar_value grad_y)

let test_grad_div () =
  let x = T.scalar T.float32 6.0 in
  let y = T.scalar T.float32 2.0 in
  let grad_x = T.grad (fun x -> T.div x y) x in
  let grad_y = T.grad (fun y -> T.div x y) y in
  check_scalar ~eps "div grad wrt x" 0.5 (scalar_value grad_x);
  check_scalar ~eps "div grad wrt y" (-1.5) (scalar_value grad_y)

(* ───── unary operations ───── *)

let test_grad_exp () =
  (* exp: d/dx e^x = e^x *)
  let grad_exp = T.grad T.exp (T.scalar T.float32 0.0) in
  check_scalar ~eps "grad(exp(x)) at x=0" 1.0 (scalar_value grad_exp)

let test_grad_log () =
  (* log: d/dx ln(x) = 1/x *)
  let grad_log = T.grad T.log (T.scalar T.float32 2.0) in
  check_scalar ~eps "grad(log(x)) at x=2" 0.5 (scalar_value grad_log)

let test_grad_sin () =
  (* sin: d/dx sin(x) = cos(x) *)
  let grad_sin = T.grad T.sin (T.scalar T.float32 0.0) in
  check_scalar ~eps "grad(sin(x)) at x=0" 1.0 (scalar_value grad_sin)

let test_grad_cos () =
  (* cos: d/dx cos(x) = -sin(x) *)
  let grad_cos = T.grad T.cos (T.scalar T.float32 0.0) in
  check_scalar ~eps "grad(cos(x)) at x=0" 0.0 (scalar_value grad_cos)

let test_grad_sqrt () =
  (* sqrt: d/dx √x = 1/(2√x) *)
  let grad_sqrt = T.grad T.sqrt (T.scalar T.float32 4.0) in
  check_scalar ~eps "grad(sqrt(x)) at x=4" 0.25 (scalar_value grad_sqrt)

let test_grad_neg () =
  (* neg: d/dx (-x) = -1 *)
  let grad_neg = T.grad T.neg (T.scalar T.float32 1.0) in
  check_scalar ~eps "grad(-x)" (-1.0) (scalar_value grad_neg)

let test_grad_relu () =
  (* ReLU gradient: 0 for x<0, 1 for x>0 *)
  let x = T.create T.float32 [| 4 |] [| -2.; -1.; 1.; 2. |] in
  let grad = T.grad (fun x -> T.sum (T.relu x)) x in
  let expected = T.create T.float32 [| 4 |] [| 0.; 0.; 1.; 1. |] in
  check_rune ~eps "relu gradient" expected grad

let test_grad_tanh () =
  (* Tanh gradient: d/dx tanh(x) = 1 - tanh²(x) *)
  let x = T.scalar T.float32 0.5 in
  let grad_tanh = T.grad T.tanh x in
  let tanh_val = T.tanh x |> scalar_value in
  let expected = 1.0 -. (tanh_val *. tanh_val) in
  check_scalar ~eps:1e-4 "tanh gradient" expected (scalar_value grad_tanh)

let test_grad_abs () =
  (* Abs gradient: sign(x) *)
  let x = T.create T.float32 [| 4 |] [| -2.; -1.; 1.; 2. |] in
  let grad_abs = T.grad (fun x -> T.sum (T.abs x)) x in
  let expected = T.create T.float32 [| 4 |] [| -1.; -1.; 1.; 1. |] in
  check_rune ~eps "abs gradient" expected grad_abs

let test_grad_sigmoid () =
  (* Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x)) *)
  let x = T.scalar T.float32 0.0 in
  let grad = T.grad T.sigmoid x in
  (* At x=0, sigmoid(0) = 0.5, so gradient = 0.5 * 0.5 = 0.25 *)
  check_scalar ~eps "sigmoid gradient at x=0" 0.25 (scalar_value grad)

let test_grad_softmax () =
  (* Softmax gradient *)
  let x = T.create T.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let grad = T.grad (fun x -> T.sum (T.softmax x ~axes:[ 0 ])) x in
  (* Sum of softmax is 1, so gradient should sum to 0 *)
  let grad_sum = T.sum grad |> scalar_value in
  check_scalar ~eps:1e-5 "softmax gradient sum" 0.0 grad_sum

let test_grad_square () =
  (* Square gradient: d/dx x^2 = 2x *)
  let x = T.scalar T.float32 3.0 in
  let grad = T.grad T.square x in
  check_scalar ~eps "square gradient at x=3" 6.0 (scalar_value grad)

let test_grad_recip () =
  (* Reciprocal gradient: d/dx (1/x) = -1/x^2 *)
  let x = T.scalar T.float32 2.0 in
  let grad = T.grad T.recip x in
  check_scalar ~eps "recip gradient at x=2" (-0.25) (scalar_value grad)

let test_grad_rsqrt () =
  (* Reciprocal square root gradient: d/dx (1/sqrt(x)) = -1/(2*x^(3/2)) *)
  let x = T.scalar T.float32 4.0 in
  let grad = T.grad T.rsqrt x in
  check_scalar ~eps "rsqrt gradient at x=4" (-0.0625) (scalar_value grad)

let test_grad_sign () =
  (* Sign gradient: 0 everywhere (except at 0 where undefined) *)
  let x = T.create T.float32 [| 4 |] [| -2.; -1.; 1.; 2. |] in
  let grad = T.grad (fun x -> T.sum (T.sign x)) x in
  let expected = T.zeros_like x in
  check_rune ~eps "sign gradient" expected grad

let test_grad_tan () =
  (* Tan gradient: d/dx tan(x) = sec^2(x) = 1/cos^2(x) *)
  let x = T.scalar T.float32 0.0 in
  let grad = T.grad T.tan x in
  check_scalar ~eps "tan gradient at x=0" 1.0 (scalar_value grad)

let test_grad_sinh () =
  (* Sinh gradient: d/dx sinh(x) = cosh(x) *)
  let x = T.scalar T.float32 0.0 in
  let grad = T.grad T.sinh x in
  check_scalar ~eps "sinh gradient at x=0" 1.0 (scalar_value grad)

let test_grad_cosh () =
  (* Cosh gradient: d/dx cosh(x) = sinh(x) *)
  let x = T.scalar T.float32 0.0 in
  let grad = T.grad T.cosh x in
  check_scalar ~eps "cosh gradient at x=0" 0.0 (scalar_value grad)

(* ───── reduction operations ───── *)

let test_grad_sum () =
  (* Sum gradient: all ones *)
  let x = T.create T.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let grad = T.grad T.sum x in
  check_rune ~eps "sum gradient" (T.ones_like x) grad

let test_grad_mean () =
  (* Mean gradient: 1/n for each element *)
  let x = T.create T.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let grad = T.grad T.mean x in
  check_rune ~eps "mean gradient" (T.full T.float32 [| 2; 2 |] 0.25) grad

let test_grad_max () =
  (* Max gradient: 1 at max element, 0 elsewhere *)
  let x = T.create T.float32 [| 2; 2 |] [| 1.; 3.; 2.; 4. |] in
  let grad = T.grad T.max x in
  let expected = T.create T.float32 [| 2; 2 |] [| 0.; 0.; 0.; 1. |] in
  check_rune ~eps "max gradient" expected grad

let test_grad_sum_with_axis () =
  (* Sum with axis specified *)
  let x = T.create T.float32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let grad = T.grad (fun x -> T.sum (T.sum x ~axes:[ 1 ])) x in
  check_rune ~eps "sum with axis gradient" (T.ones_like x) grad

let test_grad_min () =
  (* Min gradient: 1 at min element, 0 elsewhere *)
  let x = T.create T.float32 [| 2; 2 |] [| 4.; 2.; 3.; 1. |] in
  let grad = T.grad T.min x in
  let expected = T.create T.float32 [| 2; 2 |] [| 0.; 0.; 0.; 1. |] in
  check_rune ~eps "min gradient" expected grad

let test_grad_prod () =
  (* Product gradient: product of all other elements *)
  let x = T.create T.float32 [| 3 |] [| 2.; 3.; 4. |] in
  let grad = T.grad T.prod x in
  let expected = T.create T.float32 [| 3 |] [| 12.; 8.; 6. |] in
  check_rune ~eps "prod gradient" expected grad

(* ───── broadcasting gradients ───── *)

let test_grad_broadcast_add () =
  (* Addition with broadcasting: [2,3] + [3] *)
  let x = T.create T.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let bias = T.create T.float32 [| 3 |] [| 0.1; 0.2; 0.3 |] in

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
    (T.full T.float32 [| 3 |] 2.0)
    grad_bias_add

let test_grad_broadcast_mul () =
  (* Multiplication with broadcasting: [2,3] * [3] *)
  let x = T.create T.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let scale = T.create T.float32 [| 3 |] [| 2.; 3.; 4. |] in

  let _, grads_mul =
    T.value_and_grads
      (fun inputs ->
        match inputs with
        | [ a; s ] -> T.sum (T.mul a s)
        | _ -> failwith "Expected 2 inputs")
      [ x; scale ]
  in
  let grad_scale_mul = List.nth grads_mul 1 in
  let expected_mul = T.create T.float32 [| 3 |] [| 5.; 7.; 9. |] in
  check_rune ~eps "mul broadcast: scale gradient" expected_mul grad_scale_mul

let test_grad_scalar_broadcast () =
  (* Scalar broadcasting for add and mul *)
  let x = T.create T.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let scalar = T.scalar T.float32 2.0 in

  let grad_scalar_add = T.grad (fun s -> T.sum (T.add x s)) scalar in
  check_scalar ~eps "scalar add broadcast" 6.0 (scalar_value grad_scalar_add);

  let grad_scalar_mul = T.grad (fun s -> T.sum (T.mul x s)) scalar in
  check_scalar ~eps "scalar mul broadcast" 21.0 (scalar_value grad_scalar_mul)

let test_grad_expand () =
  (* Expand gradient tests *)
  (* Scalar to vector *)
  let scalar = T.scalar T.float32 5.0 in
  let grad_scalar = T.grad (fun s -> T.sum (T.expand [| 3 |] s)) scalar in
  check_scalar ~eps "expand scalar gradient" 3.0 (scalar_value grad_scalar);

  (* Vector to matrix *)
  let vec = T.create T.float32 [| 3 |] [| 10.; 20.; 30. |] in
  let grad_vec = T.grad (fun v -> T.sum (T.expand [| 2; 3 |] v)) vec in
  check_rune ~eps "expand vector gradient"
    (T.full T.float32 [| 3 |] 2.0)
    grad_vec

let test_grad_where () =
  (* Where with broadcasting *)
  let cond = T.create T.uint8 [| 2; 3 |] [| 1; 0; 1; 0; 1; 0 |] in
  let x = T.create T.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y_scalar = T.scalar T.float32 10.0 in

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

let test_grad_reshape () =
  (* Reshape gradient *)
  let x = T.create T.float32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let grad = T.grad (fun x -> T.sum (T.reshape [| 3; 2 |] x)) x in
  check_rune ~eps "reshape gradient" (T.ones_like x) grad

let test_grad_transpose () =
  (* Transpose gradient *)
  let x = T.create T.float32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let grad = T.grad (fun x -> T.sum (T.transpose x)) x in
  check_rune ~eps "transpose gradient" (T.ones_like x) grad

let test_grad_squeeze () =
  (* Squeeze gradient *)
  let x = T.create T.float32 [| 1; 3; 1 |] [| 1.; 2.; 3. |] in
  let grad = T.grad (fun x -> T.sum (T.squeeze x)) x in
  check_rune ~eps "squeeze gradient" (T.ones_like x) grad

let test_grad_unsqueeze () =
  (* Unsqueeze gradient *)
  let x = T.create T.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let grad = T.grad (fun x -> T.sum (T.unsqueeze_axis 0 x)) x in
  check_rune ~eps "unsqueeze gradient" (T.ones_like x) grad

let test_grad_flatten () =
  (* Flatten gradient *)
  let x = T.create T.float32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let grad = T.grad (fun x -> T.sum (T.flatten x)) x in
  check_rune ~eps "flatten gradient" (T.ones_like x) grad

let test_grad_flip () =
  (* Flip gradient *)
  let x = T.create T.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let grad = T.grad (fun x -> T.sum (T.flip x)) x in
  check_rune ~eps "flip gradient" (T.ones_like x) grad

let test_grad_pad () =
  (* Pad gradient *)
  let x = T.create T.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let grad = T.grad (fun x -> T.sum (T.pad [| (1, 1) |] 0. x)) x in
  check_rune ~eps "pad gradient" (T.ones_like x) grad

let test_grad_tile () =
  (* Tile gradient *)
  let x = T.create T.float32 [| 2 |] [| 1.; 2. |] in
  let grad = T.grad (fun x -> T.sum (T.tile [| 3 |] x)) x in
  let expected = T.full T.float32 [| 2 |] 3.0 in
  check_rune ~eps "tile gradient" expected grad

let test_grad_concatenate () =
  (* Concatenate gradient *)
  let x = T.create T.float32 [| 2 |] [| 1.; 2. |] in
  let y = T.create T.float32 [| 3 |] [| 3.; 4.; 5. |] in
  let grad_x = T.grad (fun x -> T.sum (T.concatenate [ x; y ])) x in
  let grad_y = T.grad (fun y -> T.sum (T.concatenate [ x; y ])) y in
  check_rune ~eps "concatenate grad x" (T.ones_like x) grad_x;
  check_rune ~eps "concatenate grad y" (T.ones_like y) grad_y

let test_grad_stack () =
  (* Stack gradient *)
  let x = T.create T.float32 [| 2 |] [| 1.; 2. |] in
  let y = T.create T.float32 [| 2 |] [| 3.; 4. |] in
  let grad_x = T.grad (fun x -> T.sum (T.stack [ x; y ])) x in
  check_rune ~eps "stack gradient" (T.ones_like x) grad_x

(* Indexing operations tests *)
let test_grad_get () =
  (* Test getting a single row *)
  let x =
    T.create T.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
  in
  let grad_fn = T.grad (fun x -> T.sum (T.get [ 1 ] x)) in
  let grad = grad_fn x in
  let expected =
    T.create T.float32 [| 3; 3 |] [| 0.; 0.; 0.; 1.; 1.; 1.; 0.; 0.; 0. |]
  in
  check_rune ~eps "get single row" expected grad;

  (* Test getting a single element *)
  let grad_fn = T.grad (fun x -> T.get [ 1; 1 ] x) in
  let grad = grad_fn x in
  let expected =
    T.create T.float32 [| 3; 3 |] [| 0.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 0. |]
  in
  check_rune ~eps "get single element" expected grad

let test_grad_slice () =
  let x =
    T.create T.float32 [| 3; 4 |]
      [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
  in

  (* Test range slicing *)
  let grad_fn =
    T.grad (fun x -> T.sum (T.slice [ T.R (1, 3); T.R (0, 2) ] x))
  in
  let grad = grad_fn x in
  let expected =
    T.create T.float32 [| 3; 4 |]
      [| 0.; 0.; 0.; 0.; 1.; 1.; 0.; 0.; 1.; 1.; 0.; 0. |]
  in
  check_rune ~eps "slice range" expected grad;

  (* Test with step *)
  let grad_fn = T.grad (fun x -> T.sum (T.slice [ T.Rs (0, 3, 2) ] x)) in
  let grad = grad_fn x in
  let expected =
    T.create T.float32 [| 3; 4 |]
      [| 1.; 1.; 1.; 1.; 0.; 0.; 0.; 0.; 1.; 1.; 1.; 1. |]
  in
  check_rune ~eps "slice with step" expected grad

let test_grad_take () =
  let x = T.create T.float32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let indices = T.create T.int32 [| 3 |] [| 1l; 3l; 0l |] in

  (* Test take without axis (flattens) *)
  let grad_fn = T.grad (fun x -> T.sum (T.take indices x)) in
  let grad = grad_fn x in
  let expected = T.create T.float32 [| 5 |] [| 1.; 1.; 0.; 1.; 0. |] in
  check_rune ~eps "take" expected grad;

  (* Test 2D take with axis *)
  let x2 =
    T.create T.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
  in
  let indices2 = T.create T.int32 [| 2 |] [| 0l; 2l |] in
  let grad_fn2 = T.grad (fun x -> T.sum (T.take ~axis:1 indices2 x)) in
  let grad2 = grad_fn2 x2 in
  let expected2 =
    T.create T.float32 [| 3; 3 |] [| 1.; 0.; 1.; 1.; 0.; 1.; 1.; 0.; 1. |]
  in
  check_rune ~eps "take with axis" expected2 grad2

let test_grad_take_along_axis () =
  let x = T.create T.float32 [| 2; 3 |] [| 4.; 1.; 2.; 3.; 5.; 6. |] in
  let indices = T.create T.int32 [| 2; 1 |] [| 0l; 1l |] in

  let grad_fn = T.grad (fun x -> T.sum (T.take_along_axis ~axis:1 indices x)) in
  let grad = grad_fn x in
  let expected = T.create T.float32 [| 2; 3 |] [| 1.; 0.; 0.; 0.; 1.; 0. |] in
  check_rune ~eps "take_along_axis" expected grad

let test_grad_leaky_relu () =
  (* Leaky ReLU gradient *)
  let x = T.create T.float32 [| 4 |] [| -2.; -1.; 1.; 2. |] in
  let grad = T.grad (fun x -> T.sum (T.leaky_relu ~negative_slope:0.01 x)) x in
  let expected = T.create T.float32 [| 4 |] [| 0.01; 0.01; 1.; 1. |] in
  check_rune ~eps "leaky_relu gradient" expected grad

let test_grad_cumsum () =
  let x = T.create T.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let grad = T.grad (fun x -> T.sum (T.cumsum ~axis:0 x)) x in
  let expected = T.create T.float32 [| 4 |] [| 4.; 3.; 2.; 1. |] in
  check_rune ~eps "cumsum gradient" expected grad

let test_grad_cummax () =
  let x = T.create T.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let grad = T.grad (fun t -> T.cummax ~axis:0 t ) x in
  let expected = T.create T.float32 [| 4; 1 |] [| 1.; 1.; 1.; 1. |] in
  check_rune ~eps "cummax gradient" expected grad

let test_grad_cumprod () =
  let x = T.create T.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let grad = T.grad (fun x -> T.sum (T.cumprod ~axis:0 x)) x in
  let expected = T.create T.float32 [| 3 |] [| 9.; 4.; 2. |] in
  check_rune ~eps "cumprod gradient" expected grad

let test_grad_elu () =
  (* ELU gradient *)
  let x = T.create T.float32 [| 2 |] [| -1.; 1. |] in
  let grad = T.grad (fun x -> T.sum (T.elu ~alpha:1.0 x)) x in
  (* For x > 0: grad = 1, for x < 0: grad = alpha * exp(x) *)
  let expected = T.create T.float32 [| 2 |] [| 0.3678794; 1. |] in
  check_rune ~eps:1e-5 "elu gradient" expected grad

let test_grad_selu () =
  (* SELU gradient *)
  let x = T.create T.float32 [| 2 |] [| -1.; 1. |] in
  let grad = T.grad (fun x -> T.sum (T.selu x)) x in
  (* SELU has specific scale and alpha values *)
  let scale = 1.0507009873554804934193349852946 in
  let alpha = 1.6732632423543772848170429916717 in
  let neg_grad = scale *. alpha *. exp (-1.0) in
  let expected = T.create T.float32 [| 2 |] [| neg_grad; scale |] in
  check_rune ~eps:1e-5 "selu gradient" expected grad

(* ───── linear algebra operations ───── *)

let test_grad_dot () =
  (* Dot product gradient *)
  let x = T.create T.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let y = T.create T.float32 [| 3 |] [| 4.; 5.; 6. |] in
  let grad_x = T.grad (fun x -> T.dot x y) x in
  let grad_y = T.grad (fun y -> T.dot x y) y in
  check_rune ~eps "dot grad wrt x" y grad_x;
  check_rune ~eps "dot grad wrt y" x grad_y

let test_grad_trace () =
  (* Trace gradient: identity matrix *)
  let x =
    T.create T.float32 [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9. |]
  in
  let grad = T.grad T.trace x in
  let expected = T.eye T.float32 3 in
  check_rune ~eps "trace gradient" expected grad

let test_grad_norm () =
  (* L2 norm gradient *)
  let x = T.create T.float32 [| 3 |] [| 3.; 0.; 4. |] in
  let grad = T.grad (fun x -> T.norm x) x in
  (* Gradient of ||x|| is x/||x|| *)
  let expected = T.create T.float32 [| 3 |] [| 0.6; 0.; 0.8 |] in
  check_rune ~eps "norm gradient" expected grad

(* ───── atomic neural network operations ───── *)

let test_grad_matmul () =
  let a = T.create T.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b = T.create T.float32 [| 3; 2 |] [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6 |] in

  let f_a a = T.sum (T.matmul a b) in
  let f_b b = T.sum (T.matmul a b) in

  let grad_a = T.grad f_a a in
  let grad_b = T.grad f_b b in

  let expected_a =
    T.create T.float32 [| 2; 3 |] [| 0.3; 0.7; 1.1; 0.3; 0.7; 1.1 |]
  in
  let expected_b = T.create T.float32 [| 3; 2 |] [| 5.; 5.; 7.; 7.; 9.; 9. |] in

  check_rune ~eps "matmul grad wrt a" expected_a grad_a;
  check_rune ~eps "matmul grad wrt b" expected_b grad_b

let test_grad_pooling () =
  let x =
    T.create T.float32 [| 1; 1; 4; 4 |]
      [|
        1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12.; 13.; 14.; 15.; 16.;
      |]
  in

  let f x = T.sum (T.avg_pool2d x ~kernel_size:(2, 2) ~stride:(2, 2)) in
  let grad = T.grad f x in

  let expected = T.full T.float32 [| 1; 1; 4; 4 |] 0.25 in
  check_rune ~eps "avg_pool2d gradient" expected grad

let test_grad_conv2d () =
  (* Test 1: Simple conv2d gradient with Valid padding *)
  let x =
    T.create T.float32 [| 1; 1; 4; 4 |]
      [|
        1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12.; 13.; 14.; 15.; 16.;
      |]
  in

  let kernel =
    T.create T.float32 [| 1; 1; 2; 2 |] [| 1.; 0.; 0.; 1. |]
    (* Identity-like kernel *)
  in

  (* Test gradient w.r.t input *)
  let f_x x = T.sum (T.convolve2d x kernel ~padding_mode:`Valid) in
  let grad_x = T.grad f_x x in

  (* JAX reference: With diagonal kernel [1,0,0,1] and Valid padding *)
  let expected_x =
    T.create T.float32 [| 1; 1; 4; 4 |]
      [| 1.; 1.; 1.; 0.; 1.; 2.; 2.; 1.; 1.; 2.; 2.; 1.; 0.; 1.; 1.; 1. |]
  in
  check_rune ~eps "conv2d gradient w.r.t input (Valid)" expected_x grad_x;

  (* Test gradient w.r.t kernel *)
  let f_k k = T.sum (T.convolve2d x k ~padding_mode:`Valid) in
  let grad_k = T.grad f_k kernel in

  (* JAX reference: kernel gradient should be [99, 90, 63, 54] *)
  let expected_k =
    T.create T.float32 [| 1; 1; 2; 2 |] [| 99.; 90.; 63.; 54. |]
  in
  check_rune ~eps "conv2d gradient w.r.t kernel" expected_k grad_k;

  (* Test 2: Conv2d with Same padding *)
  (* TODO: Same padding behavior differs from JAX/TensorFlow for even-sized kernels
     This is a known difference in padding strategy. Rune follows a different convention. *)
  let f_same x = T.sum (T.convolve2d x kernel ~padding_mode:`Same) in
  let grad_same = T.grad f_same x in

  (* Rune's Same padding produces this gradient pattern *)
  let expected_same =
    T.create T.float32 [| 1; 1; 4; 4 |]
      [| 2.; 2.; 2.; 1.; 2.; 2.; 2.; 1.; 2.; 2.; 2.; 1.; 1.; 1.; 1.; 1. |]
  in
  check_rune ~eps "conv2d gradient with Same padding" expected_same grad_same

let test_grad_avg_pool_overlapping () =
  let eps = 1e-4 in
  let x =
    T.create T.float32 [| 1; 1; 4; 4 |]
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
    T.create T.float32 [| 1; 1; 4; 4 |]
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

(* ───── compound operations (loss functions, layers) ───── *)

let test_grad_linear_layer () =
  (* Combined matmul + bias pattern *)
  let batch = 4 in
  let in_dim = 2 in
  let out_dim = 8 in

  let x =
    T.create T.float32 [| batch; in_dim |]
      (Array.init (batch * in_dim) float_of_int)
  in
  let w =
    T.create T.float32 [| in_dim; out_dim |]
      (Array.init (in_dim * out_dim) (fun i -> float_of_int i *. 0.1))
  in
  let b =
    T.create T.float32 [| out_dim |]
      (Array.init out_dim (fun i -> float_of_int i *. 0.01))
  in

  let linear w b x = T.add (T.matmul x w) b in
  let loss x w b = T.sum (linear w b x) in

  let grad_b = T.grad (fun b -> loss x w b) b in
  let grad_w = T.grad (fun w -> loss x w b) w in

  check_rune ~eps "linear layer bias gradient"
    (T.full T.float32 [| out_dim |] (float_of_int batch))
    grad_b;

  let ones_out = T.ones T.float32 [| batch; out_dim |] in
  let expected_w = T.matmul (T.transpose x) ones_out in
  check_rune ~eps "linear layer weight gradient" expected_w grad_w

let test_grad_cross_entropy () =
  (* Softmax + Cross-entropy *)
  let logits =
    T.create T.float32 [| 2; 3 |] [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6 |]
  in
  let targets = T.create T.float32 [| 2; 3 |] [| 1.; 0.; 0.; 0.; 0.; 1. |] in

  let f_ce logits =
    let probs = T.softmax logits ~axes:[ 1 ] in
    let log_probs = T.log probs in
    T.neg (T.sum (T.mul targets log_probs))
  in

  let grad_ce = T.grad f_ce logits in
  let expected_ce = T.sub (T.softmax logits ~axes:[ 1 ]) targets in
  check_rune ~eps:1e-5 "cross-entropy gradient" expected_ce grad_ce

let test_grad_binary_cross_entropy () =
  (* Binary cross-entropy with sigmoid *)
  let logits_bce = T.create T.float32 [| 4; 1 |] [| -1.; 0.5; 0.5; -1. |] in
  let targets_bce = T.create T.float32 [| 4; 1 |] [| 0.; 1.; 1.; 0. |] in

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
  let expected_bce = T.div diff (T.scalar T.float32 n) in
  check_rune ~eps:1e-5 "sigmoid BCE gradient" expected_bce grad_bce

(* ───── composition and higher-order ───── *)

let test_grad_multi_variable () =
  (* Multi-variable gradient *)
  let x = T.scalar T.float32 2.0 in
  let y = T.scalar T.float32 3.0 in

  let f_x x = T.add (T.mul x x) (T.mul y y) in
  let f_y y = T.add (T.mul x x) (T.mul y y) in
  let grad_x = T.grad f_x x in
  let grad_y = T.grad f_y y in
  check_scalar ~eps "multi-var grad wrt x" 4.0 (scalar_value grad_x);
  check_scalar ~eps "multi-var grad wrt y" 6.0 (scalar_value grad_y)

let test_grad_chain_rule () =
  (* Chain rule *)
  let x = T.scalar T.float32 2.0 in
  let f_chain x =
    let y = T.mul x x in
    T.mul y y
  in
  let grad_chain = T.grad f_chain x in
  check_scalar ~eps "chain rule: grad(x⁴) at x=2" 32.0 (scalar_value grad_chain)

let test_grad_shared_subexpression () =
  (* Shared subexpression *)
  let x = T.scalar T.float32 2.0 in
  let f_shared x =
    let a = T.mul x x in
    T.add a a
  in
  let grad_shared = T.grad f_shared x in
  check_scalar ~eps "shared subexpression gradient" 8.0
    (scalar_value grad_shared)

let test_grad_second_order () =
  (* Second-order derivative *)
  let x = T.scalar T.float32 2.0 in
  let f x = T.mul x (T.mul x x) in
  let grad_f x = T.grad f x in
  let second_deriv = T.grad grad_f x in
  check_scalar ~eps "second derivative of x³ at x=2" 12.0
    (scalar_value second_deriv)

(* ───── api functions ───── *)

let test_grad_value_and_grad () =
  (* value_and_grad *)
  let x = T.scalar T.float32 2.0 in
  let f x = T.mul x x in
  let value, grad = T.value_and_grad f x in
  check_scalar ~eps "value_and_grad value" 4.0 (scalar_value value);
  check_scalar ~eps "value_and_grad grad" 4.0 (scalar_value grad)

let test_grad_value_and_grads () =
  (* grads and value_and_grads *)
  let x = T.scalar T.float32 2.0 in
  let y = T.scalar T.float32 3.0 in
  let z = T.scalar T.float32 1.0 in

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

let test_grad_pow () =
  (* Power gradient: d/dx x^n = n*x^(n-1) *)
  let x = T.scalar T.float32 2.0 in
  let n = T.scalar T.float32 3.0 in
  let grad = T.grad (fun x -> T.pow x n) x in
  check_scalar ~eps "pow gradient: x^3 at x=2" 12.0 (scalar_value grad)

let test_grad_minimum () =
  (* Minimum gradient *)
  let x = T.scalar T.float32 2.0 in
  let y = T.scalar T.float32 3.0 in
  let grad_x = T.grad (fun x -> T.minimum x y) x in
  let grad_y = T.grad (fun y -> T.minimum x y) y in
  check_scalar ~eps "minimum grad wrt x (smaller)" 1.0 (scalar_value grad_x);
  check_scalar ~eps "minimum grad wrt y (larger)" 0.0 (scalar_value grad_y)

let test_grad_maximum () =
  (* Maximum gradient *)
  let x = T.scalar T.float32 2.0 in
  let y = T.scalar T.float32 3.0 in
  let grad_x = T.grad (fun x -> T.maximum x y) x in
  let grad_y = T.grad (fun y -> T.maximum x y) y in
  check_scalar ~eps "maximum grad wrt x (smaller)" 0.0 (scalar_value grad_x);
  check_scalar ~eps "maximum grad wrt y (larger)" 1.0 (scalar_value grad_y)

let test_grad_zero () =
  (* Gradient at zero *)
  let grad = T.grad (fun x -> T.mul x x) (T.scalar T.float32 0.0) in
  check_scalar ~eps "grad(x²) at x=0" 0.0 (scalar_value grad)

let test_grad_nan_propagation () =
  (* NaN propagation in gradients *)
  let x = T.scalar T.float32 1.0 in
  let grad = T.grad (fun x -> T.div x (T.sub x x)) x in
  (* x / (x - x) = x / 0 *)
  let grad_val = scalar_value grad in
  let is_nan = Float.is_nan grad_val || Float.is_infinite grad_val in
  Alcotest.(check bool) "NaN/Inf gradient" true is_nan

let test_grad_large_values () =
  (* Test gradient with large values *)
  let x = T.scalar T.float32 1e10 in
  let grad = T.grad (fun x -> T.div x (T.scalar T.float32 1e20)) x in
  check_scalar ~eps:1e-15 "large value gradient" 1e-20 (scalar_value grad)

let test_grad_small_values () =
  (* Test gradient with very small values *)
  let x = T.scalar T.float32 1e-10 in
  let grad = T.grad (fun x -> T.mul x (T.scalar T.float32 1e10)) x in
  check_scalar ~eps "small value gradient" 1e10 (scalar_value grad)

let suite =
  [
    ( "binary operations",
      [
        test_case "add" `Quick test_grad_add;
        test_case "mul" `Quick test_grad_mul;
        test_case "sub" `Quick test_grad_sub;
        test_case "div" `Quick test_grad_div;
        test_case "pow" `Quick test_grad_pow;
        test_case "minimum" `Quick test_grad_minimum;
        test_case "maximum" `Quick test_grad_maximum;
      ] );
    ( "unary operations",
      [
        test_case "exp" `Quick test_grad_exp;
        test_case "log" `Quick test_grad_log;
        test_case "sin" `Quick test_grad_sin;
        test_case "cos" `Quick test_grad_cos;
        test_case "sqrt" `Quick test_grad_sqrt;
        test_case "neg" `Quick test_grad_neg;
        test_case "relu" `Quick test_grad_relu;
        test_case "tanh" `Quick test_grad_tanh;
        test_case "abs" `Quick test_grad_abs;
        test_case "sigmoid" `Quick test_grad_sigmoid;
        test_case "softmax" `Quick test_grad_softmax;
        test_case "square" `Quick test_grad_square;
        test_case "recip" `Quick test_grad_recip;
        test_case "rsqrt" `Quick test_grad_rsqrt;
        test_case "sign" `Quick test_grad_sign;
        test_case "tan" `Quick test_grad_tan;
        test_case "sinh" `Quick test_grad_sinh;
        test_case "cosh" `Quick test_grad_cosh;
      ] );
    ( "reduction operations",
      [
        test_case "sum" `Quick test_grad_sum;
        test_case "mean" `Quick test_grad_mean;
        test_case "max" `Quick test_grad_max;
        test_case "sum with axis" `Quick test_grad_sum_with_axis;
        test_case "min" `Quick test_grad_min;
        test_case "prod" `Quick test_grad_prod;
      ] );
    ( "broadcasting",
      [
        test_case "broadcast add" `Quick test_grad_broadcast_add;
        test_case "broadcast mul" `Quick test_grad_broadcast_mul;
        test_case "scalar broadcast" `Quick test_grad_scalar_broadcast;
        test_case "expand" `Quick test_grad_expand;
        test_case "where" `Quick test_grad_where;
      ] );
    ( "shape manipulation",
      [
        test_case "reshape" `Quick test_grad_reshape;
        test_case "transpose" `Quick test_grad_transpose;
        test_case "squeeze" `Quick test_grad_squeeze;
        test_case "unsqueeze" `Quick test_grad_unsqueeze;
        test_case "flatten" `Quick test_grad_flatten;
        test_case "flip" `Quick test_grad_flip;
        test_case "pad" `Quick test_grad_pad;
        test_case "tile" `Quick test_grad_tile;
        test_case "concatenate" `Quick test_grad_concatenate;
        test_case "stack" `Quick test_grad_stack;
      ] );
    ( "indexing operations",
      [
        test_case "get" `Quick test_grad_get;
        test_case "slice" `Quick test_grad_slice;
        test_case "take" `Quick test_grad_take;
        test_case "take_along_axis" `Quick test_grad_take_along_axis;
      ] );
    ( "linear algebra",
      [
        test_case "dot" `Quick test_grad_dot;
        test_case "trace" `Quick test_grad_trace;
        test_case "norm" `Quick test_grad_norm;
      ] );
    ( "neural network operations",
      [
        test_case "matmul" `Quick test_grad_matmul;
        test_case "pooling" `Quick test_grad_pooling;
        test_case "conv2d" `Quick test_grad_conv2d;
        test_case "avg_pool2d overlapping" `Quick test_grad_avg_pool_overlapping;
        test_case "leaky_relu" `Quick test_grad_leaky_relu;
        test_case "elu" `Quick test_grad_elu;
        test_case "selu" `Quick test_grad_selu;
      ] );
    ( "cumulative",
      [
        test_case "cumsum" `Quick test_grad_cumsum;
        test_case "cummax" `Quick test_grad_cummax;
        test_case "cumprod" `Quick test_grad_cumprod;
      ] );
    ( "compound operations",
      [
        test_case "linear layer" `Quick test_grad_linear_layer;
        test_case "cross entropy" `Quick test_grad_cross_entropy;
        test_case "binary cross entropy" `Quick test_grad_binary_cross_entropy;
      ] );
    ( "composition and higher-order",
      [
        test_case "multi-variable" `Quick test_grad_multi_variable;
        test_case "chain rule" `Quick test_grad_chain_rule;
        test_case "shared subexpression" `Quick test_grad_shared_subexpression;
        test_case "second order" `Quick test_grad_second_order;
      ] );
    ( "api functions",
      [
        test_case "value_and_grad" `Quick test_grad_value_and_grad;
        test_case "value_and_grads" `Quick test_grad_value_and_grads;
      ] );
    ( "special cases",
      [
        test_case "gradient at zero" `Quick test_grad_zero;
        test_case "NaN propagation" `Quick test_grad_nan_propagation;
        test_case "large values" `Quick test_grad_large_values;
        test_case "small values" `Quick test_grad_small_values;
      ] );
  ]

(* Test suite *)
let () = run "Rune Autodiff Tests" suite
