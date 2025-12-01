open Alcotest
open Test_rune_support
module T = Rune

let eps = 1e-6

(* ───── Binary Operations ───── *)

let test_jvp_add () =
  let x = T.scalar T.float32 2.0 in
  let y = T.scalar T.float32 3.0 in
  let vx = T.scalar T.float32 1.0 in
  let vy = T.scalar T.float32 0.5 in

  let f inputs =
    match inputs with
    | [ a; b ] -> T.add a b
    | _ -> failwith "Expected 2 inputs"
  in
  let primal, tangent = T.jvps f [ x; y ] [ vx; vy ] in
  check_scalar ~eps "jvp(add) primal" 5.0 (scalar_value primal);
  check_scalar ~eps "jvp(add) tangent" 1.5 (scalar_value tangent)

let test_jvp_mul () =
  let x = T.scalar T.float32 2.0 in
  let y = T.scalar T.float32 3.0 in
  let vx = T.scalar T.float32 1.0 in
  let vy = T.scalar T.float32 0.5 in

  let f inputs =
    match inputs with
    | [ a; b ] -> T.mul a b
    | _ -> failwith "Expected 2 inputs"
  in
  let primal, tangent = T.jvps f [ x; y ] [ vx; vy ] in
  check_scalar ~eps "jvp(mul) primal" 6.0 (scalar_value primal);
  (* d(xy) = y*dx + x*dy = 3*1 + 2*0.5 = 4 *)
  check_scalar ~eps "jvp(mul) tangent" 4.0 (scalar_value tangent)

let test_jvp_sub () =
  let x = T.scalar T.float32 5.0 in
  let y = T.scalar T.float32 3.0 in
  let vx = T.scalar T.float32 1.0 in
  let vy = T.scalar T.float32 0.5 in

  let f inputs =
    match inputs with
    | [ a; b ] -> T.sub a b
    | _ -> failwith "Expected 2 inputs"
  in
  let primal, tangent = T.jvps f [ x; y ] [ vx; vy ] in
  check_scalar ~eps "jvp(sub) primal" 2.0 (scalar_value primal);
  check_scalar ~eps "jvp(sub) tangent" 0.5 (scalar_value tangent)

let test_jvp_div () =
  let x = T.scalar T.float32 6.0 in
  let y = T.scalar T.float32 2.0 in
  let vx = T.scalar T.float32 1.0 in
  let vy = T.scalar T.float32 0.5 in

  let f inputs =
    match inputs with
    | [ a; b ] -> T.div a b
    | _ -> failwith "Expected 2 inputs"
  in
  let primal, tangent = T.jvps f [ x; y ] [ vx; vy ] in
  check_scalar ~eps "jvp(div) primal" 3.0 (scalar_value primal);
  (* d(x/y) = dx/y - x*dy/y² = 1/2 - 6*0.5/4 = 0.5 - 0.75 = -0.25 *)
  check_scalar ~eps "jvp(div) tangent" (-0.25) (scalar_value tangent)

let test_jvp_pow () =
  let x = T.scalar T.float32 2.0 in
  let y = T.scalar T.float32 3.0 in
  let vx = T.scalar T.float32 1.0 in
  let vy = T.scalar T.float32 0.0 in

  let f inputs =
    match inputs with
    | [ a; b ] -> T.pow a b
    | _ -> failwith "Expected 2 inputs"
  in
  let primal, tangent = T.jvps f [ x; y ] [ vx; vy ] in
  check_scalar ~eps "jvp(pow) primal" 8.0 (scalar_value primal);
  (* d(x^y) = y*x^(y-1)*dx + x^y*ln(x)*dy = 3*4*1 + 0 = 12 *)
  check_scalar ~eps "jvp(pow) tangent" 12.0 (scalar_value tangent)

let test_jvp_max () =
  let x = T.scalar T.float32 2.0 in
  let y = T.scalar T.float32 3.0 in
  let vx = T.scalar T.float32 1.0 in
  let vy = T.scalar T.float32 0.5 in

  let f inputs =
    match inputs with
    | [ a; b ] -> T.maximum a b
    | _ -> failwith "Expected 2 inputs"
  in
  let primal, tangent = T.jvps f [ x; y ] [ vx; vy ] in
  check_scalar ~eps "jvp(max) primal" 3.0 (scalar_value primal);
  (* max(x,y) = y when y > x, so tangent = vy = 0.5 *)
  check_scalar ~eps "jvp(max) tangent" 0.5 (scalar_value tangent)

(* ───── Unary Operations ───── *)

let test_jvp_exp () =
  let x = T.scalar T.float32 0.0 in
  let v = T.scalar T.float32 1.0 in
  let primal, tangent = T.jvp T.exp x v in
  check_scalar ~eps "jvp(exp) primal at x=0" 1.0 (scalar_value primal);
  check_scalar ~eps "jvp(exp) tangent at x=0" 1.0 (scalar_value tangent)

let test_jvp_log () =
  let x = T.scalar T.float32 2.0 in
  let v = T.scalar T.float32 1.0 in
  let primal, tangent = T.jvp T.log x v in
  check_scalar ~eps "jvp(log) primal" (Stdlib.log 2.0) (scalar_value primal);
  check_scalar ~eps "jvp(log) tangent" 0.5 (scalar_value tangent)

let test_jvp_sin_cos () =
  let x = T.scalar T.float32 0.0 in
  let v = T.scalar T.float32 1.0 in

  let primal_sin, tangent_sin = T.jvp T.sin x v in
  check_scalar ~eps "jvp(sin) primal at x=0" 0.0 (scalar_value primal_sin);
  check_scalar ~eps "jvp(sin) tangent at x=0" 1.0 (scalar_value tangent_sin);

  let primal_cos, tangent_cos = T.jvp T.cos x v in
  check_scalar ~eps "jvp(cos) primal at x=0" 1.0 (scalar_value primal_cos);
  check_scalar ~eps "jvp(cos) tangent at x=0" 0.0 (scalar_value tangent_cos)

let test_jvp_sqrt () =
  let x = T.scalar T.float32 4.0 in
  let v = T.scalar T.float32 1.0 in
  let primal, tangent = T.jvp T.sqrt x v in
  check_scalar ~eps "jvp(sqrt) primal at x=4" 2.0 (scalar_value primal);
  (* d/dx sqrt(x) = 1/(2*sqrt(x)) = 1/4 = 0.25 *)
  check_scalar ~eps "jvp(sqrt) tangent at x=4" 0.25 (scalar_value tangent)

let test_jvp_neg () =
  let x = T.scalar T.float32 1.0 in
  let v = T.scalar T.float32 1.0 in
  let primal, tangent = T.jvp T.neg x v in
  check_scalar ~eps "jvp(neg) primal" (-1.0) (scalar_value primal);
  check_scalar ~eps "jvp(neg) tangent" (-1.0) (scalar_value tangent)

let test_jvp_relu () =
  let x = T.create T.float32 [| 4 |] [| -2.; -1.; 1.; 2. |] in
  let v = T.ones_like x in
  let primal, tangent = T.jvp T.relu x v in
  let expected_primal = T.create T.float32 [| 4 |] [| 0.; 0.; 1.; 2. |] in
  let expected_tangent = T.create T.float32 [| 4 |] [| 0.; 0.; 1.; 1. |] in
  check_rune ~eps "jvp(relu) primal" expected_primal primal;
  check_rune ~eps "jvp(relu) tangent" expected_tangent tangent

let test_jvp_tanh () =
  let x = T.scalar T.float32 0.5 in
  let v = T.scalar T.float32 1.0 in
  let primal, tangent = T.jvp T.tanh x v in
  let tanh_val = scalar_value primal in
  let expected_tangent = 1.0 -. (tanh_val *. tanh_val) in
  check_scalar ~eps:1e-4 "jvp(tanh) tangent" expected_tangent
    (scalar_value tangent)

let test_jvp_abs () =
  let x = T.create T.float32 [| 4 |] [| -2.; -1.; 1.; 2. |] in
  let v = T.ones_like x in
  let primal, tangent = T.jvp T.abs x v in
  let expected_primal = T.create T.float32 [| 4 |] [| 2.; 1.; 1.; 2. |] in
  let expected_tangent = T.create T.float32 [| 4 |] [| -1.; -1.; 1.; 1. |] in
  check_rune ~eps "jvp(abs) primal" expected_primal primal;
  check_rune ~eps "jvp(abs) tangent" expected_tangent tangent

let test_jvp_cumsum () =
  let x = T.create T.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let v = T.create T.float32 [| 3 |] [| 0.1; 0.2; 0.3 |] in
  let primal, tangent = T.jvp (fun x -> T.cumsum ~axis:0 x) x v in
  let expected_primal = T.create T.float32 [| 3 |] [| 1.; 3.; 6. |] in
  let expected_tangent = T.create T.float32 [| 3 |] [| 0.1; 0.3; 0.6 |] in
  check_rune ~eps "jvp(cumsum) primal" expected_primal primal;
  check_rune ~eps "jvp(cumsum) tangent" expected_tangent tangent

let test_jvp_cumprod () =
  let x = T.create T.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let v = T.create T.float32 [| 3 |] [| 0.1; 0.2; 0.3 |] in
  let primal, tangent = T.jvp (fun x -> T.cumprod ~axis:0 x) x v in
  let expected_primal = T.create T.float32 [| 3 |] [| 1.; 2.; 6. |] in
  let expected_tangent = T.create T.float32 [| 3 |] [| 0.1; 0.4; 1.8 |] in
  check_rune ~eps "jvp(cumprod) primal" expected_primal primal;
  check_rune ~eps "jvp(cumprod) tangent" expected_tangent tangent

let test_jvp_sigmoid () =
  let x = T.scalar T.float32 0.0 in
  let v = T.scalar T.float32 1.0 in
  let primal, tangent = T.jvp T.sigmoid x v in
  check_scalar ~eps "jvp(sigmoid) primal at x=0" 0.5 (scalar_value primal);
  (* sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)) = 0.5 * 0.5 = 0.25 *)
  check_scalar ~eps "jvp(sigmoid) tangent at x=0" 0.25 (scalar_value tangent)

let test_jvp_square () =
  let x = T.scalar T.float32 3.0 in
  let v = T.scalar T.float32 1.0 in
  let primal, tangent = T.jvp T.square x v in
  check_scalar ~eps "jvp(square) primal" 9.0 (scalar_value primal);
  check_scalar ~eps "jvp(square) tangent" 6.0 (scalar_value tangent)

let test_jvp_recip () =
  let x = T.scalar T.float32 2.0 in
  let v = T.scalar T.float32 1.0 in
  let primal, tangent = T.jvp T.recip x v in
  check_scalar ~eps "jvp(recip) primal" 0.5 (scalar_value primal);
  (* d/dx (1/x) = -1/x² = -0.25 *)
  check_scalar ~eps "jvp(recip) tangent" (-0.25) (scalar_value tangent)

let test_jvp_rsqrt () =
  let x = T.scalar T.float32 4.0 in
  let v = T.scalar T.float32 1.0 in
  let primal, tangent = T.jvp T.rsqrt x v in
  check_scalar ~eps "jvp(rsqrt) primal" 0.5 (scalar_value primal);
  (* d/dx (1/sqrt(x)) = -1/(2*x^(3/2)) = -1/16 = -0.0625 *)
  check_scalar ~eps "jvp(rsqrt) tangent" (-0.0625) (scalar_value tangent)

let test_jvp_tan () =
  let x = T.scalar T.float32 0.0 in
  let v = T.scalar T.float32 1.0 in
  let primal, tangent = T.jvp T.tan x v in
  check_scalar ~eps "jvp(tan) primal at x=0" 0.0 (scalar_value primal);
  (* d/dx tan(x) = sec²(x) = 1/cos²(x) = 1 at x=0 *)
  check_scalar ~eps "jvp(tan) tangent at x=0" 1.0 (scalar_value tangent)

let test_jvp_sinh_cosh () =
  let x = T.scalar T.float32 0.0 in
  let v = T.scalar T.float32 1.0 in

  let primal_sinh, tangent_sinh = T.jvp T.sinh x v in
  check_scalar ~eps "jvp(sinh) primal at x=0" 0.0 (scalar_value primal_sinh);
  check_scalar ~eps "jvp(sinh) tangent at x=0" 1.0 (scalar_value tangent_sinh);

  let primal_cosh, tangent_cosh = T.jvp T.cosh x v in
  check_scalar ~eps "jvp(cosh) primal at x=0" 1.0 (scalar_value primal_cosh);
  check_scalar ~eps "jvp(cosh) tangent at x=0" 0.0 (scalar_value tangent_cosh)

(* ───── Reduction Operations ───── *)

let test_jvp_sum () =
  let x = T.create T.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let v = T.ones_like x in
  let primal, tangent = T.jvp T.sum x v in
  check_scalar ~eps "jvp(sum) primal" 10.0 (scalar_value primal);
  check_scalar ~eps "jvp(sum) tangent" 4.0 (scalar_value tangent)

let test_jvp_mean () =
  let x = T.create T.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let v = T.ones_like x in
  let primal, tangent = T.jvp T.mean x v in
  check_scalar ~eps "jvp(mean) primal" 2.5 (scalar_value primal);
  check_scalar ~eps "jvp(mean) tangent" 1.0 (scalar_value tangent)

let test_jvp_max_reduction () =
  let x = T.create T.float32 [| 2; 2 |] [| 1.; 3.; 2.; 4. |] in
  let v = T.create T.float32 [| 2; 2 |] [| 0.1; 0.2; 0.3; 0.4 |] in
  let primal, tangent = T.jvp T.max x v in
  check_scalar ~eps "jvp(max) primal" 4.0 (scalar_value primal);
  (* Only the max element (4.) contributes, with tangent 0.4 *)
  check_scalar ~eps "jvp(max) tangent" 0.4 (scalar_value tangent)

let test_jvp_sum_with_axis () =
  let x = T.create T.float32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let v = T.ones_like x in
  let f x = T.sum x ~axes:[ 1 ] in
  let primal, tangent = T.jvp f x v in
  let expected_primal = T.create T.float32 [| 2 |] [| 3.; 12. |] in
  let expected_tangent = T.create T.float32 [| 2 |] [| 3.; 3. |] in
  check_rune ~eps "jvp(sum axis=1) primal" expected_primal primal;
  check_rune ~eps "jvp(sum axis=1) tangent" expected_tangent tangent

let test_jvp_prod () =
  let x = T.create T.float32 [| 3 |] [| 2.; 3.; 4. |] in
  let v = T.ones_like x in
  let primal, tangent = T.jvp T.prod x v in
  check_scalar ~eps "jvp(prod) primal" 24.0 (scalar_value primal);
  (* d(xyz) = yz*dx + xz*dy + xy*dz = 12 + 8 + 6 = 26 *)
  check_scalar ~eps "jvp(prod) tangent" 26.0 (scalar_value tangent)

(* ───── Broadcasting ───── *)

let test_jvp_broadcast_add () =
  let x = T.create T.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let bias = T.create T.float32 [| 3 |] [| 0.1; 0.2; 0.3 |] in
  let vx = T.ones_like x in
  let vb = T.ones_like bias in

  let f inputs =
    match inputs with
    | [ a; b ] -> T.add a b
    | _ -> failwith "Expected 2 inputs"
  in
  let _primal, tangent = T.jvps f [ x; bias ] [ vx; vb ] in
  (* Each position gets vx[i,j] + vb[j] = 1 + 1 = 2 *)
  check_rune ~eps "jvp(broadcast add) tangent"
    (T.full T.float32 [| 2; 3 |] 2.0)
    tangent

let test_jvp_scalar_broadcast () =
  let x = T.create T.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let scalar = T.scalar T.float32 2.0 in
  let vx = T.ones_like x in
  let vs = T.scalar T.float32 1.0 in

  let f inputs =
    match inputs with
    | [ a; s ] -> T.mul a s
    | _ -> failwith "Expected 2 inputs"
  in
  let _primal, tangent = T.jvps f [ x; scalar ] [ vx; vs ] in
  (* d(x*s) = s*dx + x*ds = 2*1 + x*1 = 2 + x *)
  let expected_tangent = T.add (T.full T.float32 [| 2; 3 |] 2.0) x in
  check_rune ~eps "jvp(scalar mul broadcast) tangent" expected_tangent tangent

(* ───── Shape Operations ───── *)

let test_jvp_reshape () =
  let x = T.create T.float32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let v = T.ones_like x in
  let f x = T.reshape [| 3; 2 |] x in
  let primal, tangent = T.jvp f x v in
  check_rune ~eps "jvp(reshape) primal" (T.reshape [| 3; 2 |] x) primal;
  check_rune ~eps "jvp(reshape) tangent" (T.reshape [| 3; 2 |] v) tangent

let test_jvp_transpose () =
  let x = T.create T.float32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |] in
  let v = T.ones_like x in
  let primal, tangent = T.jvp T.transpose x v in
  check_rune ~eps "jvp(transpose) primal" (T.transpose x) primal;
  check_rune ~eps "jvp(transpose) tangent" (T.transpose v) tangent

let test_jvp_squeeze () =
  let x = T.create T.float32 [| 1; 3; 1 |] [| 1.; 2.; 3. |] in
  let v = T.ones_like x in
  let f x = T.squeeze ~axes:[ 0; 2 ] x in
  let primal, tangent = T.jvp f x v in
  let expected_primal = T.create T.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let expected_tangent = T.ones T.float32 [| 3 |] in
  check_rune ~eps "jvp(squeeze) primal" expected_primal primal;
  check_rune ~eps "jvp(squeeze) tangent" expected_tangent tangent

let test_jvp_expand_dims () =
  let x = T.create T.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let v = T.ones_like x in
  let f x = T.expand_dims [ 0 ] x in
  let primal, tangent = T.jvp f x v in
  let expected_primal = T.create T.float32 [| 1; 3 |] [| 1.; 2.; 3. |] in
  let expected_tangent = T.ones T.float32 [| 1; 3 |] in
  check_rune ~eps "jvp(expand_dims) primal" expected_primal primal;
  check_rune ~eps "jvp(expand_dims) tangent" expected_tangent tangent

let test_jvp_concatenate () =
  let x1 = T.create T.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let x2 = T.create T.float32 [| 2; 2 |] [| 5.; 6.; 7.; 8. |] in
  let v1 = T.ones_like x1 in
  let v2 = T.full T.float32 [| 2; 2 |] 0.5 in

  let f inputs =
    match inputs with
    | [ a; b ] -> T.concatenate [ a; b ] ~axis:0
    | _ -> failwith "Expected 2 inputs"
  in
  let primal, tangent = T.jvps f [ x1; x2 ] [ v1; v2 ] in
  let expected_primal =
    T.create T.float32 [| 4; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8. |]
  in
  let expected_tangent =
    T.create T.float32 [| 4; 2 |] [| 1.; 1.; 1.; 1.; 0.5; 0.5; 0.5; 0.5 |]
  in
  check_rune ~eps "jvp(concatenate) primal" expected_primal primal;
  check_rune ~eps "jvp(concatenate) tangent" expected_tangent tangent

(* ───── Complex Compositions ───── *)

let test_jvp_softmax () =
  let x = T.create T.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let v = T.ones_like x in
  let f x = T.softmax x ~axes:[ 0 ] in
  let _primal, tangent = T.jvp f x v in
  (* Softmax Jacobian is diag(s) - s*s^T, where s = softmax(x) *)
  (* For uniform tangent v=[1,1,1], result is 0 (sum preserved) *)
  let tangent_sum = T.sum tangent |> scalar_value in
  check_scalar ~eps:1e-5 "jvp(softmax) tangent sum" 0.0 tangent_sum

let test_jvp_layer_norm () =
  let x = T.create T.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let v = T.ones_like x in
  let f x =
    let mean = T.mean x ~axes:[ 1 ] ~keepdims:true in
    let centered = T.sub x mean in
    let var = T.mean (T.square centered) ~axes:[ 1 ] ~keepdims:true in
    let std = T.sqrt (T.add var (T.scalar T.float32 1e-5)) in
    T.div centered std
  in
  let _primal, tangent = T.jvp f x v in
  (* Layer norm preserves zero mean in tangent space - check total sum is
     small *)
  let total_row_sum = T.sum (T.sum tangent ~axes:[ 1 ]) |> scalar_value in
  check_scalar ~eps:1e-3 "jvp(layer_norm) total row sum" 0.0
    (Float.abs total_row_sum)

let test_jvp_nested () =
  (* Test nested JVP calls *)
  let x = T.scalar T.float32 2.0 in
  let v = T.scalar T.float32 1.0 in

  (* f(x) = exp(sin(x²)) *)
  let f x = T.exp (T.sin (T.square x)) in
  let _primal, tangent = T.jvp f x v in

  (* Manual computation: f'(x) = exp(sin(x²)) * cos(x²) * 2x At x=2: sin(4) ≈
     -0.757, cos(4) ≈ -0.654, exp(-0.757) ≈ 0.469 f'(2) ≈ 0.469 * (-0.654) * 4 ≈
     -1.227 *)
  check_scalar ~eps:1e-3 "jvp(nested) tangent" (-1.227) (scalar_value tangent)

let test_jvp_higher_order () =
  (* Second derivative via nested JVP *)
  let x = T.scalar T.float32 1.0 in

  (* f(x) = x³ *)
  let f x = T.mul (T.square x) x in

  (* First derivative: 3x² *)
  let _, first_deriv = T.jvp f x (T.scalar T.float32 1.0) in
  check_scalar ~eps "first derivative of x³ at x=1" 3.0
    (scalar_value first_deriv);

  (* Second derivative via JVP of JVP: 6x *)
  let f_jvp x =
    let _, tangent = T.jvp f x (T.scalar T.float32 1.0) in
    tangent
  in
  let _, second_deriv = T.jvp f_jvp x (T.scalar T.float32 1.0) in
  check_scalar ~eps "second derivative of x³ at x=1" 6.0
    (scalar_value second_deriv)

(* ───── Edge Cases ───── *)

let test_jvp_zero_tangent () =
  (* Zero tangent should give zero output tangent *)
  let x = T.scalar T.float32 2.0 in
  let v = T.scalar T.float32 0.0 in
  let f x = T.mul (T.exp x) (T.sin x) in
  let _primal, tangent = T.jvp f x v in
  check_scalar ~eps "jvp with zero tangent" 0.0 (scalar_value tangent)

let test_jvp_constant_function () =
  (* Constant function should have zero tangent *)
  let x = T.scalar T.float32 2.0 in
  let v = T.scalar T.float32 1.0 in
  let f _ = T.scalar T.float32 42.0 in
  let primal, tangent = T.jvp f x v in
  check_scalar ~eps "jvp(constant) primal" 42.0 (scalar_value primal);
  check_scalar ~eps "jvp(constant) tangent" 0.0 (scalar_value tangent)

let test_jvp_identity () =
  (* Identity function should pass through tangent *)
  let x = T.create T.float32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let v = T.create T.float32 [| 2; 2 |] [| 0.1; 0.2; 0.3; 0.4 |] in
  let f x = x in
  let primal, tangent = T.jvp f x v in
  check_rune ~eps "jvp(identity) primal" x primal;
  check_rune ~eps "jvp(identity) tangent" v tangent

(* ───── Indexing Operations ───── *)

let test_jvp_slice () =
  let x = T.create T.float32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let v = T.create T.float32 [| 4 |] [| 0.1; 0.2; 0.3; 0.4 |] in
  let f x = T.slice [ T.R (1, 3) ] x in
  let primal, tangent = T.jvp f x v in
  let expected_primal = T.create T.float32 [| 2 |] [| 2.; 3. |] in
  let expected_tangent = T.create T.float32 [| 2 |] [| 0.2; 0.3 |] in
  check_rune ~eps "jvp(slice) primal" expected_primal primal;
  check_rune ~eps "jvp(slice) tangent" expected_tangent tangent

let test_jvp_gather () =
  let x = T.create T.float32 [| 4 |] [| 10.; 20.; 30.; 40. |] in
  let v = T.create T.float32 [| 4 |] [| 0.1; 0.2; 0.3; 0.4 |] in
  let indices = T.create T.int32 [| 3 |] [| 2l; 0l; 3l |] in
  let f x = T.take ~axis:0 indices x in
  let primal, tangent = T.jvp f x v in
  let expected_primal = T.create T.float32 [| 3 |] [| 30.; 10.; 40. |] in
  let expected_tangent = T.create T.float32 [| 3 |] [| 0.3; 0.1; 0.4 |] in
  check_rune ~eps "jvp(gather) primal" expected_primal primal;
  check_rune ~eps "jvp(gather) tangent" expected_tangent tangent

let test_jvp_get () =
  let x = T.create T.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let v = T.create T.float32 [| 2; 3 |] [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6 |] in
  let f x = T.get [ 1; 2 ] x in
  let primal, tangent = T.jvp f x v in
  check_scalar ~eps "jvp(get) primal" 6.0 (scalar_value primal);
  check_scalar ~eps "jvp(get) tangent" 0.6 (scalar_value tangent)

let test_jvp_take_along_axis () =
  let x =
    T.create T.float32 [| 3; 4 |]
      [| 10.; 20.; 30.; 40.; 50.; 60.; 70.; 80.; 90.; 100.; 110.; 120. |]
  in
  let v =
    T.create T.float32 [| 3; 4 |]
      [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1.0; 1.1; 1.2 |]
  in
  let indices = T.create T.int32 [| 3; 2 |] [| 1l; 3l; 0l; 2l; 2l; 1l |] in
  let f x = T.take_along_axis ~axis:1 indices x in
  let primal, tangent = T.jvp f x v in
  let expected_primal =
    T.create T.float32 [| 3; 2 |] [| 20.; 40.; 50.; 70.; 110.; 100. |]
  in
  let expected_tangent =
    T.create T.float32 [| 3; 2 |] [| 0.2; 0.4; 0.5; 0.7; 1.1; 1.0 |]
  in
  check_rune ~eps "jvp(take_along_axis) primal" expected_primal primal;
  check_rune ~eps "jvp(take_along_axis) tangent" expected_tangent tangent

(* ───── FFT Operations ───── *)

(* Check complex tensors for approximate equality using magnitude of
   difference *)
let check_complex_close ~eps msg expected actual =
  let diff = T.sub expected actual in
  (* |z|^2 = z * conj(z) for complex numbers *)
  let mag_sq = T.mul diff (T.conjugate diff) in
  (* Sum of squared magnitudes *)
  let total_err = T.sum mag_sq in
  let err_val = (T.item [] total_err : Complex.t).re in
  if
    err_val
    > eps *. eps *. Float.of_int (Array.fold_left ( * ) 1 (T.shape expected))
  then
    Alcotest.failf "%s: complex tensors differ, total squared error = %.6e" msg
      err_val

let test_jvp_fft () =
  (* FFT is linear, so JVP should be FFT of tangent *)
  let x =
    T.create T.complex64 [| 4 |]
      [|
        Complex.{ re = 1.0; im = 0.0 };
        Complex.{ re = 2.0; im = 0.0 };
        Complex.{ re = 3.0; im = 0.0 };
        Complex.{ re = 4.0; im = 0.0 };
      |]
  in
  let v =
    T.create T.complex64 [| 4 |]
      [|
        Complex.{ re = 0.1; im = 0.0 };
        Complex.{ re = 0.2; im = 0.0 };
        Complex.{ re = 0.3; im = 0.0 };
        Complex.{ re = 0.4; im = 0.0 };
      |]
  in
  let f x = T.fft ~axis:0 x in
  let primal, tangent = T.jvp f x v in
  let expected_tangent = T.fft ~axis:0 v in
  check_complex_close ~eps:1e-5 "jvp(fft) primal" (f x) primal;
  check_complex_close ~eps:1e-5 "jvp(fft) tangent" expected_tangent tangent

let test_jvp_ifft () =
  (* IFFT is linear, so JVP should be IFFT of tangent *)
  let x =
    T.create T.complex64 [| 4 |]
      [|
        Complex.{ re = 10.0; im = 0.0 };
        Complex.{ re = -2.0; im = 2.0 };
        Complex.{ re = -2.0; im = 0.0 };
        Complex.{ re = -2.0; im = -2.0 };
      |]
  in
  let v =
    T.create T.complex64 [| 4 |]
      [|
        Complex.{ re = 1.0; im = 0.0 };
        Complex.{ re = 0.0; im = 1.0 };
        Complex.{ re = -1.0; im = 0.0 };
        Complex.{ re = 0.0; im = -1.0 };
      |]
  in
  let f x = T.ifft ~axis:0 x in
  let primal, tangent = T.jvp f x v in
  let expected_tangent = T.ifft ~axis:0 v in
  check_complex_close ~eps:1e-5 "jvp(ifft) primal" (f x) primal;
  check_complex_close ~eps:1e-5 "jvp(ifft) tangent" expected_tangent tangent

let test_jvp_fft_roundtrip () =
  (* FFT followed by IFFT should be identity, tangent should pass through *)
  let x =
    T.create T.complex64 [| 4 |]
      [|
        Complex.{ re = 1.0; im = 0.5 };
        Complex.{ re = 2.0; im = -0.5 };
        Complex.{ re = 3.0; im = 0.2 };
        Complex.{ re = 4.0; im = -0.2 };
      |]
  in
  let v =
    T.create T.complex64 [| 4 |]
      [|
        Complex.{ re = 0.1; im = 0.05 };
        Complex.{ re = 0.2; im = -0.05 };
        Complex.{ re = 0.3; im = 0.02 };
        Complex.{ re = 0.4; im = -0.02 };
      |]
  in
  let f x = T.ifft ~axis:0 (T.fft ~axis:0 x) in
  let primal, tangent = T.jvp f x v in
  (* Roundtrip should give back original *)
  check_complex_close ~eps:1e-5 "jvp(fft roundtrip) primal" x primal;
  check_complex_close ~eps:1e-5 "jvp(fft roundtrip) tangent" v tangent

(* Test suite *)
let () =
  run "Rune JVP Comprehensive Tests"
    [
      ( "binary operations",
        [
          test_case "add" `Quick test_jvp_add;
          test_case "mul" `Quick test_jvp_mul;
          test_case "sub" `Quick test_jvp_sub;
          test_case "div" `Quick test_jvp_div;
          test_case "pow" `Quick test_jvp_pow;
          test_case "max" `Quick test_jvp_max;
        ] );
      ( "unary operations",
        [
          test_case "exp" `Quick test_jvp_exp;
          test_case "log" `Quick test_jvp_log;
          test_case "sin/cos" `Quick test_jvp_sin_cos;
          test_case "sqrt" `Quick test_jvp_sqrt;
          test_case "neg" `Quick test_jvp_neg;
          test_case "relu" `Quick test_jvp_relu;
          test_case "tanh" `Quick test_jvp_tanh;
          test_case "abs" `Quick test_jvp_abs;
          test_case "cumsum" `Quick test_jvp_cumsum;
          test_case "cumprod" `Quick test_jvp_cumprod;
          test_case "sigmoid" `Quick test_jvp_sigmoid;
          test_case "square" `Quick test_jvp_square;
          test_case "recip" `Quick test_jvp_recip;
          test_case "rsqrt" `Quick test_jvp_rsqrt;
          test_case "tan" `Quick test_jvp_tan;
          test_case "sinh/cosh" `Quick test_jvp_sinh_cosh;
        ] );
      ( "reduction operations",
        [
          test_case "sum" `Quick test_jvp_sum;
          test_case "mean" `Quick test_jvp_mean;
          test_case "max" `Quick test_jvp_max_reduction;
          test_case "sum with axis" `Quick test_jvp_sum_with_axis;
          test_case "prod" `Quick test_jvp_prod;
        ] );
      ( "broadcasting",
        [
          test_case "broadcast add" `Quick test_jvp_broadcast_add;
          test_case "scalar broadcast" `Quick test_jvp_scalar_broadcast;
        ] );
      ( "shape operations",
        [
          test_case "reshape" `Quick test_jvp_reshape;
          test_case "transpose" `Quick test_jvp_transpose;
          test_case "squeeze" `Quick test_jvp_squeeze;
          test_case "expand_dims" `Quick test_jvp_expand_dims;
          test_case "concatenate" `Quick test_jvp_concatenate;
        ] );
      ( "complex compositions",
        [
          test_case "softmax" `Quick test_jvp_softmax;
          test_case "layer norm" `Quick test_jvp_layer_norm;
          test_case "nested" `Quick test_jvp_nested;
          test_case "higher order" `Quick test_jvp_higher_order;
        ] );
      ( "edge cases",
        [
          test_case "zero tangent" `Quick test_jvp_zero_tangent;
          test_case "constant function" `Quick test_jvp_constant_function;
          test_case "identity" `Quick test_jvp_identity;
        ] );
      ( "fft operations",
        [
          test_case "fft" `Quick test_jvp_fft;
          test_case "ifft" `Quick test_jvp_ifft;
          test_case "fft roundtrip" `Quick test_jvp_fft_roundtrip;
        ] );
      ( "indexing operations",
        [
          test_case "slice" `Quick test_jvp_slice;
          test_case "gather" `Quick test_jvp_gather;
          test_case "get" `Quick test_jvp_get;
          test_case "take_along_axis" `Quick test_jvp_take_along_axis;
        ] );
    ]
