open Alcotest
open Test_rune_support
module T = Rune

let ctx = T.ocaml
let eps = 1e-6

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
    | [ a; b ] -> T.add a b
    | _ -> failwith "Expected 2 inputs"
  in
  let primal_add, tangent_add = T.jvps f [ x; y ] [ vx; vy ] in
  check_scalar ~eps "jvp(add) primal" 5.0 (scalar_value primal_add);
  check_scalar ~eps "jvp(add) tangent" 1.5 (scalar_value tangent_add);

  (* Test multiplication *)
  let f_mul inputs =
    match inputs with
    | [ a; b ] -> T.mul a b
    | _ -> failwith "Expected 2 inputs"
  in
  let primal_mul, tangent_mul = T.jvps f_mul [ x; y ] [ vx; vy ] in
  check_scalar ~eps "jvp(mul) primal" 6.0 (scalar_value primal_mul);
  (* d(xy) = y*dx + x*dy = 2*1 + 3*0.5 = 3.5 *)
  check_scalar ~eps "jvp(mul) tangent" 3.5 (scalar_value tangent_mul);

  (* Test division *)
  let f_div inputs =
    match inputs with
    | [ a; b ] -> T.div a b
    | _ -> failwith "Expected 2 inputs"
  in
  let primal_div, tangent_div = T.jvps f_div [ x; y ] [ vx; vy ] in
  check_scalar ~eps "jvp(div) primal" 1.5 (scalar_value primal_div);
  (* d(x/y) = dx/y - x*dy/y² = 1/2 - 3*0.5/4 = 0.5 - 0.375 = 0.125 *)
  check_scalar ~eps "jvp(div) tangent" 0.125 (scalar_value tangent_div)

let test_jvp_shape_ops () =
  let x = T.create ctx T.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let v = T.ones_like x in

  (* Reshape *)
  let f_reshape x = T.reshape [| 3; 2 |] x in
  let primal_reshape, tangent_reshape = T.jvp f_reshape x v in
  check_rune ~eps "jvp(reshape) primal" (T.reshape [| 3; 2 |] x) primal_reshape;
  check_rune ~eps "jvp(reshape) tangent"
    (T.reshape [| 3; 2 |] v)
    tangent_reshape;

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
  let a = T.create ctx T.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b =
    T.create ctx T.float32 [| 3; 2 |] [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6 |]
  in
  let va = T.ones_like a in
  let vb = T.zeros_like b in

  (* Test matmul with tangent only in first argument *)
  let f inputs =
    match inputs with
    | [ a; b ] -> T.matmul a b
    | _ -> failwith "Expected 2 inputs"
  in
  let primal, tangent = T.jvps f [ a; b ] [ va; vb ] in

  (* Expected tangent: va @ b + a @ vb = ones @ b + a @ zeros = ones @ b *)
  let expected_tangent = T.matmul va b in
  check_rune ~eps "jvp(matmul) primal" (T.matmul a b) primal;
  check_rune ~eps "jvp(matmul) tangent" expected_tangent tangent;

  (* Test with tangent in both arguments *)
  let vb2 = T.ones_like b in
  let _, tangent2 = T.jvps f [ a; b ] [ va; vb2 ] in
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
    | [ x; y; z ] ->
        let xy = T.mul x y in
        let yz = T.mul y z in
        let xz = T.mul x z in
        T.add (T.add xy yz) xz
    | _ -> failwith "Expected 3 inputs"
  in

  let primal, tangent = T.jvps f [ x; y; z ] [ vx; vy; vz ] in

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
    let aux = T.shape y in
    (* auxiliary output *)
    (y, aux)
  in

  let primal, tangent, aux = T.jvp_aux f_with_aux x v in

  check_scalar ~eps "jvp_aux primal" 9.0 (scalar_value primal);
  check_scalar ~eps "jvp_aux tangent" 6.0 (scalar_value tangent);
  check (array int) "jvp_aux auxiliary" [||] aux

(* Test suite *)
let () =
  run "Rune JVP Tests"
    [
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
