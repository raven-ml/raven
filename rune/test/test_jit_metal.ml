open Alcotest
open Test_rune_support
module T = Rune

let eps = 1e-6

(* Basic JIT compilation tests *)
let test_jit_simple () =
  let f x = T.add x x in
  let jit_f = T.jit f in

  let x = T.scalar T.float32 3.0 in
  let expected = scalar_value (f x) in
  let actual = scalar_value (jit_f x) in

  check_scalar ~eps "jit simple addition" expected actual

let test_jit_multiple_ops () =
  let f x = T.add (T.mul x x) x in
  let jit_f = T.jit f in

  let x = T.scalar T.float32 2.0 in
  let expected = scalar_value (f x) in
  let actual = scalar_value (jit_f x) in

  check_scalar ~eps "jit multiple operations" expected actual

let test_jit_with_constant () =
  let f x =
    let c = T.scalar T.float32 2.0 in
    T.add (T.mul x c) c
  in
  let jit_f = T.jit f in

  let x = T.scalar T.float32 3.0 in
  let expected = scalar_value (f x) in
  let actual = scalar_value (jit_f x) in

  check_scalar ~eps "jit with constant" expected actual

(* Shape handling tests *)
let test_jit_fixed_shape () =
  let f x = T.sum (T.mul x x) in
  let jit_f = T.jit f in

  let x = T.ones T.float32 [| 2; 3 |] in
  let expected = scalar_value (f x) in
  let actual = scalar_value (jit_f x) in

  check_scalar ~eps "jit fixed shape" expected actual

let test_jit_different_shapes () =
  let f x = T.sum (T.mul x x) in
  let jit_f = T.jit f in

  (* First shape *)
  let x1 = T.ones T.float32 [| 2; 3 |] in
  let result1 = jit_f x1 in
  check_scalar ~eps "jit shape 1" 6.0 (scalar_value result1);

  (* Different shape - should recompile *)
  let x2 = T.ones T.float32 [| 3; 4 |] in
  let result2 = jit_f x2 in
  check_scalar ~eps "jit shape 2" 12.0 (scalar_value result2)

let test_jit_batch_dimensions () =
  let f x = T.mean x ~axes:[| 1 |] in
  let jit_f = T.jit f in

  (* Different batch sizes *)
  let x1 = T.ones T.float32 [| 2; 5 |] in
  let x2 = T.ones T.float32 [| 3; 5 |] in

  let result1 = jit_f x1 in
  let result2 = jit_f x2 in

  check_rune "jit batch 1" (T.ones T.float32 [| 2 |]) result1;
  check_rune "jit batch 2" (T.ones T.float32 [| 3 |]) result2

(* Autodiff integration tests *)
let test_jit_of_grad () =
  let f x = T.mul x (T.mul x x) in
  let grad_f x = T.grad f x in
  let jit_grad_f = T.jit grad_f in

  let x = T.scalar T.float32 2.0 in
  let expected = scalar_value (grad_f x) in
  let actual = scalar_value (jit_grad_f x) in

  check_scalar ~eps "jit of grad" expected actual

let test_grad_of_jit () =
  let f x = T.add (T.mul x x) x in
  let jit_f = T.jit f in

  let x = T.scalar T.float32 3.0 in
  let expected_grad = scalar_value (T.grad f x) in
  let actual_grad = scalar_value (T.grad jit_f x) in

  check_scalar ~eps "grad of jit" expected_grad actual_grad

let test_jit_grad_composition () =
  (* Test both JIT(grad(f)) and grad(JIT(f)) give same results *)
  let f x = T.sum (T.mul x x) in

  let x =
    T.create T.float32 [| 3; 2 |] [| 0.1; 0.2; -0.3; 0.4; -0.5; 0.6 |]
  in

  (* Regular gradient *)
  let grad_f x = T.grad f x in
  let regular_grad = grad_f x in

  (* JIT of gradient function *)
  let jit_grad_f = T.jit grad_f in
  let jit_grad_result = jit_grad_f x in

  (* Gradient of JIT function *)
  let jit_f = T.jit f in
  let grad_jit_result = T.grad jit_f x in

  check_rune ~eps "JIT(grad) vs regular" regular_grad jit_grad_result;
  check_rune ~eps "grad(JIT) vs regular" regular_grad grad_jit_result

(* Complex operations *)
let test_jit_reduction_ops () =
  let f x = T.mean (T.relu x) ~axes:[| 1 |] in
  let jit_f = T.jit f in

  let x =
    T.create T.float32 [| 4; 5 |]
      [|
        -0.1;
        0.2;
        -0.3;
        0.4;
        -0.5;
        0.6;
        -0.7;
        0.8;
        -0.9;
        1.0;
        -1.1;
        1.2;
        -1.3;
        1.4;
        -1.5;
        1.6;
        -1.7;
        1.8;
        -1.9;
        2.0;
      |]
  in
  let expected = f x in
  let actual = jit_f x in

  check_rune ~eps "jit reduction ops" expected actual

(* TODO: Implement when jit2 is available let test_jit_broadcast_ops () = let f
   x y = T.add (T.mul x y) x in let jit_f = T.jit2 f in

   let x = T.ones T.float32 [|3; 1|] in let y = T.ones T.float32 [|1;
   4|] in

   let expected = f x y in let actual = jit_f x y in

   check_rune "jit broadcast ops" expected actual *)

(* Test suite *)
let () =
  run "Rune JIT Tests"
    [
      ( "basic compilation",
        [
          test_case "simple operation" `Quick test_jit_simple;
          test_case "multiple operations" `Quick test_jit_multiple_ops;
          test_case "with constant" `Quick test_jit_with_constant;
        ] );
      ( "shape handling",
        [
          test_case "fixed shape" `Quick test_jit_fixed_shape;
          test_case "different shapes" `Quick test_jit_different_shapes;
          test_case "batch dimensions" `Quick test_jit_batch_dimensions;
        ] );
      ( "autodiff integration",
        [
          test_case "JIT of grad" `Quick test_jit_of_grad;
          test_case "grad of JIT" `Quick test_grad_of_jit;
          test_case "JIT-grad composition" `Quick test_jit_grad_composition;
        ] );
      ( "complex operations",
        [
          test_case "reduction ops" `Quick test_jit_reduction_ops;
          (* TODO: Enable when jit2 is available test_case "broadcast ops"
             `Quick test_jit_broadcast_ops; *)
        ] );
    ]
