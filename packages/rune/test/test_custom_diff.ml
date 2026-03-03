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

let eps = 1e-4

(* ───── Custom VJP ───── *)

let test_custom_vjp_sin () =
  (* Custom sin with explicit backward: d/dx sin(x) = cos(x) *)
  let my_sin x =
    T.custom_vjp
      ~fwd:(fun x -> (T.sin x, x))
      ~bwd:(fun x g -> T.mul g (T.cos x))
      x
  in
  let x = T.scalar T.float32 1.0 in
  let y = my_sin x in
  check_scalar ~eps "custom_vjp sin primal" (Float.sin 1.0) (scalar_value y);
  let grad_x = T.grad my_sin x in
  check_scalar ~eps "custom_vjp sin grad" (Float.cos 1.0) (scalar_value grad_x)

let test_custom_vjp_surrogate () =
  (* Surrogate gradient: forward uses sign, backward uses identity *)
  let surrogate_sign x =
    T.custom_vjp ~fwd:(fun x -> (T.sign x, ())) ~bwd:(fun () g -> g) x
  in
  let x = T.scalar T.float32 0.5 in
  let y = surrogate_sign x in
  check_scalar ~eps "surrogate sign primal" 1.0 (scalar_value y);
  let grad_x = T.grad surrogate_sign x in
  check_scalar ~eps "surrogate sign grad" 1.0 (scalar_value grad_x)

let test_custom_vjp_residuals () =
  (* Use residuals to avoid recomputation in backward *)
  let my_exp x =
    T.custom_vjp
      ~fwd:(fun x ->
        let y = T.exp x in
        (y, y))
      ~bwd:(fun y g -> T.mul g y)
      x
  in
  let x = T.scalar T.float32 2.0 in
  let grad_x = T.grad my_exp x in
  check_scalar ~eps "custom_vjp residuals grad" (Float.exp 2.0)
    (scalar_value grad_x)

let test_custom_vjp_composition () =
  (* custom_vjp composes with standard AD *)
  let my_sin x =
    T.custom_vjp
      ~fwd:(fun x -> (T.sin x, x))
      ~bwd:(fun x g -> T.mul g (T.cos x))
      x
  in
  let f x = T.mul (my_sin x) x in
  let x = T.scalar T.float32 1.0 in
  (* d/dx (sin(x) * x) = cos(x) * x + sin(x) *)
  let expected = (Float.cos 1.0 *. 1.0) +. Float.sin 1.0 in
  let grad_x = T.grad f x in
  check_scalar ~eps "custom_vjp composition grad" expected (scalar_value grad_x)

(* ───── Custom VJPs (multi-input) ───── *)

let test_custom_vjps_mul () =
  (* Custom mul with explicit backward *)
  let my_mul xs =
    T.custom_vjps
      ~fwd:(fun xs ->
        match xs with
        | [ a; b ] -> (T.mul a b, (a, b))
        | _ -> failwith "expected 2 inputs")
      ~bwd:(fun (a, b) g -> [ T.mul g b; T.mul g a ])
      xs
  in
  let x = T.scalar T.float32 3.0 in
  let y = T.scalar T.float32 4.0 in
  let result = my_mul [ x; y ] in
  check_scalar ~eps "custom_vjps mul primal" 12.0 (scalar_value result);
  let grads =
    T.grads
      (fun xs ->
        match xs with [ a; b ] -> my_mul [ a; b ] | _ -> failwith "bad")
      [ x; y ]
  in
  check_scalar ~eps "custom_vjps mul grad_x" 4.0
    (scalar_value (List.nth grads 0));
  check_scalar ~eps "custom_vjps mul grad_y" 3.0
    (scalar_value (List.nth grads 1))

(* ───── Custom JVP ───── *)

let test_custom_jvp_sin () =
  (* Custom sin with explicit tangent rule *)
  let my_sin x =
    T.custom_jvp
      ~fwd:(fun x -> T.sin x)
      ~jvp_rule:(fun x dx ->
        let y = T.sin x in
        let dy = T.mul dx (T.cos x) in
        (y, dy))
      x
  in
  let x = T.scalar T.float32 1.0 in
  let v = T.scalar T.float32 1.0 in
  let primal, tangent = T.jvp my_sin x v in
  check_scalar ~eps "custom_jvp sin primal" (Float.sin 1.0)
    (scalar_value primal);
  check_scalar ~eps "custom_jvp sin tangent" (Float.cos 1.0)
    (scalar_value tangent)

let test_custom_jvps_mul () =
  (* Custom mul with explicit tangent rule for multiple inputs *)
  let my_mul xs =
    T.custom_jvps
      ~fwd:(fun xs ->
        match xs with
        | [ a; b ] -> T.mul a b
        | _ -> failwith "expected 2 inputs")
      ~jvp_rule:(fun xs dxs ->
        match (xs, dxs) with
        | [ a; b ], [ da; db ] ->
            let y = T.mul a b in
            let dy = T.add (T.mul da b) (T.mul a db) in
            (y, dy)
        | _ -> failwith "expected 2 inputs")
      xs
  in
  let x = T.scalar T.float32 3.0 in
  let y = T.scalar T.float32 4.0 in
  let vx = T.scalar T.float32 1.0 in
  let vy = T.scalar T.float32 0.5 in
  let primal, tangent = T.jvps my_mul [ x; y ] [ vx; vy ] in
  check_scalar ~eps "custom_jvps mul primal" 12.0 (scalar_value primal);
  (* tangent = da*b + a*db = 1*4 + 3*0.5 = 5.5 *)
  check_scalar ~eps "custom_jvps mul tangent" 5.5 (scalar_value tangent)

(* ───── Fallthrough behavior ───── *)

let test_custom_vjp_under_jvp () =
  (* custom_vjp under JVP should trace through fwd normally *)
  let my_sin x =
    T.custom_vjp
      ~fwd:(fun x -> (T.sin x, x))
      ~bwd:(fun _x _g -> failwith "bwd should not be called under JVP")
      x
  in
  let x = T.scalar T.float32 1.0 in
  let v = T.scalar T.float32 1.0 in
  let primal, tangent = T.jvp my_sin x v in
  check_scalar ~eps "custom_vjp under jvp primal" (Float.sin 1.0)
    (scalar_value primal);
  check_scalar ~eps "custom_vjp under jvp tangent" (Float.cos 1.0)
    (scalar_value tangent)

let test_custom_jvp_under_vjp () =
  (* custom_jvp under VJP should trace through fwd normally *)
  let my_sin x =
    T.custom_jvp
      ~fwd:(fun x -> T.sin x)
      ~jvp_rule:(fun _x _dx ->
        failwith "jvp_rule should not be called under VJP")
      x
  in
  let x = T.scalar T.float32 1.0 in
  let grad_x = T.grad my_sin x in
  check_scalar ~eps "custom_jvp under vjp grad" (Float.cos 1.0)
    (scalar_value grad_x)

let test_custom_vjp_no_ad () =
  (* custom_vjp outside AD should just compute fwd *)
  let my_sin x =
    T.custom_vjp
      ~fwd:(fun x -> (T.sin x, ()))
      ~bwd:(fun () _g -> failwith "bwd should not be called outside AD")
      x
  in
  let x = T.scalar T.float32 1.0 in
  let y = my_sin x in
  check_scalar ~eps "custom_vjp no ad" (Float.sin 1.0) (scalar_value y)

let test_custom_jvp_no_ad () =
  (* custom_jvp outside AD should just compute fwd *)
  let my_sin x =
    T.custom_jvp
      ~fwd:(fun x -> T.sin x)
      ~jvp_rule:(fun _x _dx ->
        failwith "jvp_rule should not be called outside AD")
      x
  in
  let x = T.scalar T.float32 1.0 in
  let y = my_sin x in
  check_scalar ~eps "custom_jvp no ad" (Float.sin 1.0) (scalar_value y)

(* ───── Higher-order derivatives ───── *)

let test_custom_vjp_higher_order () =
  (* grad(grad(f)) should work with custom_vjp *)
  let my_sin x =
    T.custom_vjp
      ~fwd:(fun x -> (T.sin x, x))
      ~bwd:(fun x g -> T.mul g (T.cos x))
      x
  in
  let x = T.scalar T.float32 1.0 in
  (* d²/dx² sin(x) = -sin(x) *)
  let grad2 = T.grad (T.grad my_sin) x in
  check_scalar ~eps "custom_vjp higher order"
    (-.Float.sin 1.0)
    (scalar_value grad2)

(* ───── Multidimensional tensors ───── *)

let test_custom_vjp_array () =
  (* custom_vjp works on non-scalar tensors *)
  let my_square x =
    T.custom_vjp
      ~fwd:(fun x -> (T.mul x x, x))
      ~bwd:(fun x g -> T.mul g (T.mul (T.scalar T.float32 2.0) x))
      x
  in
  let x = T.create T.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let grad_x = T.grad (fun x -> T.sum (my_square x)) x in
  check_shape "custom_vjp array shape" [| 2; 3 |] grad_x;
  (* d/dx sum(x²) = 2x *)
  (* d/dx_i sum(x²) = 2*x_i *)
  let expected = T.create T.float32 [| 2; 3 |] [| 2.; 4.; 6.; 8.; 10.; 12. |] in
  check_scalar ~eps "custom_vjp array max diff" 0.0
    (scalar_value (T.max (T.abs (T.sub grad_x expected))))

(* ───── Gradient checking ───── *)

let test_custom_vjp_gradcheck () =
  (* Verify custom VJP agrees with finite differences *)
  let my_square x =
    T.custom_vjp
      ~fwd:(fun x -> (T.mul x x, x))
      ~bwd:(fun x g -> T.mul g (T.mul (T.scalar T.float32 2.0) x))
      x
  in
  let x = T.scalar T.float32 3.0 in
  let result = T.check_gradient my_square x in
  match result with
  | `Pass _ -> ()
  | `Fail r ->
      Windtrap.fail
        (Printf.sprintf "custom_vjp gradcheck failed: max_abs_error=%f"
           r.max_abs_error)

(* ───── Test suite ───── *)

let tests =
  [
    group "custom vjp"
      [
        test "sin" test_custom_vjp_sin;
        test "surrogate gradient" test_custom_vjp_surrogate;
        test "residuals" test_custom_vjp_residuals;
        test "composition" test_custom_vjp_composition;
      ];
    group "custom vjps" [ test "multi-input mul" test_custom_vjps_mul ];
    group "custom jvp"
      [
        test "sin" test_custom_jvp_sin;
        test "multi-input mul" test_custom_jvps_mul;
      ];
    group "fallthrough"
      [
        test "custom_vjp under jvp" test_custom_vjp_under_jvp;
        test "custom_jvp under vjp" test_custom_jvp_under_vjp;
        test "custom_vjp no ad" test_custom_vjp_no_ad;
        test "custom_jvp no ad" test_custom_jvp_no_ad;
      ];
    group "higher-order" [ test "grad of grad" test_custom_vjp_higher_order ];
    group "multidimensional" [ test "array grad" test_custom_vjp_array ];
    group "gradient checking"
      [ test "custom_vjp gradcheck" test_custom_vjp_gradcheck ];
  ]

let () = run "Rune Custom Diff Tests" tests
