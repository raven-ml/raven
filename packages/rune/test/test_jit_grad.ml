(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Test suite for JIT + grad composition.

   Each test verifies that jit(grad(f))(x) produces the same result as
   grad(f)(x), ensuring that automatic differentiation composes correctly with
   JIT compilation. *)

open Windtrap
open Test_rune_support

module T = struct
  include Nx
  include Rune
end

let eps = 1e-4

let get_cpu_device () : Tolk.Device.t option =
  try Some (Tolk_cpu.create "CPU") with _ -> None

(* ───── jit(grad(f)) vs grad(f) ───── *)

let test_jit_grad_square () =
  match get_cpu_device () with
  | None -> ()
  | Some dev ->
      (* f(x) = sum(x * x), grad = 2 * x *)
      let f x = T.sum (T.mul x x) in
      let x = T.full T.float32 [| 4 |] 3.0 in
      let expected = T.grad f x in
      let grad_jit = T.jit ~device:dev (T.grad f) in
      let _ = grad_jit x in
      (* warmup *)
      let _ = grad_jit x in
      (* capture *)
      let result = grad_jit x in
      (* replay *)
      check_rune ~eps "jit(grad(sum(x*x)))" expected result

let test_jit_grad_add_const () =
  match get_cpu_device () with
  | None -> ()
  | Some dev ->
      (* f(x) = sum(x + 1), grad = all ones *)
      let f x = T.sum (T.add x (T.scalar T.float32 1.0)) in
      let x = T.full T.float32 [| 4 |] 2.0 in
      let expected = T.grad f x in
      let grad_jit = T.jit ~device:dev (T.grad f) in
      let _ = grad_jit x in
      let _ = grad_jit x in
      let result = grad_jit x in
      check_rune ~eps "jit(grad(sum(x+1)))" expected result

let test_jit_grad_sin () =
  match get_cpu_device () with
  | None -> ()
  | Some dev ->
      (* f(x) = sum(sin(x)), grad = cos(x) *)
      let f x = T.sum (T.sin x) in
      let x = T.full T.float32 [| 4 |] 1.0 in
      let expected = T.grad f x in
      let grad_jit = T.jit ~device:dev (T.grad f) in
      let _ = grad_jit x in
      let _ = grad_jit x in
      let result = grad_jit x in
      check_rune ~eps "jit(grad(sum(sin(x))))" expected result

let test_jit_grad_polynomial () =
  match get_cpu_device () with
  | None -> ()
  | Some dev ->
      (* f(x) = sum((x + 1) * x) = sum(x^2 + x), grad = 2x + 1 *)
      let f x = T.sum (T.mul (T.add x (T.scalar T.float32 1.0)) x) in
      let x = T.full T.float32 [| 4 |] 2.0 in
      let expected = T.grad f x in
      let grad_jit = T.jit ~device:dev (T.grad f) in
      let _ = grad_jit x in
      let _ = grad_jit x in
      let result = grad_jit x in
      check_rune ~eps "jit(grad(sum((x+1)*x)))" expected result

let test_jit_grad_cube () =
  match get_cpu_device () with
  | None -> ()
  | Some dev ->
      (* f(x) = sum(x * x * x), grad = 3 * x^2 *)
      let f x = T.sum (T.mul (T.mul x x) x) in
      let x = T.full T.float32 [| 4 |] 2.0 in
      let expected = T.grad f x in
      let grad_jit = T.jit ~device:dev (T.grad f) in
      let _ = grad_jit x in
      let _ = grad_jit x in
      let result = grad_jit x in
      check_rune ~eps "jit(grad(sum(x*x*x)))" expected result

let test_jit_grad_replay_different_input () =
  match get_cpu_device () with
  | None -> ()
  | Some dev ->
      (* Verify replay produces correct result for different input values *)
      let f x = T.sum (T.mul x x) in
      let x1 = T.full T.float32 [| 4 |] 3.0 in
      let x2 = T.full T.float32 [| 4 |] 5.0 in
      let grad_jit = T.jit ~device:dev (T.grad f) in
      let _ = grad_jit x1 in
      (* warmup *)
      let _ = grad_jit x1 in
      (* capture *)
      let expected = T.grad f x2 in
      let result = grad_jit x2 in
      (* replay with different input *)
      check_rune ~eps "jit(grad(sum(x*x))) replay" expected result

(* ───── Test runner ───── *)

let () =
  run "JIT + grad"
    [
      group "jit(grad(f))"
        [
          test "sum(x*x)" test_jit_grad_square;
          test "sum(x+1)" test_jit_grad_add_const;
          test "sum(sin(x))" test_jit_grad_sin;
          test "sum((x+1)*x)" test_jit_grad_polynomial;
          test "sum(x*x*x)" test_jit_grad_cube;
          test "replay different input" test_jit_grad_replay_different_input;
        ];
    ]
