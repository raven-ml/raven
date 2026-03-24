(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Test suite for JIT + vmap composition.

   Each test verifies that jit(vmap(f))(x) produces the same result as
   vmap(f)(x), ensuring that vectorized mapping composes correctly with JIT
   compilation. *)

open Windtrap
open Test_rune_support

module T = struct
  include Nx
  include Rune
end

let eps = 1e-4

let get_cpu_device () : Tolk.Device.t option =
  try Some (Tolk_cpu.create "CPU") with _ -> None

(* ───── jit(vmap(f)) vs vmap(f) ───── *)

let test_jit_vmap_mul_scalar () =
  match get_cpu_device () with
  | None -> ()
  | Some dev ->
      (* f(x) = x * 2, vmapped over batch dim *)
      let f x = T.mul x (T.scalar T.float32 2.0) in
      let x = T.full T.float32 [| 3; 4 |] 3.0 in
      let expected = T.vmap f x in
      let vmap_jit = T.jit ~device:dev (T.vmap f) in
      let _ = vmap_jit x in
      (* warmup *)
      let _ = vmap_jit x in
      (* capture *)
      let result = vmap_jit x in
      (* replay *)
      check_rune ~eps "jit(vmap(x*2))" expected result

let test_jit_vmap_self_add () =
  match get_cpu_device () with
  | None -> ()
  | Some dev ->
      (* f(x) = x + x, vmapped *)
      let f x = T.add x x in
      let x = T.full T.float32 [| 3; 4 |] 2.0 in
      let expected = T.vmap f x in
      let vmap_jit = T.jit ~device:dev (T.vmap f) in
      let _ = vmap_jit x in
      let _ = vmap_jit x in
      let result = vmap_jit x in
      check_rune ~eps "jit(vmap(x+x))" expected result

let test_jit_vmap_sum () =
  match get_cpu_device () with
  | None -> ()
  | Some dev ->
      (* f(x) = sum(x), vmapped: reduce per-batch element *)
      let f x = T.sum x in
      let x = T.full T.float32 [| 3; 4 |] 1.0 in
      let expected = T.vmap f x in
      let vmap_jit = T.jit ~device:dev (T.vmap f) in
      let _ = vmap_jit x in
      let _ = vmap_jit x in
      let result = vmap_jit x in
      check_rune ~eps "jit(vmap(sum(x)))" expected result

let test_jit_vmap_square () =
  match get_cpu_device () with
  | None -> ()
  | Some dev ->
      (* f(x) = x * x, vmapped *)
      let f x = T.mul x x in
      let x = T.full T.float32 [| 3; 4 |] 3.0 in
      let expected = T.vmap f x in
      let vmap_jit = T.jit ~device:dev (T.vmap f) in
      let _ = vmap_jit x in
      let _ = vmap_jit x in
      let result = vmap_jit x in
      check_rune ~eps "jit(vmap(x*x))" expected result

(* ───── Test runner ───── *)

let () =
  run "JIT + vmap"
    [
      group "jit(vmap(f))"
        [
          test "x * 2" test_jit_vmap_mul_scalar;
          test "x + x" test_jit_vmap_self_add;
          test "sum(x)" test_jit_vmap_sum;
          test "x * x" test_jit_vmap_square;
        ];
    ]
