(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Test suite for the JIT effect handler.

   Tests are split into two groups: - Without device: tests the graph building
   and state machine - With device: tests the full pipeline (build → compile →
   execute → replay)

   Without a device, capture should fail (no eager fallback). *)

open Windtrap
open Test_rune_support

module T = struct
  include Nx
  include Rune
end

let eps = 1e-5

(* ───── Without device: graph capture + state machine ───── *)

let test_jit_no_device_fails () =
  (* JIT without device should fail on capture *)
  let f x = T.add x (T.scalar T.float32 1.0) in
  let f_jit = T.jit f in
  let x = T.scalar T.float32 5.0 in
  let _ = f_jit x in
  (* warmup: ok, runs eagerly *)
  let raised = ref false in
  (try ignore (f_jit x) (* capture: should fail, no device *)
   with Failure _ -> raised := true);
  is_true !raised

let test_jit_warmup_is_eager () =
  (* First call should execute eagerly and return correct result *)
  let f x = T.add x (T.scalar T.float32 1.0) in
  let f_jit = T.jit f in
  let x = T.scalar T.float32 5.0 in
  let result = f_jit x in
  check_scalar ~eps "warmup result" 6.0 (scalar_value result)

let test_jit_warmup_calls_f () =
  let called = ref false in
  let f x =
    called := true;
    T.add x (T.scalar T.float32 1.0)
  in
  let f_jit = T.jit f in
  let x = T.scalar T.float32 0.0 in
  let _ = f_jit x in
  is_true !called

(* ───── With device: full pipeline ───── *)

(* To test the full pipeline, we need a Tolk device. The CPU device is available
   via tolk.cpu. These tests will only run when a device is available. *)

let get_cpu_device () : Tolk.Device.t option =
  try Some (Tolk_cpu.create "CPU") with _ -> None

let test_jit_capture_compiles () =
  match get_cpu_device () with
  | None -> () (* skip: no device *)
  | Some dev ->
      let f x = T.add x (T.scalar T.float32 1.0) in
      let f_jit = T.jit ~device:dev f in
      let x = T.scalar T.float32 5.0 in
      let _ = f_jit x in
      (* warmup *)
      (* Capture should build graph, compile, and return result *)
      let result = f_jit x in
      check_scalar ~eps "capture result" 6.0 (scalar_value result)

let test_jit_replay_no_recompile () =
  match get_cpu_device () with
  | None -> ()
  | Some dev ->
      let call_count = ref 0 in
      let f x =
        incr call_count;
        T.add x (T.scalar T.float32 1.0)
      in
      let f_jit = T.jit ~device:dev f in
      let x = T.scalar T.float32 5.0 in
      let _ = f_jit x in
      (* warmup: f called *)
      equal int 1 !call_count;
      let _ = f_jit x in
      (* capture: f called under handler *)
      equal int 2 !call_count;
      let _ = f_jit x in
      (* replay: f should NOT be called *)
      equal int 2 !call_count

let test_jit_replay_different_values () =
  match get_cpu_device () with
  | None -> ()
  | Some dev ->
      let f x = T.mul x (T.scalar T.float32 3.0) in
      let f_jit = T.jit ~device:dev f in
      let _ = f_jit (T.scalar T.float32 2.0) in
      (* warmup *)
      let _ = f_jit (T.scalar T.float32 2.0) in
      (* capture *)
      let result = f_jit (T.scalar T.float32 7.0) in
      (* replay *)
      check_scalar ~eps "replay 7*3" 21.0 (scalar_value result)

let test_jit_shape_mismatch_rejected () =
  match get_cpu_device () with
  | None -> ()
  | Some dev ->
      let f x = T.add x x in
      let f_jit = T.jit ~device:dev f in
      let x4 = T.full T.float32 [| 4 |] 1.0 in
      let x8 = T.full T.float32 [| 8 |] 1.0 in
      let _ = f_jit x4 in
      (* warmup *)
      let _ = f_jit x4 in
      (* capture *)
      let raised = ref false in
      (try ignore (f_jit x8) with Invalid_argument _ -> raised := true);
      is_true !raised

let test_jit_chain () =
  match get_cpu_device () with
  | None -> ()
  | Some dev ->
      let f x =
        let y = T.add x (T.scalar T.float32 1.0) in
        T.mul y (T.scalar T.float32 2.0)
      in
      let f_jit = T.jit ~device:dev f in
      let x = T.scalar T.float32 4.0 in
      let _ = f_jit x in
      (* warmup *)
      let _ = f_jit x in
      (* capture *)
      let result = f_jit x in
      (* replay *)
      check_scalar ~eps "chain (4+1)*2" 10.0 (scalar_value result)

let test_jit_reduce () =
  match get_cpu_device () with
  | None -> ()
  | Some dev ->
      let f x = T.sum ~axes:[ 0 ] x in
      let f_jit = T.jit ~device:dev f in
      let x = T.full T.float32 [| 4 |] 3.0 in
      let _ = f_jit x in
      let _ = f_jit x in
      let result = f_jit x in
      check_scalar ~eps "sum [3;3;3;3]" 12.0 (scalar_value result)

(* ───── Test runner ───── *)

let () =
  run "JIT"
    [
      group "no device"
        [
          test "warmup is eager" test_jit_warmup_is_eager;
          test "warmup calls f" test_jit_warmup_calls_f;
          test "no device fails on capture" test_jit_no_device_fails;
        ];
      group "with device"
        [
          test "capture compiles" test_jit_capture_compiles;
          test "replay without recompile" test_jit_replay_no_recompile;
          test "replay different values" test_jit_replay_different_values;
          test "shape mismatch rejected" test_jit_shape_mismatch_rejected;
          test "chain (x+1)*2" test_jit_chain;
          test "reduce sum" test_jit_reduce;
        ];
    ]
