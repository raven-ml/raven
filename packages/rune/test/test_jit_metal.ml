(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jit on the Metal device: kernels compile and run on the GPU, data moves
   through copies. Compiled only on macOS. *)

open Windtrap
open Rune_test_support.Support

let test_elementwise_on_metal () =
  let f x = Nx.tanh (Nx.add (Nx.mul x x) x) in
  let g = Rune.jit' ~device:"METAL" f in
  let x = vec32 [| 1.0; -2.0; 0.5 |] in
  check_arr ~msg:"first call" (to_arr (f x)) (g x);
  check_arr ~msg:"replay" (to_arr (f x)) (g x)

let test_matmul_grad_on_metal () =
  let w = Nx.create f32 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let f x = Nx.sum (Nx.matmul x w) in
  let g = Rune.jit' ~device:"METAL" (fun x -> Rune.grad' f x) in
  let x = Nx.create f32 [| 2; 3 |] [| 1.0; 0.0; -1.0; 0.5; 2.0; 1.0 |] in
  check_arr ~msg:"grad through metal jit" (to_arr (Rune.grad' f x)) (g x)

(* Captured tensors are uploaded once per compilation and stay resident on the
   device: a later mutation of the capture is not observed (unlike on the CPU
   device, whose buffers alias the tensor's memory). *)
let test_capture_is_uploaded_once () =
  let c = vec32 [| 10.0; 20.0; 30.0 |] in
  let g = Rune.jit' ~device:"METAL" (fun x -> Nx.add x c) in
  check_arr ~msg:"initial capture" [| 11.0; 21.0; 31.0 |]
    (g (vec32 [| 1.0; 1.0; 1.0 |]));
  Nx.blit (vec32 [| 0.0; 0.0; 0.0 |]) c;
  check_arr ~msg:"capture stays at its compile-time value"
    [| 11.0; 21.0; 31.0 |]
    (g (vec32 [| 1.0; 1.0; 1.0 |]))

(* Captures the function assigns to are the exception: the writeback keeps the
   host value current and the buffer is re-uploaded on every call, so in-place
   state compounds across calls and eager mutations of it are observed. *)
let test_assigned_capture_carries_state () =
  let s = vec32 [| 1.0; 2.0 |] in
  let g =
    Rune.jit' ~device:"METAL" (fun x ->
        Nx.blit (Nx.add s x) s;
        Nx.mul_s s 10.0)
  in
  check_arr ~msg:"first call" [| 20.0; 30.0 |] (g (vec32 [| 1.0; 1.0 |]));
  check_arr ~msg:"state written back" [| 2.0; 3.0 |] s;
  check_arr ~msg:"second call compounds" [| 30.0; 40.0 |]
    (g (vec32 [| 1.0; 1.0 |]));
  Nx.blit (vec32 [| 0.0; 0.0 |]) s;
  check_arr ~msg:"eager mutation of assigned state is observed" [| 10.0; 10.0 |]
    (g (vec32 [| 1.0; 1.0 |]))

let tests =
  [
    group "metal device"
      [
        test "element-wise chain matches eager" test_elementwise_on_metal;
        test "grad inside jit matches eager" test_matmul_grad_on_metal;
        test "captures are uploaded once" test_capture_is_uploaded_once;
        test "assigned captures carry state" test_assigned_capture_carries_state;
      ];
  ]

let () = run "rune jit metal" tests
