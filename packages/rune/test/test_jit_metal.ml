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

(* Multi-kernel compiled traces replay as batched device graphs: the kernels are
   recorded into an indirect command buffer on the first call and later calls
   patch the rebound buffers (fresh outputs, resident inputs) into it instead of
   launching each kernel individually. *)
let test_graph_batched_replay () =
  let w1 =
    Nx.create f32 [| 4; 4 |]
      (Array.init 16 (fun i -> (float_of_int (i mod 5) /. 4.0) -. 0.5))
  in
  let w2 =
    Nx.create f32 [| 4; 4 |]
      (Array.init 16 (fun i -> float_of_int (i mod 3) -. 1.0))
  in
  let f x = Nx.matmul (Nx.tanh (Nx.matmul x w1)) w2 in
  let g = Rune.jit' ~device:"METAL" f in
  let launches0 = !Tolk.Realize.graph_launches in
  List.iteri
    (fun i data ->
      let x = Nx.create f32 [| 2; 4 |] data in
      check_arr
        ~msg:(Printf.sprintf "call %d matches eager" (i + 1))
        (to_arr (f x))
        (g x))
    [
      Array.init 8 (fun i -> float_of_int i /. 8.0);
      Array.init 8 (fun i -> float_of_int (7 - i));
      Array.make 8 (-0.25);
    ];
  is_true ~msg:"every call dispatched a device graph"
    (!Tolk.Realize.graph_launches - launches0 >= 3);
  let x = Nx.create f32 [| 4; 4 |] (Array.init 16 (fun i -> float_of_int i)) in
  check_arr ~msg:"a resident output feeds the next call"
    (to_arr (f (f x)))
    (g (g x))

let tests =
  [
    group "metal device"
      [
        test "element-wise chain matches eager" test_elementwise_on_metal;
        test "grad inside jit matches eager" test_matmul_grad_on_metal;
        test "multi-kernel traces replay as device graphs"
          test_graph_batched_replay;
      ];
  ]

let () = run "rune jit metal" tests
