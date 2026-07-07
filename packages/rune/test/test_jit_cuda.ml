(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Jit on the CUDA device: kernels compile through NVRTC and run on the GPU,
   data moves through copies. Skipped when no CUDA driver or GPU is present. *)

open Windtrap
open Rune_test_support.Support

let cuda_probe =
  lazy (match Tolk_cuda.create "CUDA" with _ -> Ok () | exception Failure msg -> Error msg)

let require_cuda () =
  match Lazy.force cuda_probe with
  | Ok () -> ()
  | Error msg -> skip ~reason:msg ()

let test_elementwise_on_cuda () =
  require_cuda ();
  let f x = Nx.tanh (Nx.add (Nx.mul x x) x) in
  let g = Rune.jit' ~device:"CUDA" f in
  let x = vec32 [| 1.0; -2.0; 0.5 |] in
  check_arr ~msg:"first call" (to_arr (f x)) (g x);
  check_arr ~msg:"replay" (to_arr (f x)) (g x)

let test_matmul_grad_on_cuda () =
  require_cuda ();
  let w = Nx.create f32 [| 3; 2 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let f x = Nx.sum (Nx.matmul x w) in
  let g = Rune.jit' ~device:"CUDA" (fun x -> Rune.grad' f x) in
  let x = Nx.create f32 [| 2; 3 |] [| 1.0; 0.0; -1.0; 0.5; 2.0; 1.0 |] in
  check_arr ~msg:"grad through cuda jit" (to_arr (Rune.grad' f x)) (g x)

let tests =
  [
    group "cuda device"
      [
        test "element-wise chain matches eager" test_elementwise_on_cuda;
        test "grad inside jit matches eager" test_matmul_grad_on_cuda;
      ];
  ]

let () = run "rune jit cuda" tests
