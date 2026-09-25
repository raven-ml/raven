(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The device memory a compiled gradient needs when every layer is under
   [Rune.remat], on the device the [DEV] environment variable names.

   The figure is the device's allocated bytes across the first call, less the
   weights and input uploaded to it and the gradients computed on it: on the CPU
   device those alias host memory and count nothing. What remains is the memory
   planner's arena, the most intermediates live at once, and on Metal the
   compiled queue's command storage (33 KiB here). The arenas are shared by
   every program the process compiles and only grow, so this suite compiles one
   program and runs alone in its executable. *)

open Windtrap

let layers = 16
let batch = 256
let dim = 32
let hidden = 8 * dim
let f32 = Nx.float32

(* A residual MLP block. Its backward pass reads the [batch; hidden]
   pre-activation, eight times the size of the block's input. *)
let block (w1, w2) x = Nx.add x (Nx.matmul (Nx.relu (Nx.matmul x w1)) w2)
let device = Option.value (Sys.getenv_opt "DEV") ~default:"CPU"

let device_bytes () =
  ignore (Rune.jit_stats ());
  Tolk.Helpers.Global_counters.mem_used ~device ()

(* Without remat the gradient keeps every layer's pre-activation until the
   backward pass reaches it, [layers * batch * hidden] floats: the arena is
   5,013,504 bytes on the CPU device. With remat it keeps each layer's input, an
   eighth of that, and recomputes one layer at a time: 1,802,240 bytes. The
   layers are unrolled: a staged [Rune.scan] already recomputes each step in its
   backward loop, remat or not. *)
let test_remat_keeps_under_half_the_activations () =
  let s = Nx.Ptree.(pair tensor tensor) in
  let remat_block =
    Rune.remat Nx.Ptree.(s @-> tensor @-> returns tensor) block
  in
  let x = Nx.mul_s (Nx.Rng.normal (Nx.Rng.key 1) f32 [| batch; dim |]) 0.1 in
  let loss (w1s, w2s) =
    let h = ref x in
    for i = 0 to layers - 1 do
      h := remat_block (Nx.slice [ Nx.I i ] w1s, Nx.slice [ Nx.I i ] w2s) !h
    done;
    Nx.mean (Nx.mul !h !h)
  in
  let grad = Rune.jit Nx.Ptree.(s @-> returns s) (Rune.grad s loss) in
  let w1s =
    Nx.mul_s (Nx.Rng.normal (Nx.Rng.key 2) f32 [| layers; dim; hidden |]) 0.02
  in
  let w2s =
    Nx.mul_s (Nx.Rng.normal (Nx.Rng.key 3) f32 [| layers; hidden; dim |]) 0.02
  in
  let before = device_bytes () in
  let g1, _ = grad (w1s, w2s) in
  ignore (Nx.item [] (Nx.sum g1) : float);
  let transfers =
    if device = "CPU" then 0
    else (2 * (Nx.nbytes w1s + Nx.nbytes w2s)) + Nx.nbytes x
  in
  let arena = device_bytes () - before - transfers in
  let activations = layers * batch * hidden * 4 in
  is_true
    ~msg:
      (Printf.sprintf
         "an arena of %d bytes is under half of %d activation bytes" arena
         activations)
    (arena < activations / 2)

let () =
  run "rune remat memory"
    [
      test "jit (grad) under remat keeps under half the activations"
        test_remat_keeps_under_half_the_activations;
    ]
