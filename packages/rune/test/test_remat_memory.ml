(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The device memory a compiled gradient needs when every layer is under
   [Rune.remat].

   On the CPU device a compiled call's inputs, constants and outputs alias host
   memory, so the device memory it allocates is the memory planner's arena: the
   most intermediates live at once. The arenas are shared by every program the
   process compiles and only grow, so this suite compiles one program and runs
   alone in its executable. *)

open Windtrap

let layers = 16
let batch = 256
let dim = 32
let hidden = 8 * dim
let f32 = Nx.float32

(* A residual MLP block. Its backward pass reads the [batch; hidden]
   pre-activation, eight times the size of the block's input. *)
let block (w1, w2) x = Nx.add x (Nx.matmul (Nx.relu (Nx.matmul x w1)) w2)

let device_bytes () =
  ignore (Rune.jit_stats ());
  Option.value ~default:0
    (Hashtbl.find_opt Tolk.Helpers.Global_counters.mem_used_per_device "CPU")

(* Without remat the gradient keeps every layer's pre-activation until the
   backward pass reaches it, [layers * batch * hidden] floats. With remat it
   keeps each layer's input, an eighth of that, and recomputes one layer at a
   time, so the peak stays under half of it. The layers are unrolled: a staged
   [Rune.scan] already recomputes each step in its backward loop, remat or
   not. *)
let test_remat_bounds_the_peak () =
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
  let peak = device_bytes () - before in
  let activations = layers * batch * hidden * 4 in
  is_true
    ~msg:
      (Printf.sprintf "peak %d bytes, under half of the %d activation bytes"
         peak activations)
    (peak < activations / 2)

let () =
  run "rune remat memory"
    [
      xfail
        ~reason:
          "jit shares the recomputed forward with the original, so the \
           activations stay live"
        (test "jit (grad) under remat bounds the peak"
           test_remat_bounds_the_peak);
    ]
