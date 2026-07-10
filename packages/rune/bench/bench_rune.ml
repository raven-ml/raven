(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Rune (tape-based autodiff over Ptree structures) on representative Nx
   workloads. Eager Nx cases (no autodiff) are included as baselines to quantify
   the cost of gradient tracking itself. *)

(* MLP: 3 layers, 784 -> 256 -> 128 -> 10, batch 128, float32. *)

type mlp = {
  w1 : Nx.float32_t;
  b1 : Nx.float32_t;
  w2 : Nx.float32_t;
  b2 : Nx.float32_t;
  w3 : Nx.float32_t;
  b3 : Nx.float32_t;
}

module Mlp = struct
  type t = mlp

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p =
    {
      w1 = f p.w1;
      b1 = f p.b1;
      w2 = f p.w2;
      b2 = f p.b2;
      w3 = f p.w3;
      b3 = f p.b3;
    }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    {
      w1 = f p.w1 q.w1;
      b1 = f p.b1 q.b1;
      w2 = f p.w2 q.w2;
      b2 = f p.b2 q.b2;
      w3 = f p.w3 q.w3;
      b3 = f p.b3 q.b3;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) p =
    f p.w1;
    f p.b1;
    f p.w2;
    f p.b2;
    f p.w3;
    f p.b3
end

let batch = 128
let d_in = 784
let d_h1 = 256
let d_h2 = 128
let d_out = 10

let forward p x =
  let h1 = Nx.relu (Nx.add (Nx.matmul x p.w1) p.b1) in
  let h2 = Nx.relu (Nx.add (Nx.matmul h1 p.w2) p.b2) in
  Nx.add (Nx.matmul h2 p.w3) p.b3

let loss p x y = Nx.mean (Nx.square (Nx.sub (forward p x) y))

let init_mlp () =
  {
    w1 = Nx.randn Nx.float32 [| d_in; d_h1 |];
    b1 = Nx.zeros Nx.float32 [| d_h1 |];
    w2 = Nx.randn Nx.float32 [| d_h1; d_h2 |];
    b2 = Nx.zeros Nx.float32 [| d_h2 |];
    w3 = Nx.randn Nx.float32 [| d_h2; d_out |];
    b3 = Nx.zeros Nx.float32 [| d_out |];
  }

(* MLP value_and_grad: one optimizer-less training step. *)
let mlp_grad_benchmarks params x y =
  let f p = loss p x y in
  [
    Thumper.bench "mlp forward (nx eager)" (fun () -> loss params x y);
    Thumper.bench ~tags:[ "lab" ] "mlp value_and_grad" (fun () ->
        Rune.value_and_grad (module Mlp) f params);
  ]

(* MLP jvp: forward-mode directional derivative of the loss. *)
let mlp_jvp_benchmarks params x y =
  let tangents = Mlp.map (fun t -> Nx.ones_like t) params in
  let f p = loss p x y in
  [
    Thumper.bench "mlp jvp" (fun () -> Rune.jvp (module Mlp) f params tangents);
  ]

(* vmap of per-sample grads, stacked along a new batch axis. *)
let vmap_benchmarks params x =
  let c = Nx.randn Nx.float32 [| d_in |] in
  let ew_loss xi = Nx.sum (Nx.square (Nx.sin (Nx.mul xi c))) in
  let per_sample_w3_loss xi w3 =
    Nx.sum (Nx.square (forward { params with w3 } xi))
  in
  [
    Thumper.bench "per-sample grads ew" (fun () ->
        Rune.vmap' (fun xi -> Rune.grad' ew_loss xi) x);
    Thumper.bench "per-sample mlp w3 grads" (fun () ->
        Rune.vmap'
          (fun xi -> Rune.grad' (fun w3 -> per_sample_w3_loss xi w3) params.w3)
          x);
  ]

(* Deep chain: 100 sequential elementwise ops on a small tensor, so per-op
   handler/tape overhead dominates over kernel time. The eager case is the
   no-autodiff floor: (grad - 2 * eager) / ops approximates the per-op cost of
   gradient tracking. *)
let chain_ops = 100

let chain x =
  let t = ref x in
  for _ = 1 to chain_ops do
    t := Nx.sin !t
  done;
  Nx.sum !t

let chain_benchmarks x0 =
  [
    Thumper.bench "chain fwd (nx eager)" (fun () -> chain x0);
    Thumper.bench ~tags:[ "lab" ] "chain grad" (fun () -> Rune.grad' chain x0);
  ]

(* Jit: compiled execution of the same computations. Compilation — tracing plus
   kernel build — is hoisted into [setup], which builds the jitted closure and
   calls it once, so the timed region replays the compiled program only.
   [eager run mlp] is the same forward pass without jit, the no-jit baseline. *)
let jit_benchmarks params x x0 =
  [
    Thumper.bench_with_setup ~tags:[ "lab" ]
      ~setup:(fun () ->
        let f = Rune.jit (module Mlp) (fun p -> forward p x) in
        ignore (Sys.opaque_identity (f params));
        f)
      "jit run mlp"
      (fun f -> f params);
    Thumper.bench "eager run mlp" (fun () -> forward params x);
    Thumper.bench_with_setup
      ~setup:(fun () ->
        let f = Rune.jit' chain in
        ignore (Sys.opaque_identity (f x0));
        f)
      "jit run chain"
      (fun f -> f x0);
  ]

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let params = init_mlp () in
  let x = Nx.randn Nx.float32 [| batch; d_in |] in
  let y = Nx.randn Nx.float32 [| batch; d_out |] in
  let x0 = Nx.randn Nx.float32 [| 64 |] in
  Thumper.run "rune"
    ~budgets:
      [
        Thumper.Budget.no_slower_than ~metric:Thumper.Metric.wall_time 0.05;
        Thumper.Budget.no_more_alloc_than 0.01;
      ]
    [
      Thumper.group "MlpGrad" (mlp_grad_benchmarks params x y);
      Thumper.group "MlpJvp" (mlp_jvp_benchmarks params x y);
      Thumper.group "PerSampleGrads" (vmap_benchmarks params x);
      Thumper.group "DeepChain" (chain_benchmarks x0);
      Thumper.group "Jit" (jit_benchmarks params x x0);
    ]
