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

(* Jit footprint: the trace-plus-compile cost of a first Rune.jit call on the
   campaign's compile-heavy workloads, and the steady-state replay cost.

   The workloads mirror the tolk-direct compile graphs so the first-call number
   sits alongside tolk's per-stage totals: [elementwise] a+b*c (one kernel, the
   control), [lorenz] an Euler fold (one fused kernel of ~9 ops/step), and [rnn]
   an affine recurrence h <- x_t@W + h@U with a sum-of-squares loss (one kernel
   per step), forward and — the real user-shaped case, which exists only on the
   rune side — reverse through [Rune.grad].

   Two measurements. [replay] compiles once in setup and times one replay call.
   [first-call] builds a fresh jit and drives the one call that traces, lowers,
   and compiles. Repeating an identical compile in a live process is served by
   tolk's process-global program cache (not reachable from here to clear), so
   the first-call gate cases scale the output by a distinct constant each
   iteration: a fresh kernel body, a genuine recompile. That perturbation reaches
   a single fused kernel but not a matmul kernel's body, so the absolute cold
   cost of the matmul-heavy sizes is taken process-isolated through [--cold]
   instead, one fresh process per measurement. *)

type ew = { a : Nx.float32_t; b : Nx.float32_t; c : Nx.float32_t }

module Ew = struct
  type t = ew

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p =
    { a = f p.a; b = f p.b; c = f p.c }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) x y =
    { a = f x.a y.a; b = f x.b y.b; c = f x.c y.c }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) p =
    f p.a;
    f p.b;
    f p.c
end

type lorenz_state = { x : Nx.float32_t; y : Nx.float32_t; z : Nx.float32_t }

module Lorenz = struct
  type t = lorenz_state

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p =
    { x = f p.x; y = f p.y; z = f p.z }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    { x = f a.x b.x; y = f a.y b.y; z = f a.z b.z }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) p =
    f p.x;
    f p.y;
    f p.z
end

type rnn_params = {
  w : Nx.float32_t;
  u : Nx.float32_t;
  h0 : Nx.float32_t;
  xs : Nx.float32_t list;
}

module Rnn = struct
  type t = rnn_params

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p =
    { w = f p.w; u = f p.u; h0 = f p.h0; xs = List.map f p.xs }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    {
      w = f a.w b.w;
      u = f a.u b.u;
      h0 = f a.h0 b.h0;
      xs = List.map2 f a.xs b.xs;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) p =
    f p.w;
    f p.u;
    f p.h0;
    List.iter f p.xs
end

let ew_dim = 256
let lorenz_sigma = 10.0
let lorenz_rho = 28.0
let lorenz_beta = 2.5
let lorenz_dt = 0.0625
let rnn_batch = 32
let rnn_in = 32
let rnn_hidden = 32

let ew_forward p = Nx.add p.a (Nx.mul p.b p.c)

let lorenz_step { x; y; z } =
  let dx = Nx.mul_s (Nx.sub y x) lorenz_sigma in
  let dy = Nx.sub (Nx.mul x (Nx.sub (Nx.scalar Nx.float32 lorenz_rho) z)) y in
  let dz = Nx.sub (Nx.mul x y) (Nx.mul_s z lorenz_beta) in
  {
    x = Nx.add x (Nx.mul_s dx lorenz_dt);
    y = Nx.add y (Nx.mul_s dy lorenz_dt);
    z = Nx.add z (Nx.mul_s dz lorenz_dt);
  }

let lorenz n p =
  let rec loop i s = if i = 0 then s else loop (i - 1) (lorenz_step s) in
  let { x; y; z } = loop n p in
  Nx.sum (Nx.add (Nx.add x y) z)

let rnn_forward p =
  let step (h, loss) x_t =
    let h = Nx.add (Nx.matmul x_t p.w) (Nx.matmul h p.u) in
    (h, Nx.add loss (Nx.sum (Nx.square h)))
  in
  let _, loss = List.fold_left step (p.h0, Nx.scalar Nx.float32 0.0) p.xs in
  loss

let rnn_grad p = Rune.grad (module Rnn) rnn_forward p

let init_ew () =
  {
    a = Nx.randn Nx.float32 [| ew_dim; ew_dim |];
    b = Nx.randn Nx.float32 [| ew_dim; ew_dim |];
    c = Nx.randn Nx.float32 [| ew_dim; ew_dim |];
  }

let init_lorenz () =
  {
    x = Nx.randn Nx.float32 [| 64 |];
    y = Nx.randn Nx.float32 [| 64 |];
    z = Nx.randn Nx.float32 [| 64 |];
  }

let init_rnn horizon =
  {
    w = Nx.randn Nx.float32 [| rnn_in; rnn_hidden |];
    u = Nx.randn Nx.float32 [| rnn_hidden; rnn_hidden |];
    h0 = Nx.randn Nx.float32 [| rnn_batch; rnn_hidden |];
    xs = List.init horizon (fun _ -> Nx.randn Nx.float32 [| rnn_batch; rnn_in |]);
  }

(* A distinct scalar per recompile — and, through a per-process seed, per
   process — so neither the in-process program cache nor the on-disk compile
   caches (keyed by a kernel's semantic key and source) can serve a first-call
   measurement with an earlier compile. The value is immaterial, only its
   uniqueness; kept small so the scaled output cannot overflow. *)
let recompile_seed = int_of_float (Unix.gettimeofday () *. 1e6) land 0xffff
let recompile_ctr = ref 0

let fresh_scale () =
  incr recompile_ctr;
  1.0 +. (float_of_int ((recompile_seed lsl 14) + !recompile_ctr) *. 1e-9)

let jit_footprint_benchmarks ew_params lorenz_params rnn2 rnn10 rnn20 =
  let replay_ew () =
    let f = Rune.jit (module Ew) ew_forward in
    ignore (Sys.opaque_identity (f ew_params));
    f
  in
  let replay_lorenz n () =
    let f = Rune.jit (module Lorenz) (fun p -> lorenz n p) in
    ignore (Sys.opaque_identity (f lorenz_params));
    f
  in
  let replay_rnn_fwd params () =
    let f = Rune.jit (module Rnn) rnn_forward in
    ignore (Sys.opaque_identity (f params));
    f
  in
  let replay_rnn_grad params () =
    let f = Rune.jit2 (module Rnn) (module Rnn) rnn_grad in
    ignore (Sys.opaque_identity (f params));
    f
  in
  [
    Thumper.bench ~tags:[ "lab" ] "elementwise first-call" (fun () ->
        let s = fresh_scale () in
        let f = Rune.jit (module Ew) (fun p -> Nx.mul_s (ew_forward p) s) in
        Sys.opaque_identity (f ew_params));
    Thumper.bench ~tags:[ "lab" ] "lorenz n10 first-call" (fun () ->
        let s = fresh_scale () in
        let f = Rune.jit (module Lorenz) (fun p -> Nx.mul_s (lorenz 10 p) s) in
        Sys.opaque_identity (f lorenz_params));
    Thumper.bench_with_setup ~setup:replay_ew "elementwise replay" (fun f ->
        f ew_params);
    Thumper.bench_with_setup ~setup:(replay_lorenz 10) "lorenz n10 replay"
      (fun f -> f lorenz_params);
    Thumper.bench_with_setup ~setup:(replay_lorenz 50) "lorenz n50 replay"
      (fun f -> f lorenz_params);
    Thumper.bench_with_setup ~setup:(replay_lorenz 100) "lorenz n100 replay"
      (fun f -> f lorenz_params);
    Thumper.bench_with_setup ~setup:(replay_rnn_fwd rnn2) "rnn-fwd h2 replay"
      (fun f -> f rnn2);
    Thumper.bench_with_setup ~setup:(replay_rnn_fwd rnn10) "rnn-fwd h10 replay"
      (fun f -> f rnn10);
    Thumper.bench_with_setup ~setup:(replay_rnn_fwd rnn20) "rnn-fwd h20 replay"
      (fun f -> f rnn20);
    Thumper.bench_with_setup ~setup:(replay_rnn_grad rnn2) "rnn-grad h2 replay"
      (fun f -> f rnn2);
    Thumper.bench_with_setup ~setup:(replay_rnn_grad rnn10)
      "rnn-grad h10 replay" (fun f -> f rnn10);
    Thumper.bench_with_setup ~setup:(replay_rnn_grad rnn20)
      "rnn-grad h20 replay" (fun f -> f rnn20);
  ]

(* Process-isolated cold compile: one fresh jit of the given workload, timed by
   wall clock (the compile shells out to the kernel compiler). Driven one fresh
   process per call so the program cache starts empty. *)
let cold_compile spec =
  let wall f =
    let t0 = Unix.gettimeofday () in
    f ();
    (Unix.gettimeofday () -. t0) *. 1000.
  in
  let ms =
    match spec with
    | [ "ew" ] ->
        let p = init_ew () in
        wall (fun () ->
            let f = Rune.jit (module Ew) ew_forward in
            ignore (Sys.opaque_identity (f p)))
    | [ "lorenz"; n ] ->
        let n = int_of_string n in
        let p = init_lorenz () in
        wall (fun () ->
            let f = Rune.jit (module Lorenz) (fun q -> lorenz n q) in
            ignore (Sys.opaque_identity (f p)))
    | [ "rnnfwd"; h ] ->
        let p = init_rnn (int_of_string h) in
        wall (fun () ->
            let f = Rune.jit (module Rnn) rnn_forward in
            ignore (Sys.opaque_identity (f p)))
    | [ "rnngrad"; h ] ->
        let p = init_rnn (int_of_string h) in
        wall (fun () ->
            let f = Rune.jit2 (module Rnn) (module Rnn) rnn_grad in
            ignore (Sys.opaque_identity (f p)))
    | _ ->
        prerr_endline "usage: --cold (ew | lorenz N | rnnfwd H | rnngrad H)";
        exit 2
  in
  Printf.printf "%.3f\n" ms

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  match Array.to_list Sys.argv with
  | _ :: "--cold" :: rest -> cold_compile rest
  | _ ->
      let params = init_mlp () in
      let x = Nx.randn Nx.float32 [| batch; d_in |] in
      let y = Nx.randn Nx.float32 [| batch; d_out |] in
      let x0 = Nx.randn Nx.float32 [| 64 |] in
      let ew_params = init_ew () in
      let lorenz_params = init_lorenz () in
      let rnn2 = init_rnn 2 in
      let rnn10 = init_rnn 10 in
      let rnn20 = init_rnn 20 in
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
          Thumper.group "JitFootprint"
            (jit_footprint_benchmarks ew_params lorenz_params rnn2 rnn10 rnn20);
        ]
