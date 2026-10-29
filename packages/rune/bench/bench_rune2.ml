(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rune's costs, each row a call a user makes. Every input is a host value
   and every row waits for what it computes: an operation over values of every
   set builds a formula and computes nothing, so a row over them would time
   the formula. The rows that share a workload with the former rune's suite
   keep its code and id. *)

let f32 = Nx.float32
let ops = 100

let chain x =
  let t = ref x in
  for _ = 1 to ops do
    t := Nx.sin !t
  done;
  !t

(* rune's chain: [ops] sines and their sum. *)
let chain_sum x = Nx.sum (chain x)

(* Values placed on the host, which compute where they are made and are read
   from their memory. *)
let host x = Nx.place Nx.Host.on x
let one () = host (Nx.Rng.uniform ~key:(Nx.Rng.key 0) f32 [| 1 |])
let floats k = host (Nx.Rng.uniform ~key:(Nx.Rng.key 1) f32 [| k |])

(* rune's chain input: 64 elements. *)
let x64 () = host (Nx.Rng.uniform ~key:(Nx.Rng.key 12) f32 [| 64 |])

(* [y] once the work that computes it is done. *)
let synced y =
  Nx.sync y;
  y

let synced2 (y, z) = (synced y, synced z)

(* [y] once the work that computes each of its arrays is done. *)
let synced_all s y =
  Nx.Ptree.fold s (fun _ x () -> Nx.sync x) y ();
  y

let t = Nx.Ptree.tensor

(* Trace: a step of [ops] sines traced over no row, which evaluates nothing. *)
let trace_rows () =
  let init = one () and none = host (Nx.zeros f32 [| 0; 1 |]) in
  let step c _ = (chain c, ()) in
  [
    Thumper.bench "chain-100" (fun () ->
        synced (fst (Rune.scan t t Nx.Ptree.unit ~f:step ~init none)));
  ]

(* Loops: a step of [ops] sines over [rows] rows, eagerly and under forward
   and reverse mode. A cost per operation or per row shows at 100 rows as at
   1000, and one quadratic in the rows a hundredfold. *)
let rows = 100

let loop_rows () =
  let init = one () and xs = host (Nx.zeros f32 [| rows; 1 |]) in
  let step c _ = (chain c, ()) in
  let scan c = fst (Rune.scan t t Nx.Ptree.unit ~f:step ~init:c xs) in
  let id = Printf.sprintf "%dx%d" ops rows in
  [
    Thumper.bench ("scan-" ^ id) (fun () -> synced (scan init));
    (* One scan of the step's derivative, and a scan stacking its residuals
       then a reversed scan of its transpose. *)
    Thumper.bench ("scan-jvp-" ^ id) (fun () ->
        synced2 (Rune.jvp' scan init init));
    Thumper.bench ("scan-grad-" ^ id) (fun () ->
        synced (Rune.grad' (fun c -> Nx.sum (scan c)) init));
  ]

(* Declined: [ops] adds of constants in a traced step, under reverse mode and in
   a map, which reach neither the tracer nor the map and compute at the call,
   against the same adds eagerly. Each add is waited for. *)
let adds a b =
  for _ = 1 to ops do
    Nx.sync (Nx.add a b)
  done

let declined_rows () =
  let a = one () and b = one () and xs = host (Nx.zeros f32 [| 1; 1 |]) in
  [
    Thumper.bench "add-1-nx-eager" (fun () -> adds a b);
    Thumper.bench "add-1-under-grad" (fun () ->
        synced
          (Rune.grad'
             (fun x ->
               adds a b;
               Nx.sum x)
             a));
    Thumper.bench "add-1-under-scan" (fun () ->
        synced
          (fst
             (Rune.scan t t Nx.Ptree.unit
                ~f:(fun c _ ->
                  adds a b;
                  (c, ()))
                ~init:a xs)));
    Thumper.bench "add-1-under-vmap" (fun () ->
        synced
          (Rune.vmap'
             (fun x ->
               adds a b;
               x)
             xs));
  ]

(* DeepChain: rune's chain over 64 elements, eagerly and under forward and
   reverse mode. *)
let deep_rows () =
  let x = x64 () in
  let v = host (Nx.ones_like x) in
  [
    Thumper.bench "chain-fwd-nx-eager" (fun () -> synced (chain_sum x));
    Thumper.bench "chain-jvp" (fun () -> synced2 (Rune.jvp' chain_sum x v));
    Thumper.bench "chain-grad" (fun () -> synced (Rune.grad' chain_sum x));
  ]

(* Remat: DeepChain's gradient through a remat, whose backward pass computes the
   chain again. *)
let remat_rows () =
  let x = x64 () in
  let f = Rune.remat Nx.Ptree.(tensor @-> returns tensor) chain in
  [
    Thumper.bench "chain-grad" (fun () ->
        synced (Rune.grad' (fun x -> Nx.sum (f x)) x));
  ]

(* MlpGrad: one step's loss and gradient of a three-layer perceptron, 784 to 256
   to 128 to 10, a batch of 128, float32, against its forward pass. *)
type 'a mlp = { w1 : 'a; b1 : 'a; w2 : 'a; b2 : 'a; w3 : 'a; b3 : 'a }

module Mlp = struct
  type ('a, _) t = 'a mlp

  let walk c { w1; b1; w2; b2; w3; b3 } =
    let open Nx.Ptree.Walk in
    let w1 = field c "w1" leaf w1 in
    let b1 = field c "b1" leaf b1 in
    let w2 = field c "w2" leaf w2 in
    let b2 = field c "b2" leaf b2 in
    let w3 = field c "w3" leaf w3 in
    let b3 = field c "b3" leaf b3 in
    { w1; b1; w2; b2; w3; b3 }
end

let mlp = Nx.Ptree.instantiate (module Mlp)

(* Draws held in host memory, so that each use reads them. *)
let normal k shape =
  Nx.place Nx.Host.on (Nx.Rng.normal ~key:(Nx.Rng.key k) f32 shape)

(* rune's relu. *)
let relu x =
  let zero = Nx.scalar_like x 0. in
  Nx.where (Nx.less_equal x zero) zero x

let forward p x =
  let h1 = relu (Nx.add (Nx.matmul x p.w1) p.b1) in
  let h2 = relu (Nx.add (Nx.matmul h1 p.w2) p.b2) in
  Nx.add (Nx.matmul h2 p.w3) p.b3

let mlp_params () =
  let zeros n = Nx.place Nx.Host.on (Nx.zeros f32 [| n |]) in
  {
    w1 = normal 1 [| 784; 256 |];
    b1 = zeros 256;
    w2 = normal 2 [| 256; 128 |];
    b2 = zeros 128;
    w3 = normal 3 [| 128; 10 |];
    b3 = zeros 10;
  }

let mlp_rows () =
  let params = mlp_params () in
  let x = normal 4 [| 128; 784 |] and y = normal 5 [| 128; 10 |] in
  let loss p = Nx.mean (Nx.square (Nx.sub (forward p x) y)) in
  [
    Thumper.bench "mlp forward (nx eager)" (fun () -> synced (loss params));
    Thumper.bench "mlp value_and_grad" (fun () ->
        let l, g = Rune.value_and_grad mlp loss params in
        (synced l, synced_all mlp g));
  ]

(* PerSampleGrads: a gradient per example of a batch of 128, one map of reverse
   mode: of an elementwise loss of the example, and of the perceptron's loss of
   the example with respect to its last layer's weights. *)
let per_sample_rows () =
  let params = mlp_params () in
  let x = normal 10 [| 128; 784 |] and c = normal 11 [| 784 |] in
  let ew xi = Nx.sum (Nx.square (Nx.sin (Nx.mul xi c))) in
  let w3_loss xi w3 =
    Nx.sum (Nx.square (forward { params with w3 } (Nx.reshape [| 1; 784 |] xi)))
  in
  [
    Thumper.bench "per-sample grads ew" (fun () ->
        synced (Rune.vmap' (Rune.grad' ew) x));
    Thumper.bench "per-sample mlp w3 grads" (fun () ->
        synced (Rune.vmap' (fun xi -> Rune.grad' (w3_loss xi) params.w3) x));
  ]

(* Jacobians: of a chain of [ops] sines over 64 float32 elements, each one map
   of forward or reverse mode over the elements. *)
let jacobian_rows () =
  let x = x64 () in
  [
    Thumper.bench "jacfwd-64" (fun () -> synced (Rune.jacfwd' chain x));
    Thumper.bench "jacrev-64" (fun () -> synced (Rune.jacrev' chain x));
  ]

(* Hmc: one leapfrog trajectory of 20 steps on a Gaussian over 16 floats, as a
   sampler runs it eagerly, each step taking the log density's gradient. *)
let hmc_rows () =
  let mean = normal 6 [| 16 |] and precision = normal 7 [| 16 |] in
  let log_density q =
    Nx.mul_s (Nx.sum (Nx.mul precision (Nx.square (Nx.sub q mean)))) (-0.5)
  in
  let eps = 0.1 in
  let grad q = snd (Rune.value_and_grad' log_density q) in
  let leapfrog q p =
    let q = ref q and p = ref p and g = ref (grad q) in
    for _ = 1 to 20 do
      p := Nx.add !p (Nx.mul_s !g (eps /. 2.));
      q := Nx.add !q (Nx.mul_s !p eps);
      g := grad !q;
      p := Nx.add !p (Nx.mul_s !g (eps /. 2.))
    done;
    (!q, !p)
  in
  let q0 = normal 8 [| 16 |] and p0 = normal 9 [| 16 |] in
  [ Thumper.bench "leapfrog" (fun () -> synced2 (leapfrog q0 p0)) ]

(* Jit: a compiled call's hit path, and rune's chain compiled; a training
   step's update of donated state, written in place; the gradient of a
   compiled call; and compiled operations whose lowering a past fix made
   cheap: indexing, sorting, an FFT, products, dense linear algebra and
   transcendental functions. *)
let n = 4096
let f64 = Nx.float64

let positions k bound =
  host
    (Nx.cast Nx.int64
       (Nx.mul
          (Nx.Rng.uniform ~key:(Nx.Rng.key 2) f32 [| k |])
          (Nx.scalar f32 (Float.of_int bound))))

(* [warm make run] is a case of [run (make ())], the compiled function [make]
   gives compiled in the worker by one uncounted call: compiling may start
   domains, after which thumper cannot fork its workers. *)
let warm name make run =
  Thumper.bench_with_setup name
    ~setup:(fun () ->
      let f = make () in
      ignore (run f);
      f)
    run

let pair_fn () = Nx.Ptree.(tensor @-> tensor @-> returns tensor)
let split_fn () = Nx.Ptree.(tensor @-> returns (pair tensor tensor))

(* Uniform host values of [dt] in [[lo, hi)]. *)
let between dt k lo hi shape =
  host
    (Nx.add_s
       (Nx.mul_s (Nx.Rng.uniform ~key:(Nx.Rng.key k) dt shape) (hi -. lo))
       lo)

(* A two-layer perceptron's loss over a batch of 32 rows of 64 inputs, 128
   hidden units and 10 outputs (rune's workload). *)
let small_mlp = Nx.Ptree.(pair (pair tensor tensor) (pair tensor tensor))

let small_loss ((w1, b1), (w2, b2)) x =
  let h = Nx.tanh (Nx.add (Nx.matmul x w1) b1) in
  let y = Nx.add (Nx.matmul h w2) b2 in
  Nx.mean (Nx.mul y y)

let grad_of_jit () =
  let p =
    ( (between f32 30 (-0.1) 0.1 [| 64; 128 |], host (Nx.zeros f32 [| 128 |])),
      (between f32 31 (-0.1) 0.1 [| 128; 10 |], host (Nx.zeros f32 [| 10 |])) )
  and x = between f32 32 (-1.) 1. [| 32; 64 |] in
  warm "grad-of-jit-mlp-float32"
    (fun () ->
      let l =
        Rune.jit Nx.Ptree.(small_mlp @-> tensor @-> returns tensor) small_loss
      in
      fun p -> Rune.grad small_mlp (fun p -> l p x) p)
    (fun g -> synced_all small_mlp (g p))

(* [k] rows of 64 written into a donated table of 4096 rows, each call taking
   the last one's result. A store of 16 rows alone over the donated memory
   costs about the eager call's 2 us; a pass over the table costs 50 times
   that. 1024 rows, a cache's prefill, are past the masked form's work bound,
   and combine in the table's memory. Where [flat], the table is held flat
   and reshaped to rows around the scatter, which keeps it in place. *)
let donated_scatter ?(flat = false) k =
  let slots = 4096 and width = 64 in
  Thumper.bench_with_setup
    (Printf.sprintf "scatter-%d-rows-into-donated-%s%dx%d" k
       (if flat then "flat-" else "")
       slots width)
    ~setup:(fun () ->
      let write cache u i =
        let p = Nx.broadcast_to [| k; width |] (Nx.reshape [| k; 1 |] i) in
        if not flat then Nx.scatter ~axis:0 p u cache
        else
          let rows = Nx.reshape [| slots; width |] cache in
          Nx.reshape [| slots * width |] (Nx.scatter ~axis:0 p u rows)
      in
      let g =
        Rune.jit
          Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
          write
      in
      let u = between f32 33 (-1.) 1. [| k; width |] in
      let i =
        host
          (Nx.create Nx.int64 [| k |]
             (Array.init k (fun r -> Int64.of_int (r * (slots / k)))))
      in
      let shape = if flat then [| slots * width |] else [| slots; width |] in
      let table = host (Nx.zeros f32 shape) in
      (g, ref (g (Nx.donate table) u i), u, i))
    (fun (g, table, u, i) ->
      table := g (Nx.donate !table) u i;
      synced !table)

(* The log density of 8 Gaussian processes at 11 points and its gradient in
   each process's values and log length scale and amplitude, in float64: a
   Cholesky factor of each covariance, a triangular solve and the log of the
   factor's diagonal (rune's workload). *)
let gaussian () =
  let chains = 8 and m = 11 in
  let params = Nx.Ptree.(pair tensor tensor) in
  let pts =
    host (Nx.init f64 [| m |] (fun i -> Float.of_int (i.(0) - (m / 2))))
  in
  let sq =
    Nx.square (Nx.sub (Nx.reshape [| m; 1 |] pts) (Nx.reshape [| 1; m |] pts))
  in
  let jitter = Nx.mul_s (Nx.eye f64 m) 1e-6 in
  let density (h, f) =
    let scale i =
      Nx.reshape [| chains; 1; 1 |] (Nx.exp (Nx.slice [ Nx.A; Nx.I i ] h))
    in
    let rho = scale 0 and alpha = scale 1 in
    let cov =
      Nx.add jitter
        (Nx.mul (Nx.square alpha)
           (Nx.exp (Nx.div sq (Nx.mul_s (Nx.square rho) (-2.)))))
    in
    let l = Nx.cholesky cov in
    let w = Nx.solve_triangular l f in
    Nx.sub
      (Nx.mul_s (Nx.sum (Nx.square w)) (-0.5))
      (Nx.sum (Nx.log (Nx.diagonal l)))
  in
  let p =
    ( between f64 34 0. 1. [| chains; 2 |],
      between f64 35 (-1.) 1. [| chains; m |] )
  in
  let out = Nx.Ptree.pair Nx.Ptree.tensor params in
  warm
    (Printf.sprintf "gaussian-process-grad-float64-%dx%d" chains m)
    (fun () ->
      Rune.jit
        Nx.Ptree.(params @-> returns out)
        (Rune.value_and_grad params density))
    (fun g -> synced_all out (g p))

let jit_rows () =
  let x = x64 () and a = one () and b = one () in
  let xs = floats n and ps = positions n n and g = floats n in
  let into = floats 1024 and us = floats 16 and qs = positions 16 1024 in
  let zs = host (Nx.cast Nx.complex64 xs) in
  let m = Nx.reshape [| 64; 64 |] xs in
  let pm = between f32 36 (-1.) 1. [| 64; 64 |] in
  let table = between f32 37 (-1.) 1. [| 4096; 64 |]
  and tokens = positions 1024 4096 in
  let sa =
    host
      (Nx.add
         (between f64 38 0. 1. [| 32; 32 |])
         (Nx.mul_s (Nx.eye f64 32) 32.))
  and sb = between f64 39 0. 1. [| 32 |] in
  let sym =
    let a = between f32 40 (-1.) 1. [| 64; 8; 8 |] in
    host (Nx.add a (Nx.matrix_transpose a))
  in
  let angles = between f32 41 (-30.) 30. [| 65536 |] in
  let bases = between f32 43 0.01 30. [| 65536 |] in
  let exponents = between f32 47 (-3.) 3. [| 65536 |] in
  let scaled = between f32 49 (-1.) 1. [| 100; 400 |] in
  [
    warm "add-1-hit"
      (fun () -> Rune.jit (pair_fn ()) Nx.add)
      (fun f -> synced (f a b));
    (* A step that calls OCaml between two additions: its program evaluated
       around the callback, each addition a compiled call. *)
    warm "callback-1-hit"
      (fun () ->
        let s = Nx.Ptree.(tensor @-> returns tensor) in
        Rune.jit (pair_fn ()) (fun a b ->
            let c = Nx.add a b in
            Nx.add (Rune.callback s ~like:c Fun.id c) b))
      (fun f -> synced (f a b));
    warm "jit run chain"
      (fun () -> Rune.jit' chain_sum)
      (fun f -> synced (f x));
    (* A root, which the compiler declines: the program is evaluated. *)
    warm "chain-100-rooted-hit"
      (fun () ->
        Rune.jit' (fun x ->
            let y = chain x in
            Rune.root Nx.Ptree.tensor
              ~residual:(fun r -> Nx.sub r y)
              (fun () -> y)))
      (fun f -> synced (f x));
    (* A call that stacks a map's lanes, under a map of other lanes than
       those it was first traced in. *)
    warm "lanes-3-after-2-hit"
      (fun () ->
        let a = Rune.axis () in
        let g = Rune.jit' (fun x -> Nx.add x (Nx.sum (Rune.lanes a x))) in
        ignore (Rune.vmap' ~axis:a g (host (Nx.ones f32 [| 2; 1 |])));
        (Rune.vmap' ~axis:a g, host (Nx.ones f32 [| 3; 1 |])))
      (fun (f, xs) -> synced (f xs));
    grad_of_jit ();
    Thumper.bench_with_setup "donated-step-4096"
      ~setup:(fun () ->
        let step =
          Rune.jit (pair_fn ()) (fun w g ->
              Nx.sub w (Nx.mul (Nx.scalar_like w 0.01) g))
        in
        let w = ref (step (Nx.donate (floats n)) g) in
        (step, w))
      (fun (step, w) ->
        w := step (Nx.donate !w) g;
        synced !w);
    warm "gather-4096"
      (fun () -> Rune.jit (pair_fn ()) (fun p x -> Nx.take p x))
      (fun f -> synced (f ps xs));
    warm "scatter-1024x16"
      (fun () ->
        Rune.jit
          Nx.Ptree.(tensor @-> tensor @-> tensor @-> returns tensor)
          (fun p u into -> Nx.scatter ~axis:0 p u into))
      (fun f -> synced (f qs us into));
    donated_scatter 16;
    (* An embedding's gradient: 1024 tokens' rows added into a table of
       zeros. *)
    warm "take-grad-1024-of-4096x64"
      (fun () ->
        Rune.jit (pair_fn ()) (fun i table ->
            Rune.grad'
              (fun t ->
                let e = Nx.take ~axis:0 i t in
                Nx.sum (Nx.mul e e))
              table))
      (fun f -> synced (f tokens table));
    donated_scatter 1024;
    donated_scatter ~flat:true 1024;
    (* A sort keeping its positions: keys and positions sorted together. *)
    warm "sort-with-positions-4096"
      (fun () -> Rune.jit (split_fn ()) (fun x -> Nx.sort x))
      (fun f -> synced2 (f xs));
    warm "fft-4096"
      (fun () -> Rune.jit' (fun x -> Nx.fft x))
      (fun f -> synced (f zs));
    warm "matmul-64"
      (fun () -> Rune.jit (pair_fn ()) Nx.matmul)
      (fun f -> synced (f m m));
    (* The gradient of 8 sines through a remat, compiled: the backward pass
       stores the remat's computed argument and computes the sines again. A
       longer chain takes too long to compile. *)
    warm "remat-chain-8-grad-hit"
      (fun () ->
        let sines x =
          List.fold_left (fun t _ -> Nx.sin t) x (List.init 8 Fun.id)
        in
        let f = Rune.remat Nx.Ptree.(tensor @-> returns tensor) sines in
        Rune.jit' (Rune.grad' (fun x -> Nx.sum (f (Nx.neg x)))))
      (fun g -> synced (g x));
    (* A product of a value's positive part, whose maximum the product reads
       once per column. *)
    warm "maximum-matmul-float32-64"
      (fun () ->
        Rune.jit (pair_fn ()) (fun x w ->
            Nx.matmul (Nx.maximum x (Nx.zeros_like x)) w))
      (fun f -> synced (f pm m));
    (* An LU factorization with partial pivoting and two triangular solves,
       each a loop of one step per row. *)
    warm "solve-float64-32"
      (fun () -> Rune.jit (pair_fn ()) Nx.solve)
      (fun f -> synced (f sa sb));
    (* 56 rounds of Jacobi rotations, held in a loop. *)
    warm "eigh-float32-64x8x8"
      (fun () -> Rune.jit (split_fn ()) (fun a -> Nx.eigh f32 a))
      (fun f -> synced2 (f sym));
    gaussian ();
    (* Sines of a buffer's angles in [[-30, 30]]; sines of angles a tanh
       bounds to [[-30, 30]], which need no long reduction, read net of the
       tanh row; and tanh, one exponential each. *)
    warm "sin-float32-64K"
      (fun () -> Rune.jit' Nx.sin)
      (fun f -> synced (f angles));
    warm "sin-bounded-float32-64K"
      (fun () -> Rune.jit' (fun x -> Nx.sin (Nx.mul_s (Nx.tanh x) 30.)))
      (fun f -> synced (f angles));
    warm "tanh-float32-64K"
      (fun () -> Rune.jit' Nx.tanh)
      (fun f -> synced (f angles));
    (* A running sum over the negated logs of uniform draws, as nested
       sampling's error estimate: its windows of 256 read each element up to
       256 times, which compute each draw and its log once when stored. *)
    warm "cumsum-neg-log-uniform-8x1024"
      (fun () ->
        ( Rune.jit
            Nx.Ptree.(v Nx.Rng.walk @-> returns tensor)
            (fun key ->
              let u = Nx.Rng.uniform ~key f64 [| 8; 1024 |] in
              Nx.cumsum ~axis:1 (Nx.neg (Nx.log u))),
          Nx.Rng.key 48 ))
      (fun (f, key) -> synced (f key));
    (* A running sum of a cheap value, its windows of 400 each computing it
       again at every trip when inlined: stored, it is computed once. *)
    warm "cumsum-scaled-float32-100x400"
      (fun () -> Rune.jit' (fun x -> Nx.cumsum ~axis:1 (Nx.mul_s x 2.)))
      (fun f -> synced (f scaled));
    (* Powers of bases in [[0.01, 30)] to exponents in [[-3, 3)]. *)
    warm "pow-float32-64K"
      (fun () -> Rune.jit (pair_fn ()) Nx.pow)
      (fun f -> synced (f bases exponents));
  ]

(* Loops whose lanes stop apart: Newton's iteration for one value's square root,
   to a relative residual, mapped over 256 lanes from 0.5 to 10,000, whose lanes
   stop after 4 to 10 trips, each held at its donor meanwhile (rune's
   workload). *)
let lanes = 256

let lane_newton a =
  Rune.iterate' ~max:64
    ~until:(fun x -> Nx.less (Nx.abs (Nx.sub (Nx.mul x x) a)) (Nx.mul_s a 1e-5))
    ~f:(fun x -> Nx.mul_s (Nx.add x (Nx.div a x)) 0.5)
    (Nx.add_s a 1.)

let lane_values () =
  let spread i = Float.of_int i.(0) /. Float.of_int (lanes - 1) in
  host (Nx.init f32 [| lanes |] (fun i -> 0.5 *. (20_000. ** spread i)))

let lane_rows () =
  let a = lane_values () in
  let mapped = Rune.vmap' lane_newton in
  [
    Thumper.bench (Printf.sprintf "vmap-newton-%d-lanes-eager" lanes) (fun () ->
        synced (mapped a));
  ]

(* Bodies: a scan in a custom_jvp rule under grad, inside a total's collect: the
   rule's program threads the total's running value through the scan's steps,
   each adding its row's sum (rune's workload, 64 x 16 rows). *)
let added = Rune.Total.make ()

let scanned_rule =
  Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
      let y =
        fst
          (Rune.scan'
             ~f:(fun c xi ->
               Rune.Total.add added (Nx.sum xi);
               (Nx.add c (Nx.sin xi), c))
             ~init:(Nx.zeros_like (Nx.get [ 0 ] x))
             x)
      in
      (y, fun dx -> Nx.sum ~axes:[ 0 ] (Nx.mul (Nx.cos x) dx)))

let grad_rule_in_scope x =
  Rune.grad'
    (fun x ->
      let y, _ =
        Rune.Total.collect added
          ~init:(host (Nx.scalar f32 0.))
          (fun () -> scanned_rule x)
      in
      Nx.sum y)
    x

let body_rows () =
  let rows = Nx.reshape [| 64; 16 |] (floats (64 * 16)) in
  [
    Thumper.bench "grad-scan-in-a-rule-in-a-total-scope-eager" (fun () ->
        synced (grad_rule_in_scope rows));
  ]

(* Totals: 64 lanes, each adding its row's sum and, on each of a scan's 16
   trips, the row's element, to a total collected around the map, which adds the
   lanes of each addition in turn. *)
let summed = Rune.Total.make ()

let lane_sums r =
  Rune.Total.add summed (Nx.sum r);
  fst
    (Rune.scan'
       ~f:(fun c xi ->
         Rune.Total.add summed xi;
         (Nx.add c xi, c))
       ~init:(Nx.zeros_like (Nx.get [ 0 ] r))
       r)

let total_rows () =
  let rows = Nx.reshape [| 64; 16 |] (floats (64 * 16)) in
  let around f =
    snd
      (Rune.Total.collect summed
         ~init:(host (Nx.scalar f32 0.))
         (fun () -> ignore (synced (f rows))))
  in
  [
    Thumper.bench "vmap-scan-64-lanes-eager" (fun () ->
        synced (around (Rune.vmap' lane_sums)));
    warm "vmap-scan-64-lanes-replay"
      (fun () -> Rune.jit' (Rune.vmap' lane_sums))
      (fun g -> synced (around g));
  ]

(* Roots: the gradient of a scan whose step holds a root, x² = a + r for each
   row r, its solve Newton's iteration and its linear solve the default dense
   one: the backward scan solves the transposed system once per row. *)
let root_rows () =
  let rs = floats 16 and a = host (Nx.full f32 [| 1 |] 2.) in
  let sqrt_root a =
    Rune.root Nx.Ptree.tensor
      ~residual:(fun x -> Nx.sub (Nx.mul x x) a)
      (fun () ->
        Rune.iterate' ~max:32
          ~until:(fun x ->
            Nx.less_s (Nx.max (Nx.abs (Nx.sub (Nx.mul x x) a))) 1e-5)
          ~f:(fun x -> Nx.mul_s (Nx.add x (Nx.div a x)) 0.5)
          (Nx.add_s a 1.))
  in
  let f a =
    let c, _ =
      Rune.scan' ~init:a
        ~f:(fun c r -> (sqrt_root (Nx.add c (Nx.reshape [| 1 |] r)), c))
        rs
    in
    Nx.sum c
  in
  [
    Thumper.bench "grad-root-in-scan-16-eager" (fun () ->
        synced (Rune.grad' f a));
    (* Compiled: each row's derivative solves the root's linear system. *)
    warm "grad-root-in-scan-16-replay"
      (fun () -> Rune.jit' (Rune.grad' f))
      (fun g -> synced (g a));
  ]

(* WarmStart: a compiled function's first call, its trace, lowering and link,
   with tolk's caches warm, as a fresh process's would be. *)
let warm_rows () =
  let x = one () in
  [
    Thumper.bench_with_setup "chain-100"
      ~setup:(fun () -> ignore (Rune.jit' chain x))
      (fun () -> synced (Rune.jit' chain x));
  ]

(* Custom: a custom_jvp call outside every transformation, the shape of a log
   density with its gradient: a rule of [ops] sines and a sum, whose tangent map
   pairs the tangent with a value the rule computes. The rule and its map are
   traced, then the rule's value evaluated. And a second derivative through such
   a rule. *)
let custom_rows () =
  let x = x64 () in
  let density =
    Rune.custom_jvp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
        let y = chain x in
        (Nx.sum y, fun dx -> Nx.sum (Nx.mul y dx)))
  in
  (* The same shape as a custom_vjp, under grad of grad: each order evaluates
     the residuals' code once. *)
  let rule =
    Rune.custom_vjp Nx.Ptree.tensor Nx.Ptree.tensor (fun x ->
        let y = chain x in
        (Nx.sum y, fun g -> Nx.mul g y))
  in
  let grad_grad x =
    Rune.grad' (fun x -> Nx.sum (Rune.grad' rule x)) x
  in
  [
    Thumper.bench "custom-jvp-call" (fun () -> synced (density x));
    Thumper.bench "custom-vjp-grad-grad" (fun () -> synced (grad_grad x));
  ]

(* JitFootprint and JitLoops: replays of compiled programs, each compiled in
   its setup's first call: [lorenz] an Euler integration of 64 states, its
   steps written out or as one scan; [rnn] an affine recurrence
   h <- x_t W + h U over a batch of 32 with a sum-of-squares loss, its value
   and its gradient. *)

type 'a lorenz = { lx : 'a; ly : 'a; lz : 'a }

module Lorenz = struct
  type ('a, _) t = 'a lorenz

  let walk c { lx; ly; lz } =
    let open Nx.Ptree.Walk in
    let lx = field c "x" leaf lx in
    let ly = field c "y" leaf ly in
    let lz = field c "z" leaf lz in
    { lx; ly; lz }
end

let lorenz_t = Nx.Ptree.instantiate (module Lorenz)

let lorenz_step { lx = x; ly = y; lz = z } =
  let dt = 0.0625 in
  let dx = Nx.mul_s (Nx.sub y x) 10. in
  let dy = Nx.sub (Nx.mul x (Nx.sub (Nx.scalar_like z 28.) z)) y in
  let dz = Nx.sub (Nx.mul x y) (Nx.mul_s z 2.5) in
  {
    lx = Nx.add x (Nx.mul_s dx dt);
    ly = Nx.add y (Nx.mul_s dy dt);
    lz = Nx.add z (Nx.mul_s dz dt);
  }

let lorenz_sum { lx; ly; lz } = Nx.sum (Nx.add (Nx.add lx ly) lz)

let lorenz n s =
  let rec go i s = if i = 0 then s else go (i - 1) (lorenz_step s) in
  lorenz_sum (go n s)

let lorenz_scan n s =
  let rows = Nx.zeros f32 [| n; 1 |] in
  let s, () =
    Rune.scan lorenz_t Nx.Ptree.tensor Nx.Ptree.unit
      ~f:(fun s _ -> (lorenz_step s, ()))
      ~init:s rows
  in
  lorenz_sum s

type 'a rnn = { w : 'a; u : 'a; h0 : 'a; xs : 'a list }

module Rnn = struct
  type ('a, _) t = 'a rnn

  let walk c { w; u; h0; xs } =
    let open Nx.Ptree.Walk in
    let w = field c "w" leaf w in
    let u = field c "u" leaf u in
    let h0 = field c "h0" leaf h0 in
    let xs = field c "xs" (list leaf) xs in
    { w; u; h0; xs }
end

let rnn_t = Nx.Ptree.instantiate (module Rnn)

let rnn_forward p =
  let step (h, loss) x =
    let h = Nx.add (Nx.matmul x p.w) (Nx.matmul h p.u) in
    (h, Nx.add loss (Nx.sum (Nx.square h)))
  in
  snd (List.fold_left step (p.h0, Nx.scalar_like p.h0 0.) p.xs)

let rnn_params horizon =
  let k = ref 20 in
  let draw shape =
    incr k;
    Nx.mul_s (normal !k shape) 0.1
  in
  {
    w = draw [| 32; 32 |];
    u = draw [| 32; 32 |];
    h0 = draw [| 32; 32 |];
    xs = List.init horizon (fun _ -> draw [| 32; 32 |]);
  }

let footprint_rows () =
  let state k = normal k [| 64 |] in
  let s = { lx = state 13; ly = state 14; lz = state 15 } in
  let lorenz_row name f =
    warm name
      (fun () -> Rune.jit Nx.Ptree.(lorenz_t @-> returns tensor) f)
      (fun g -> synced (g s))
  in
  let rnn_row name horizon f =
    let p = rnn_params horizon in
    warm name (fun () -> f ()) (fun g -> g p)
  in
  let fwd () =
    let g = Rune.jit Nx.Ptree.(rnn_t @-> returns tensor) rnn_forward in
    fun p -> synced (g p)
  in
  let grad () =
    let g =
      Rune.jit Nx.Ptree.(rnn_t @-> returns rnn_t) (Rune.grad rnn_t rnn_forward)
    in
    fun p -> synced_all rnn_t (g p)
  in
  [
    lorenz_row "lorenz n10 replay" (lorenz 10);
    lorenz_row "lorenz n100 replay" (lorenz 100);
    rnn_row "rnn-fwd h2 replay" 2 fwd;
    rnn_row "rnn-fwd h20 replay" 20 fwd;
    rnn_row "rnn-grad h20 replay" 20 grad;
  ]

(* Newton's iteration for the square roots of 1024 values in [[0.5, 4.5)],
   until every residual is small: the condition is tested in the loop, with
   no wait on the host per trip (rune's workload). *)
let newton a =
  Rune.iterate' ~max:64
    ~until:(fun x -> Nx.less_s (Nx.max (Nx.abs (Nx.sub (Nx.mul x x) a))) 1e-3)
    ~f:(fun x -> Nx.mul_s (Nx.add x (Nx.div a x)) 0.5)
    (Nx.add_s a 1.)

let jit_loop_rows () =
  let state k = normal k [| 64 |] in
  let s = { lx = state 13; ly = state 14; lz = state 15 } in
  let a = between f32 42 0.5 4.5 [| 1024 |] in
  let lanes_a = lane_values () in
  [
    warm "lorenz-scan-n100-replay"
      (fun () ->
        Rune.jit Nx.Ptree.(lorenz_t @-> returns tensor) (lorenz_scan 100))
      (fun g -> synced (g s));
    (* The scan's gradient: a known loop stacking residuals, then a reversed
       one. *)
    warm "lorenz-scan-grad-n100-replay"
      (fun () ->
        Rune.jit
          Nx.Ptree.(lorenz_t @-> returns lorenz_t)
          (Rune.grad lorenz_t (lorenz_scan 100)))
      (fun g -> synced_all lorenz_t (g s));
    warm "newton-1024-replay"
      (fun () -> Rune.jit' newton)
      (fun g -> synced (g a));
    (* Its gradient: the loop stacks its steps in room for 64, and the
       backward loop runs over the steps taken. *)
    warm "newton-grad-1024-replay"
      (fun () -> Rune.jit' (Rune.grad' (fun a -> Nx.sum (newton a))))
      (fun g -> synced (g a));
    (* The lanes stop apart: the loop runs while any lane runs on. *)
    warm
      (Printf.sprintf "vmap-newton-%d-lanes-replay" lanes)
      (fun () -> Rune.jit' (Rune.vmap' lane_newton))
      (fun g -> synced (g lanes_a));
  ]

(* TrainStep: one compiled training step of a perceptron 784 -> 256 -> 128 ->
   10 over a batch of 128, its loss's gradient and an Adam update of donated
   parameters and moments, the shape of kaun's training step: its words are the
   hit path's over 18 leaves. *)
type 'a adam = { params : 'a mlp; m : 'a mlp; v : 'a mlp }

module Adam = struct
  type ('a, _) t = 'a adam

  let walk c { params; m; v } =
    let open Nx.Ptree.Walk in
    let params = field c "params" Mlp.walk params in
    let m = field c "m" Mlp.walk m in
    let v = field c "v" Mlp.walk v in
    { params; m; v }
end

let adam = Nx.Ptree.instantiate (module Adam)

let map3 f a b c =
  {
    w1 = f a.w1 b.w1 c.w1;
    b1 = f a.b1 b.b1 c.b1;
    w2 = f a.w2 b.w2 c.w2;
    b2 = f a.b2 b.b2 c.b2;
    w3 = f a.w3 b.w3 c.w3;
    b3 = f a.b3 b.b3 c.b3;
  }

let train_rows () =
  let x = normal 17 [| 128; 784 |] and y = normal 18 [| 128; 10 |] in
  let zeros_like p =
    map3 (fun w _ _ -> Nx.place Nx.Host.on (Nx.zeros_like w)) p p p
  in
  let step =
    Rune.jit
      Nx.Ptree.(tensor @-> tensor @-> adam @-> returns (pair tensor adam))
      (fun x y { params; m; v } ->
        let loss p = Nx.mean (Nx.square (Nx.sub (forward p x) y)) in
        let l, g = Rune.value_and_grad mlp loss params in
        let lr = 1e-3 and b1 = 0.9 and b2 = 0.999 in
        let moment b m g = Nx.add (Nx.mul_s m b) (Nx.mul_s g (1. -. b)) in
        let m = map3 (fun m g _ -> moment b1 m g) m g g in
        let v = map3 (fun v g _ -> moment b2 v (Nx.square g)) v g g in
        let update p m v =
          Nx.sub p (Nx.div (Nx.mul_s m lr) (Nx.add_s (Nx.sqrt v) 1e-8))
        in
        let params = map3 update params m v in
        (l, { params; m; v }))
  in
  [
    Thumper.bench_with_setup "train-step"
      ~setup:(fun () ->
        let p = mlp_params () in
        let state = ref { params = p; m = zeros_like p; v = zeros_like p } in
        let _, s = step x y (Nx.Ptree.donate adam !state) in
        state := s;
        state)
      (fun state ->
        let l, s = step x y (Nx.Ptree.donate adam !state) in
        state := synced_all adam s;
        synced l);
  ]

(* Probes: [ops] emits of a pair with no collect open, which do nothing. *)
let probe_rows () =
  let p = Rune.Probe.make Nx.Ptree.(pair tensor tensor) in
  let v = (one (), one ()) in
  [
    Thumper.bench "emit-100-closed" (fun () ->
        for _ = 1 to ops do
          Rune.Probe.emit p v
        done);
  ]

(* A trial's setup compiles its programs: a cold tolk cache takes seconds for
   the largest, which the default 10 s deadline would kill. *)
let config = Thumper.Config.(default |> deadline 60.)

let () =
  exit
    (Thumper.run "rune2" ~config
       [
         Thumper.group "Trace" (trace_rows ());
         Thumper.group "Loops" (loop_rows () @ lane_rows ());
         Thumper.group "Declined" (declined_rows ());
         Thumper.group "DeepChain" (deep_rows ());
         Thumper.group "Remat" (remat_rows ());
         Thumper.group "Jit" (jit_rows ());
         Thumper.group "WarmStart" (warm_rows ());
         Thumper.group "MlpGrad" (mlp_rows ());
         Thumper.group "PerSampleGrads" (per_sample_rows ());
         Thumper.group "Jacobians" (jacobian_rows ());
         Thumper.group "Hmc" (hmc_rows ());
         Thumper.group "Custom" (custom_rows ());
         Thumper.group "JitLoops" (jit_loop_rows ());
         Thumper.group "JitFootprint" (footprint_rows ());
         Thumper.group "TrainStep" (train_rows ());
         Thumper.group "Bodies" (body_rows ());
         Thumper.group "Totals" (total_rows ());
         Thumper.group "Roots" (root_rows ());
         Thumper.group "Probes" (probe_rows ());
       ])
