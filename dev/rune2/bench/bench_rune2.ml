(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rune2's costs, each row a call a user makes. Every input is a host value
   and every row waits for what it computes: an operation over values of every
   set builds a formula and computes nothing, so a row over them would time
   the formula. The rows that share a workload with packages/rune's suite
   keep its code and id. *)

module Rune = Rune2

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
   step's update of donated state, written in place; and compiled operations
   of the index family, an FFT and a matrix product. *)
let n = 4096

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

let jit_rows () =
  let x = x64 () and a = one () and b = one () in
  let xs = floats n and ps = positions n n and g = floats n in
  let into = floats 1024 and us = floats 16 and qs = positions 16 1024 in
  let zs = host (Nx.cast Nx.complex64 xs) in
  let m = Nx.reshape [| 64; 64 |] xs in
  [
    warm "add-1-hit" (fun () -> Rune.jit (pair_fn ()) Nx.add) (fun f -> synced (f a b));
    warm "jit run chain" (fun () -> Rune.jit' chain_sum) (fun f -> synced (f x));
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
    warm "sort-4096" (fun () -> Rune.jit' (fun x -> fst (Nx.sort x))) (fun f -> synced (f xs));
    warm "fft-4096" (fun () -> Rune.jit' (fun x -> Nx.fft x)) (fun f -> synced (f zs));
    warm "matmul-64" (fun () -> Rune.jit (pair_fn ()) Nx.matmul) (fun f -> synced (f m m));
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

let lane_rows () =
  let a =
    host
      (Nx.init f32 [| lanes |] (fun i ->
           0.5 *. (20_000. ** (Float.of_int i.(0) /. Float.of_int (lanes - 1)))))
  in
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
          ~zero:(host (Nx.scalar f32 0.))
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

let jit_loop_rows () =
  let state k = normal k [| 64 |] in
  let s = { lx = state 13; ly = state 14; lz = state 15 } in
  [
    warm "lorenz-scan-n100-replay"
      (fun () -> Rune.jit Nx.Ptree.(lorenz_t @-> returns tensor) (lorenz_scan 100))
      (fun g -> synced (g s));
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
         Thumper.group "Roots" (root_rows ());
         Thumper.group "Probes" (probe_rows ());
       ])
