(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* rune2's costs per operation over one-element float32 values, where the
   overhead of tracing and evaluating dominates kernel time. Each row's time
   divided by its operations is the cost per operation. *)

module Rune = Rune2

let f32 = Nx.float32
let ops = 100

let chain x =
  let t = ref x in
  for _ = 1 to ops do
    t := Nx.sin !t
  done;
  !t

let one () = Nx.copy (Nx.Rng.uniform ~key:(Nx.Rng.key 0) f32 [| 1 |])

(* Trace: a step of [ops] sines traced over no row, which evaluates nothing. *)
let trace_rows () =
  let init = one () and none = Nx.zeros f32 [| 0; 1 |] in
  let step c _ = (chain c, ()) in
  let t = Nx.Ptree.tensor in
  [
    Thumper.bench "chain-100" (fun () ->
        Rune.scan t t Nx.Ptree.unit ~f:step ~init none);
  ]

(* Loops: a step of [ops] sines over [rows] rows, against the OCaml fold of the
   same step. *)
let rows = 1000

let loop_rows () =
  let init = one () and xs = Nx.copy (Nx.zeros f32 [| rows; 1 |]) in
  let step c _ = (chain c, ()) in
  let t = Nx.Ptree.tensor in
  [
    Thumper.bench "scan-eager-100x1000" (fun () ->
        Rune.scan t t Nx.Ptree.unit ~f:step ~init xs);
    Thumper.bench "fold-eager-100x1000" (fun () ->
        let c = ref init in
        for i = 0 to rows - 1 do
          c := fst (step !c (Nx.get [ i ] xs))
        done;
        !c);
  ]

(* Declined: [ops] adds of constants in a traced step, under reverse mode and in
   a map, which reach neither the tracer nor the map and compute at the call,
   against the same adds eagerly. *)
let adds a b =
  for _ = 1 to ops do
    ignore (Sys.opaque_identity (Nx.add a b))
  done

let declined_rows () =
  let a = one () and b = one () and xs = Nx.copy (Nx.zeros f32 [| 1; 1 |]) in
  let t = Nx.Ptree.tensor in
  [
    Thumper.bench "add-1" (fun () -> adds a b);
    Thumper.bench "add-1-under-grad" (fun () ->
        Rune.grad'
          (fun x ->
            adds a b;
            Nx.sum x)
          a);
    Thumper.bench "add-1-under-scan" (fun () ->
        Rune.scan t t Nx.Ptree.unit
          ~f:(fun c _ ->
            adds a b;
            (c, ()))
          ~init:a xs);
    Thumper.bench "add-1-under-vmap" (fun () ->
        Rune.vmap'
          (fun x ->
            adds a b;
            x)
          xs);
  ]

(* DeepChain: [ops] sines and a sum, the shape of rune's row, eagerly and under
   forward and reverse mode. *)
let deep_rows () =
  let x = one () in
  let v = Nx.copy (Nx.ones_like x) in
  let f x = Nx.sum (chain x) in
  [
    Thumper.bench "chain-fwd" (fun () -> f x);
    Thumper.bench "chain-jvp" (fun () -> Rune.jvp' f x v);
    Thumper.bench "chain-grad" (fun () -> Rune.grad' f x);
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

let relu x = Nx.maximum x (Nx.zeros_like x)

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
    Thumper.bench "mlp-forward" (fun () -> loss params);
    Thumper.bench "mlp-value-and-grad" (fun () ->
        Rune.value_and_grad mlp loss params);
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
    Thumper.bench "ew" (fun () -> Rune.vmap' (Rune.grad' ew) x);
    Thumper.bench "mlp-w3" (fun () ->
        Rune.vmap' (fun xi -> Rune.grad' (w3_loss xi) params.w3) x);
  ]

(* Jacobians: of a chain of [ops] sines over 64 float32 elements, each one map
   of forward or reverse mode over the elements. *)
let jacobian_rows () =
  let x =
    Nx.place Nx.Host.on (Nx.Rng.uniform ~key:(Nx.Rng.key 12) f32 [| 64 |])
  in
  [
    Thumper.bench "jacfwd-64" (fun () -> Rune.jacfwd' chain x);
    Thumper.bench "jacrev-64" (fun () -> Rune.jacrev' chain x);
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
  [ Thumper.bench "leapfrog" (fun () -> leapfrog q0 p0) ]

let () =
  exit
    (Thumper.run "rune2"
       [
         Thumper.group "Trace" (trace_rows ());
         Thumper.group "Loops" (loop_rows ());
         Thumper.group "Declined" (declined_rows ());
         Thumper.group "DeepChain" (deep_rows ());
         Thumper.group "MlpGrad" (mlp_rows ());
         Thumper.group "PerSampleGrads" (per_sample_rows ());
         Thumper.group "Jacobians" (jacobian_rows ());
         Thumper.group "Hmc" (hmc_rows ());
       ])
