(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A training step of a generative model of spike trains driven by a latent ODE,
   shaped as a user's: a gated neural ODE cell of three latents and [hidden]
   units, run by a scan over a burn-in and then over a horizon of drawn inputs;
   rates read out by a product to [outputs] channels; a loss of two entropic
   transport costs between the model's rates and a target's and between two of
   the model's draws, each [sinkhorn_iterations] Sinkhorn iterations in the log
   domain over the batch; its gradient through the scans; and an Adam update.
   The step draws from the key it takes and consumes the state it returns. *)

type size = { batch : int; hidden : int; outputs : int; horizon : int }

let latents = 3
let burnin = 16
let dt_over_tau = 0.1
let sinkhorn_iterations = 20
let temperature = 0.05
let regularisation = 1e-3
let learning_rate = 1e-3
let beta1 = 0.9
let beta2 = 0.999
let adam_eps = 1e-8

type 'a params = {
  g_a1 : 'a;
  g_bias1 : 'a;
  h_a1 : 'a;
  h_a2 : 'a;
  h_bias1 : 'a;
  h_bias2 : 'a;
  c_rates : 'a;
  bias_rates : 'a;
  us_scale : 'a;
  x0_scale : 'a;
}

let to_list p =
  [
    p.g_a1;
    p.g_bias1;
    p.h_a1;
    p.h_a2;
    p.h_bias1;
    p.h_bias2;
    p.c_rates;
    p.bias_rates;
    p.us_scale;
    p.x0_scale;
  ]

let of_list = function
  | [
      g_a1;
      g_bias1;
      h_a1;
      h_a2;
      h_bias1;
      h_bias2;
      c_rates;
      bias_rates;
      us_scale;
      x0_scale;
    ] ->
      {
        g_a1;
        g_bias1;
        h_a1;
        h_a2;
        h_bias1;
        h_bias2;
        c_rates;
        bias_rates;
        us_scale;
        x0_scale;
      }
  | _ -> invalid_arg "Lorenz.of_list: not ten tensors"

let params = Nx.Ptree.(iso of_list to_list (list tensor))

(* The parameters, the optimiser's two moments, and its step count. *)
let state = Nx.Ptree.(pair params (pair params (pair params tensor)))

let requad x =
  let x = Nx.mul_s x 10. in
  Nx.mul_s (Nx.add x (Nx.sqrt (Nx.add_s (Nx.square x) 4.))) 0.05

let cell p ?input x =
  let gate = Nx.sigmoid (Nx.add (Nx.matmul x p.g_a1) p.g_bias1) in
  let drive = requad (Nx.add (Nx.matmul x p.h_a1) p.h_bias1) in
  let h = Nx.sub (Nx.add (Nx.matmul drive p.h_a2) p.h_bias2) x in
  let h = match input with Some u -> Nx.add h u | None -> h in
  Nx.add x (Nx.mul_s (Nx.mul gate h) dt_over_tau)

(* The rates of [size.batch] draws over the horizon, one row of [horizon *
   outputs] a draw. *)
let rates size p key =
  let ks = Nx.Rng.split ~n:2 key in
  let x0 =
    Nx.mul
      (Nx.Rng.normal ks.(0) Nx.float32 [| size.batch; latents |])
      p.x0_scale
  in
  let us =
    Nx.mul
      (Nx.Rng.normal ks.(1) Nx.float32 [| size.horizon; size.batch; latents |])
      p.us_scale
  in
  let x0, _ =
    Rune.scan'
      ~f:(fun x t -> (cell p x, t))
      ~init:x0
      (Nx.zeros Nx.float32 [| burnin |])
  in
  let _, xs =
    Rune.scan'
      ~f:(fun x u ->
        let x = cell p ~input:u x in
        (x, x))
      ~init:x0 us
  in
  let r = Nx.sigmoid (Nx.add (Nx.matmul xs p.c_rates) p.bias_rates) in
  Nx.reshape [| size.batch; -1 |] (Nx.transpose ~axes:[ 1; 0; 2 ] r)

(* The squared distances between the rows of [f] and of [g], over their mean. *)
let cost f g =
  let norms x = Nx.sum ~axes:[ 1 ] (Nx.square x) in
  let c =
    Nx.add
      (Nx.sub
         (Nx.reshape [| -1; 1 |] (norms f))
         (Nx.mul_s (Nx.matmul f (Nx.matrix_transpose g)) 2.))
      (Nx.reshape [| 1; -1 |] (norms g))
  in
  Nx.div c (Nx.mean c)

(* The entropic transport cost between two uniform batches of rows. *)
let sinkhorn c =
  let n = Nx.dim 0 c in
  let log_weight = -.log (Float.of_int n) in
  let k = Nx.div_s (Nx.neg c) temperature in
  let u = ref (Nx.zeros Nx.float32 [| n; 1 |]) in
  let v = ref (Nx.zeros Nx.float32 [| 1; n |]) in
  for _ = 1 to sinkhorn_iterations do
    u :=
      Nx.neg
        (Nx.sub_s
           (Nx.logsumexp ~axes:[ 1 ] ~keepdims:true (Nx.add k !v))
           log_weight);
    v :=
      Nx.neg
        (Nx.sub_s
           (Nx.logsumexp ~axes:[ 0 ] ~keepdims:true (Nx.add k !u))
           log_weight)
  done;
  Nx.sum (Nx.mul (Nx.exp (Nx.add (Nx.add k !u) !v)) c)

let loss size target key p =
  let ks = Nx.Rng.split ~n:2 key in
  let model = rates size p ks.(0) and model' = rates size p ks.(1) in
  Nx.add
    (Nx.sub
       (sinkhorn (cost model target))
       (Nx.mul_s (sinkhorn (cost model model')) 0.5))
    (Nx.mul_s (Nx.mean (Nx.square p.h_a2)) regularisation)

let adam count p m v g =
  let m = Nx.add (Nx.mul_s m beta1) (Nx.mul_s g (1. -. beta1)) in
  let v = Nx.add (Nx.mul_s v beta2) (Nx.mul_s (Nx.square g) (1. -. beta2)) in
  let correction beta =
    Nx.sub (Nx.ones_like count) (Nx.exp (Nx.mul_s count (log beta)))
  in
  let m_hat = Nx.div m (correction beta1) in
  let v_hat = Nx.div v (correction beta2) in
  let p =
    Nx.sub p
      (Nx.mul_s
         (Nx.div m_hat (Nx.add_s (Nx.sqrt v_hat) adam_eps))
         learning_rate)
  in
  (p, (m, v))

let step size target key (p, (m, (v, count))) =
  let value, g = Rune.value_and_grad params (loss size target key) p in
  let count = Nx.add_s count 1. in
  let updated =
    List.map2
      (fun (p, m) (v, g) -> adam count p m v g)
      (List.combine (to_list p) (to_list m))
      (List.combine (to_list v) (to_list g))
  in
  let p = of_list (List.map fst updated) in
  let m = of_list (List.map (fun (_, (m, _)) -> m) updated) in
  let v = of_list (List.map (fun (_, (_, v)) -> v) updated) in
  (value, (p, (m, (v, count))))

let signature =
  Nx.Ptree.(
    tensor @-> Nx.Rng.ptree @-> consumes state @@ returns (pair tensor state))

(* The step compiled, and its target, key and first state, held at [placement],
   drawn from a fixed key. *)
let setup placement size =
  Nx.Rng.with_key (Nx.Rng.key 7) @@ fun () ->
  let normal shape scale = Nx.mul_s (Nx.randn Nx.float32 shape) scale in
  let n = latents and nh = size.hidden and o = size.outputs in
  let p =
    {
      g_a1 = Nx.zeros Nx.float32 [| n; n |];
      g_bias1 = Nx.full Nx.float32 [| n |] 3.;
      h_a1 = normal [| n; nh |] (1. /. sqrt (Float.of_int n));
      h_a2 = normal [| nh; n |] (0.5 /. sqrt (Float.of_int nh));
      h_bias1 = normal [| nh |] 1.;
      h_bias2 = Nx.zeros Nx.float32 [| n |];
      c_rates = normal [| n; o |] (0.6 /. sqrt (Float.of_int n));
      bias_rates = normal [| o |] 0.6;
      us_scale = Nx.ones Nx.float32 [| latents |];
      x0_scale = Nx.ones Nx.float32 [| n |];
    }
  in
  (* Every operand has storage of its own, as a trained model's has: a constant
     is one element seen at every index, which a compiled call folds. *)
  let p = of_list (List.map Nx.copy (to_list p)) in
  let zeros () =
    of_list (List.map (fun x -> Nx.copy (Nx.zeros_like x)) (to_list p))
  in
  let s = (p, (zeros (), (zeros (), Nx.copy (Nx.zeros Nx.float32 [||])))) in
  let target = Nx.rand Nx.float32 [| size.batch; size.horizon * o |] in
  let f = Rune.jit signature (step size) in
  let place t x = Nx.Ptree.place t placement x in
  ( f,
    place Nx.Ptree.tensor target,
    place Nx.Rng.ptree (Nx.Rng.key 2026),
    place state s )
