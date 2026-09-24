(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Data-parallel training through [Rune.pmap2] on kaun layers: a tiny causal
   attention + linear stack trained for a few SGD steps with parameters
   replicated and the batch sharded over two CPU devices. The loss trajectory
   and final weights must match the single-device [Rune.jit2] step up to fp32
   reduction order (the cross-device gradient allreduce reorders the batch sum).
   Momentum runs thread a real SGD state through the step — the state is a
   structure over the model's ([Vega.sgd_ptree model]) whose leaves replicate
   like the parameters — and the pmapped trajectory must match the jitted one.
   Runs on CPU device instances; no pretrained weights involved. *)

open Windtrap
open Kaun

let devs2 = [ "CPU:1"; "CPU:2" ]
let batch = 8
let seq = 4
let dim = 8
let vocab = 11
let lr = 0.05

(* The model: pre-norm causal self-attention with a residual, then a linear head
   over the vocabulary. *)

type 'a model = {
  ln : 'a Layer_norm.t;
  attn : 'a Attention.t;
  head : 'a Linear.t;
}

module Model = struct
  type 'a t = 'a model

  let walk c { ln; attn; head } =
    let open Nx.Ptree.Walk in
    let ln = field c "ln" Layer_norm.walk ln in
    let attn = field c "attn" Attention.walk attn in
    let head = field c "head" Linear.walk head in
    { ln; attn; head }
end

let model : Nx.float32_t model Nx.Ptree.t = Nx.Ptree.instantiate (module Model)

(* Deterministic init, no RNG: every run and both step implementations see the
   same weights and batch. *)

let fill i n =
  Array.init n (fun j -> sin (float_of_int ((i * 7919) + j)) *. 0.3)

let mat i r c = Nx.create Nx.float32 [| r; c |] (fill i (r * c))
let vec i n = Nx.create Nx.float32 [| n |] (fill i n)

let model_init () =
  let linear i = { Linear.w = mat i dim dim; b = Some (vec (i + 1) dim) } in
  {
    ln = { Layer_norm.gamma = Nx.ones Nx.float32 [| dim |]; beta = vec 2 dim };
    attn =
      { Attention.q = linear 3; k = linear 5; v = linear 7; out = linear 9 };
    head = { Linear.w = mat 11 dim vocab; b = Some (vec 12 vocab) };
  }

let x_init () =
  Nx.create Nx.float32 [| batch; seq; dim |] (fill 13 (batch * seq * dim))

let tgt_init () =
  Nx.create Nx.int32 [| batch; seq |]
    (Array.init (batch * seq) (fun i -> Int32.of_int (i * 5 mod vocab)))

let loss_fn x tgt m =
  let h =
    Nx.add x
      (Attention.apply ~head_dim:(dim / 2)
         ~mask:(Attention.causal_mask ~seq ())
         m.attn (Layer_norm.apply m.ln x))
  in
  Loss.softmax_cross_entropy_sparse (Linear.apply m.head h) tgt

(* Step structures for pmap2: the batch joins the parameters and the optimizer
   state as leaves so it can be sharded on axis 0 while everything else
   replicates. The state is walked by its own structure over the model's. *)

type step_in = {
  m : Nx.float32_t model;
  opt : Nx.float32_t model Vega.sgd_state;
  x : Nx.float32_t;
  tgt : (int32, Nx.int32_elt) Nx.t;
}

let step_in =
  Nx.Ptree.(
    iso
      (fun ((m, opt), (x, tgt)) -> { m; opt; x; tgt })
      (fun { m; opt; x; tgt } -> ((m, opt), (x, tgt)))
      (pair (pair model (Vega.sgd_ptree model)) (pair tensor tensor)))

type step_out = {
  m' : Nx.float32_t model;
  opt' : Nx.float32_t model Vega.sgd_state;
  loss : Nx.float32_t;
}

let step_out =
  Nx.Ptree.(
    iso
      (fun ((m', opt'), loss) -> { m'; opt'; loss })
      (fun { m'; opt'; loss } -> ((m', opt'), loss))
      (pair (pair model (Vega.sgd_ptree model)) tensor))

(* One SGD step: value_and_grad inside the (jitted or pmapped) function, so
   under pmap the gradients allreduce across devices before the update. With
   momentum 0 the velocity is never read; with momentum the state advances from
   the replicated leaves on every device, identically. *)
let train_step ~momentum { m; opt; x; tgt } =
  let loss, grads = Rune.value_and_grad model (loss_fn x tgt) m in
  let m', opt' =
    Vega.sgd_step model ~lr:(Vega.lr lr) ~momentum opt ~params:m ~grads
  in
  { m'; opt'; loss }

let init ~momentum:_ =
  let m = model_init () in
  let opt = Vega.sgd_init model m in
  { m; opt; x = x_init (); tgt = tgt_init () }

(* One [in_axes] entry per leaf: everything replicated except the two batch
   leaves, sharded on axis 0. *)
let in_axes s =
  let n = Nx.Ptree.fold step_in (fun _ _ n -> n + 1) s 0 in
  List.init (n - 2) (fun _ -> None) @ [ Some 0; Some 0 ]

let trajectory ~momentum ~steps step0 =
  let s = ref (init ~momentum) in
  Array.init steps (fun _ ->
      let out = step0 !s in
      s := { !s with m = out.m'; opt = out.opt' };
      (Nx.item [] out.loss, out.m'))

let run_both ~momentum ~steps =
  let mom = if momentum then 0.9 else 0.0 in
  let jit =
    trajectory ~momentum ~steps
      (Rune.jit2 step_in step_out (train_step ~momentum:mom))
  in
  let pm =
    trajectory ~momentum ~steps
      (Rune.pmap2 ~devices:devs2
         ~in_axes:(in_axes (init ~momentum))
         step_in step_out (train_step ~momentum:mom))
  in
  (jit, pm)

let check_losses (jit, pm) =
  Array.iteri
    (fun i (l, _) ->
      equal
        ~msg:(Printf.sprintf "loss at step %d" (i + 1))
        (float 1e-6) l
        (fst pm.(i)))
    jit

let test_dp_matches_jit () =
  let ((jit, pm) as r) = run_both ~momentum:false ~steps:3 in
  check_losses r;
  (* Final weights leafwise within the fp32 allreduce band. *)
  let _, m_jit = jit.(2) and _, m_pm = pm.(2) in
  let leaf = ref 0 in
  ignore
    (Nx.Ptree.map2 model
       (fun (type a b) _ (a : (a, b) Nx.t) (b : (a, b) Nx.t) : (a, b) Nx.t ->
         incr leaf;
         let d =
           Nx.item [] (Nx.cast Nx.float64 (Nx.max (Nx.abs (Nx.sub a b))))
         in
         is_true
           ~msg:(Printf.sprintf "weight leaf %d: max |jit - pmap| = %g" !leaf d)
           (d <= 1e-6);
         a)
       m_jit m_pm)

let test_dp_momentum_matches_jit () =
  check_losses (run_both ~momentum:true ~steps:5)

let tests =
  [
    group "data-parallel"
      [
        test "2-device pmap SGD follows the jit trajectory" test_dp_matches_jit;
        test "replicated momentum state stays coherent"
          test_dp_momentum_matches_jit;
      ];
  ]

let () = run "kaun pmap dp" tests
