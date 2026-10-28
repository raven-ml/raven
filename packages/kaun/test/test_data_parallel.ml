(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Data-parallel training through [Rune.jit] over a batch split across two CPU
   devices, on kaun layers: a tiny causal attention + linear stack trained for a
   few SGD steps, the parameters entering from the host as a copy on each
   device. The loss trajectory and final weights must match the single-device
   step up to fp32 reduction order (the cross-device gradient allreduce reorders
   the batch sum). Momentum runs thread a real SGD state through the step — the
   state is a structure over the model's ([Vega.sgd_ptree model]) whose leaves
   are copies like the parameters — and the split trajectory must match the
   single-device one. Runs on two memory devices of one set, which nx.cpu
   computes; no pretrained weights involved. Every description is made where it
   is used, at the set its values lie on. *)

open Windtrap
open Kaun

module Two =
  (val Nx.devices
         (List.map
            (fun n -> Result.get_ok (Rig.memory_device n))
            [ "kaun-dp-0"; "kaun-dp-1" ]))

let batch = 8
let seq = 4
let dim = 8
let vocab = 11
let lr = 0.05

(* The model: pre-norm causal self-attention with a residual, then a linear head
   over the vocabulary. *)

type ('a, 'd) model = {
  ln : ('a, 'd) Layer_norm.t;
  attn : ('a, 'd) Attention.t;
  head : ('a, 'd) Linear.t;
}

module Model = struct
  type ('a, 'd) t = ('a, 'd) model

  let walk c { ln; attn; head } =
    let open Nx.Ptree.Walk in
    let ln = field c "ln" Layer_norm.walk ln in
    let attn = field c "attn" Attention.walk attn in
    let head = field c "head" Linear.walk head in
    { ln; attn; head }
end

let model () = Nx.Ptree.instantiate (module Model)

(* Deterministic init, no RNG: every run and both step implementations see the
   same weights and batch. *)

let fill i n =
  Array.init n (fun j -> sin (float_of_int ((i * 7919) + j)) *. 0.3)

(* The parameters at [at], whole on each of its devices. *)
let model_init at =
  let mat i r c =
    Nx.place at (Nx.create Nx.float32 [| r; c |] (fill i (r * c)))
  in
  let vec i n = Nx.place at (Nx.create Nx.float32 [| n |] (fill i n)) in
  let linear i = { Linear.w = mat i dim dim; b = Some (vec (i + 1) dim) } in
  {
    ln =
      {
        Layer_norm.gamma = Nx.ones_at at Nx.float32 [| dim |];
        beta = vec 2 dim;
      };
    attn =
      { Attention.q = linear 3; k = linear 5; v = linear 7; out = linear 9 };
    head = { Linear.w = mat 11 dim vocab; b = Some (vec 12 vocab) };
  }

let x_init at =
  Nx.place at
    (Nx.create Nx.float32 [| batch; seq; dim |] (fill 13 (batch * seq * dim)))

let tgt_init at =
  Nx.place at
    (Nx.create Nx.int64 [| batch; seq |]
       (Array.init (batch * seq) (fun i -> Int64.of_int (i * 5 mod vocab))))

let loss_fn x tgt m =
  let h =
    Nx.add x
      (Attention.apply ~head_dim:(dim / 2)
         ~mask:(Attention.causal_mask ~seq (Nx.placement x))
         m.attn (Layer_norm.apply m.ln x))
  in
  Loss.softmax_cross_entropy_sparse (Linear.apply m.head h) tgt

(* The step's state: the parameters and the optimizer state, a structure over
   the model's. A step reads the state, a copy on each device, and the batch,
   split on axis 0, as separate arguments. *)

type 'd state = {
  m : ('d Nx.float32_t, 'd) model;
  opt : (('d Nx.float32_t, 'd) model, 'd) Vega.sgd_state;
}

let state model =
  Nx.Ptree.v (fun c { m; opt } ->
      let open Nx.Ptree.Walk in
      let m = field c "m" (structure model) m in
      let opt = field c "opt" (structure (Vega.sgd_ptree model)) opt in
      { m; opt })

let step_signature model =
  let state = state model in
  Nx.Ptree.(state @-> tensor @-> tensor @-> returns (pair state tensor))

(* One SGD step: value_and_grad inside the compiled function, so over a split
   batch the gradients sum across devices before the update. With momentum 0 the
   velocity is never read; with momentum the state advances from the copied
   leaves on every device, identically. *)
let train_step model ~momentum { m; opt } x tgt =
  let loss, grads = Rune.value_and_grad model (loss_fn x tgt) m in
  let m, opt =
    Vega.sgd_step model ~lr:(Vega.lr lr) ~momentum opt ~params:m ~grads
  in
  ({ m; opt }, loss)

let trajectory ~steps ~at ~batch_at momentum =
  let model = model () in
  let step = Rune.jit (step_signature model) (train_step model ~momentum) in
  let m = model_init at in
  let s = ref { m; opt = Vega.sgd_init model m } in
  let x = x_init batch_at and tgt = tgt_init batch_at in
  Array.init steps (fun _ ->
      let next, loss = step !s x tgt in
      s := next;
      let floats =
        List.map
          (fun (Nx.P t) -> Nx.to_array (Nx.cast Nx.float64 t))
          (fst (Nx.Ptree.flatten model next.m))
      in
      (Nx.item [] loss, floats))

let run_both ~momentum ~steps =
  let mom = if momentum then 0.9 else 0.0 in
  ( trajectory ~steps ~at:Nx.Host.on ~batch_at:Nx.Host.on mom,
    trajectory ~steps ~at:Two.on ~batch_at:(Two.split ~axis:0) mom )

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
  List.iteri
    (fun i (a, b) ->
      equal
        ~msg:(Printf.sprintf "weight leaf %d: one device and two" (i + 1))
        (array (float 1e-6))
        a b)
    (List.combine m_jit m_pm)

let test_dp_momentum_matches_jit () =
  check_losses (run_both ~momentum:true ~steps:5)

(* Constructors and moves state where their values lie. *)

let placement =
  Testable.make ~pp:Nx.Placement.pp ~equal:(fun a b -> Nx.Placement.equal a b)

let at_of x = Nx.placement x

let test_drawn_at_the_key () =
  let key = Nx.Rng.key 3 in
  let host = Linear.init ~key ~inputs:3 ~outputs:2 in
  let placed =
    Linear.init ~key:(Nx.Rng.place Two.on key) ~inputs:3 ~outputs:2
  in
  equal ~msg:"w lies where the key lies" placement Two.on (at_of placed.w);
  equal ~msg:"w draws the host key's values" (array float_exact)
    (Nx.to_array host.w) (Nx.to_array placed.w)

let test_made_at_the_placement () =
  let cut = Two.split ~axis:0 in
  let ln = Layer_norm.init ~dim:4 cut in
  equal ~msg:"Layer_norm.gamma" placement cut (at_of ln.gamma);
  let rms = Rms_norm.init ~dim:4 cut in
  equal ~msg:"Rms_norm.gamma" placement cut (at_of rms.gamma);
  let bn, stats = Batch_norm.init ~features:4 cut in
  equal ~msg:"Batch_norm.beta" placement cut (at_of bn.beta);
  equal ~msg:"Batch_norm.Stats.var" placement cut (at_of stats.var);
  let kv = Two.split ~axis:1 in
  let c = Attention.Cache.make ~slots:4 ~kv_heads:2 ~head_dim:2 Nx.float32 kv in
  equal ~msg:"Cache.keys" placement kv (at_of c.keys);
  let mask = Attention.causal_mask ~seq:3 Two.on in
  equal ~msg:"causal_mask" placement Two.on (at_of mask)

let test_moved_whole () =
  let cut = Two.split ~axis:0 in
  let rope = Rope.place cut (Rope.make ~head_dim:4 ~context:8 ()) in
  let x = Nx.place Two.on (Nx.zeros Nx.float32 [| 1; 1; 2; 4 |]) in
  let pos = Nx.place Two.on (Nx.zeros Nx.int64 [| 1; 2 |]) in
  equal ~msg:"a rotation by a placed schedule lies whole on each device"
    placement Two.on
    (at_of (Rope.apply rope ~pos x));
  let index = Cache_index.place cut (Cache_index.rows ~context:4 [| 2; 2 |]) in
  equal ~msg:"Cache_index.positions" placement Two.on
    (at_of (Cache_index.positions index))

let tests =
  [
    group "placement"
      [
        test "a layer drawn from a placed key lies there" test_drawn_at_the_key;
        test "norms, caches and masks are made at their placement"
          test_made_at_the_placement;
        test "schedules and indices move whole onto each device"
          test_moved_whole;
      ];
    group "data-parallel"
      [
        test "SGD over a batch split in two follows one device"
          test_dp_matches_jit;
        test "replicated momentum state stays coherent"
          test_dp_momentum_matches_jit;
      ];
  ]

let () = exit (run "kaun data parallel" tests)
