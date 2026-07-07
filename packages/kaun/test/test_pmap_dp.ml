(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Data-parallel training through [Rune.pmap2] on kaun layers: a tiny causal
   attention + linear stack trained for a few SGD steps with parameters
   replicated and the batch sharded over two CPU devices. The loss trajectory
   and final weights must match the single-device [Rune.jit2] step up to fp32
   reduction order (the cross-device gradient allreduce reorders the batch
   sum). A momentum run checks that replicated optimizer state threaded
   through the pmapped step stays coherent across devices. Runs on CPU device
   instances; no pretrained weights involved. *)

open Windtrap
open Kaun

let devs2 = [ "CPU:1"; "CPU:2" ]
let batch = 8
let seq = 4
let dim = 8
let vocab = 11
let lr = 0.05

(* The model: pre-norm causal self-attention with a residual, then a linear
   head over the vocabulary. *)

type model = { ln : Layer_norm.t; attn : Attention.t; head : Linear.t }

module Model = struct
  type t = model

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) m =
    {
      ln = Layer_norm.map f m.ln;
      attn = Attention.map f m.attn;
      head = Linear.map f m.head;
    }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    {
      ln = Layer_norm.map2 f a.ln b.ln;
      attn = Attention.map2 f a.attn b.attn;
      head = Linear.map2 f a.head b.head;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) m =
    Layer_norm.iter f m.ln;
    Attention.iter f m.attn;
    Linear.iter f m.head
end

(* Deterministic init, no RNG: every run and both step implementations see
   the same weights and batch. *)

let fill i n = Array.init n (fun j -> sin (float_of_int ((i * 7919) + j)) *. 0.3)
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
    (Array.init (batch * seq) (fun i -> Int32.of_int ((i * 5) mod vocab)))

let loss_fn x tgt m =
  let h =
    Nx.add x
      (Attention.apply ~num_heads:2 ~causal:true m.attn
         (Layer_norm.apply m.ln x))
  in
  Loss.softmax_cross_entropy_sparse (Linear.apply m.head h) tgt

(* Step structures for pmap2: the batch joins the parameters (and, for the
   momentum run, the velocity) as leaves so it can be sharded on axis 0 while
   everything else replicates. *)

type step_in = {
  m : model;
  v : model option;
  x : Nx.float32_t;
  tgt : (int32, Nx.int32_elt) Nx.t;
}

module Step_in = struct
  type t = step_in

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) s =
    {
      m = Model.map f s.m;
      v = Option.map (Model.map f) s.v;
      x = f s.x;
      tgt = f s.tgt;
    }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    {
      m = Model.map2 f a.m b.m;
      v =
        (match (a.v, b.v) with
        | Some va, Some vb -> Some (Model.map2 f va vb)
        | None, None -> None
        | _ -> invalid_arg "Step_in.map2: velocity mismatch");
      x = f a.x b.x;
      tgt = f a.tgt b.tgt;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) s =
    Model.iter f s.m;
    Option.iter (Model.iter f) s.v;
    f s.x;
    f s.tgt
end

type step_out = { m' : model; v' : model option; loss : Nx.float32_t }

module Step_out = struct
  type t = step_out

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) s =
    {
      m' = Model.map f s.m';
      v' = Option.map (Model.map f) s.v';
      loss = f s.loss;
    }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) a b =
    {
      m' = Model.map2 f a.m' b.m';
      v' =
        (match (a.v', b.v') with
        | Some va, Some vb -> Some (Model.map2 f va vb)
        | None, None -> None
        | _ -> invalid_arg "Step_out.map2: velocity mismatch");
      loss = f a.loss b.loss;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) s =
    Model.iter f s.m';
    Option.iter (Model.iter f) s.v';
    f s.loss
end

(* One SGD step: value_and_grad inside the (jitted or pmapped) function, so
   under pmap the gradients allreduce across devices before the update. *)
let train_step ~momentum { m; v; x; tgt } =
  let loss, grads = Rune.value_and_grad (module Model) (loss_fn x tgt) m in
  match v with
  | None ->
      let st = Vega.sgd_init (module Model) m in
      let m', _ = Vega.sgd_step (module Model) ~lr st ~params:m ~grads in
      { m'; v' = None; loss }
  | Some velocity ->
      let m', st =
        Vega.sgd_step
          (module Model)
          ~lr ~momentum { Vega.velocity } ~params:m ~grads
      in
      { m'; v' = Some st.Vega.velocity; loss }

let init ~momentum =
  let m = model_init () in
  {
    m;
    v = (if momentum then Some (Model.map Nx.zeros_like m) else None);
    x = x_init ();
    tgt = tgt_init ();
  }

(* One [in_axes] entry per leaf: everything replicated except the two batch
   leaves, sharded on axis 0. *)
let in_axes s =
  let n = ref 0 in
  Step_in.iter (fun _ -> incr n) s;
  List.init (!n - 2) (fun _ -> None) @ [ Some 0; Some 0 ]

let trajectory ~momentum ~steps step0 =
  let s = ref (init ~momentum) in
  Array.init steps (fun _ ->
      let out = step0 !s in
      s := { !s with m = out.m'; v = out.v' };
      (Nx.item [] out.loss, out.m'))

let run_both ~momentum ~steps =
  let mom = if momentum then 0.9 else 0.0 in
  let jit =
    trajectory ~momentum ~steps
      (Rune.jit2 (module Step_in) (module Step_out) (train_step ~momentum:mom))
  in
  let pm =
    trajectory ~momentum ~steps
      (Rune.pmap2 ~devices:devs2
         ~in_axes:(in_axes (init ~momentum))
         (module Step_in)
         (module Step_out)
         (train_step ~momentum:mom))
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
    (Model.map2
       (fun (type a b) (a : (a, b) Nx.t) (b : (a, b) Nx.t) : (a, b) Nx.t ->
         incr leaf;
         let d =
           Nx.item [] (Nx.astype Nx.float64 (Nx.max (Nx.abs (Nx.sub a b))))
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
