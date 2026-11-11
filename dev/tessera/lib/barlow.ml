(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('a, 'd) projector = {
  hidden :
    (('a, 'd) Kaun.Linear.t * ('d Nx.float32_t, 'd) Kaun.Batch_norm.t) list;
  out : ('a, 'd) Kaun.Linear.t;
}

type ('a, 'd) params = {
  net : ('a, 'd) Model.net;
  projector : ('a, 'd) projector;
}

(* Batch normalisation's scale and shift are float32 fixed arrays, as the
   model's norms are, so a cast rounds the products' parameters alone. *)
let walk_projector c p =
  let open Nx.Ptree.Walk in
  let layer c (l, bn) =
    let l = index c 0 Kaun.Linear.walk l in
    let bn = index c 1 (under Kaun.Batch_norm.walk tensor) bn in
    (l, bn)
  in
  let hidden = field c "hidden" (list layer) p.hidden in
  let out = field c "out" Kaun.Linear.walk p.out in
  { hidden; out }

module Params = struct
  type nonrec ('a, 'd) t = ('a, 'd) params

  let walk c p =
    let open Nx.Ptree.Walk in
    let net = field c "net" Model.Net.walk p.net in
    let projector = field c "projector" walk_projector p.projector in
    { net; projector }
end

let f32 = Nx.float32

let projector ~key widths =
  if List.length widths < 2 || List.exists (fun w -> w <= 0) widths then
    invalid_arg
      "Tessera.Barlow.projector: fewer than two widths, or one not positive";
  let at = Nx.Rng.placement key in
  let linear i inputs outputs =
    Kaun.Linear.make ~key:(Nx.Rng.fold_in key i) ~inputs ~outputs f32
  in
  let rec layers i = function
    | [ a; b ] -> ([], linear i a b)
    | a :: (b :: _ as rest) ->
        let bn =
          {
            Kaun.Batch_norm.gamma = Nx.ones_at at f32 [| b |];
            beta = Nx.zeros_at at f32 [| b |];
          }
        in
        let hidden, out = layers (i + 1) rest in
        ((linear i a b, bn) :: hidden, out)
    | _ -> assert false
  in
  let hidden, out = layers 0 widths in
  { hidden; out }

(* BatchNorm1d's eps (PyTorch's default, which v1.0 keeps). *)
let bn_eps = 1e-5
let relu x = Nx.maximum x (Nx.scalar_like x 0.)
let linear p x = Nx.cast f32 (Product.apply p x)

let project p e =
  let x =
    List.fold_left
      (fun x (l, (bn : _ Kaun.Batch_norm.t)) ->
        relu
          (Nx.standardize ~axes:[ 0 ] ~eps:bn_eps ~scale:bn.gamma ~shift:bn.beta
             (linear l x)))
      e p.hidden
  in
  linear p.out x

(* Objective *)

type weights = { off_diagonal : float; mixup : float }

let v1_weights = { off_diagonal = 5e-3; mixup = 5e-3 }

type form = Gram | Correlation

(* The guard on a column's deviation (ssl_model.py@290feaf:20-26). *)
let std_guard = 1e-9

(* Each column centred and divided by its unbiased deviation, at least
   [std_guard]. *)
let whiten z =
  let mean = Nx.mean ~axes:[ 0 ] ~keepdims:true z in
  let std = Nx.std ~axes:[ 0 ] ~keepdims:true ~ddof:1 z in
  Nx.div (Nx.sub z mean) (Nx.maximum std (Nx.scalar_like std std_guard))

let tr x = Nx.matrix_transpose x
let gram x = Nx.matmul x (tr x)

let objective_in form w ~alpha ~perm z1 z2 zm =
  let shape = Nx.shape z1 in
  if Nx.ndim z1 <> 2 || Nx.shape z2 <> shape || Nx.shape zm <> shape then
    invalid_arg "Tessera.Barlow.objective: projections of different shapes";
  let b = shape.(0) in
  if b < 2 then invalid_arg "Tessera.Barlow.objective: a group of fewer than 2";
  if Nx.shape perm <> [| b |] then
    invalid_arg
      (Printf.sprintf "Tessera.Barlow.objective: a permutation of [%d] expected"
         b);
  let n = Float.of_int b in
  let w1 = whiten z1 and w2 = whiten z2 and wm = whiten zm in
  let w2p = Nx.take ~axis:0 perm w2 in
  let beta = Nx.rsub_s 1. alpha in
  (* The mixup's correlations less their targets, [dᵀ w₁ / B] and [dᵀ w₂ₚ / B],
     share [d]. *)
  let d = Nx.sub (Nx.sub wm (Nx.mul alpha w1)) (Nx.mul beta w2p) in
  let diag = Nx.div_s (Nx.sum ~axes:[ 0 ] (Nx.mul w1 w2)) n in
  let on = Nx.sum (Nx.square (Nx.sub_s diag 1.)) in
  let off, mixed =
    match form with
    | Gram ->
        let g1 = gram w1 and g2 = gram w2 and gd = gram d in
        let g2p = Nx.take ~axis:1 perm (Nx.take ~axis:0 perm g2) in
        let frob g g' = Nx.sum (Nx.mul g g') in
        ( Nx.sub (Nx.div_s (frob g1 g2) (n *. n)) (Nx.sum (Nx.square diag)),
          Nx.div_s (Nx.add (frob gd g1) (frob gd g2p)) (n *. n) )
    | Correlation ->
        let c = Nx.div_s (Nx.matmul (tr w1) w2) n in
        let p = Nx.dim 1 c in
        let eye = Nx.cast Nx.bool (Nx.eye_at (Nx.beside c) Nx.int32 p) in
        let off = Nx.sum (Nx.where eye (Nx.scalar_like c 0.) (Nx.square c)) in
        let frob x y = Nx.sum (Nx.square (Nx.matmul (tr x) y)) in
        (off, Nx.div_s (Nx.add (frob d w1) (frob d w2p)) (n *. n))
  in
  Nx.add (Nx.add on (Nx.mul_s off w.off_diagonal)) (Nx.mul_s mixed w.mixup)

(* The Gram form where its products cost at most half the P × P form's and its
   subtraction amplifies rounding by at most about 2, as the interface
   states. *)
let objective w ~alpha ~perm z1 z2 zm =
  let b = Nx.dim 0 z1 and p = Nx.dim 1 z1 in
  objective_in
    (if 2 * b <= p then Gram else Correlation)
    w ~alpha ~perm z1 z2 zm

type 'd views = {
  a1 : 'd Batch.t;
  a2 : 'd Batch.t;
  keys : 'd Nx.Rng.key;
  alpha : 'd Nx.float32_t;
  perm : 'd Nx.int64_t;
}

let pack (v : Nx.host Pixels.t) =
  let n = Pixels.length v in
  if n = 0 then invalid_arg "Tessera.Barlow.pack: no pixel";
  let length r =
    let l = Nx.to_array (Nx_ragged.lengths r) in
    if Array.exists (fun x -> x <> l.(0)) l then
      invalid_arg "Tessera.Barlow.pack: rows of different lengths";
    Int64.to_int l.(0)
  in
  let l2 = length v.s2 and l1 = length v.s1 in
  if l2 <> l1 then invalid_arg "Tessera.Barlow.pack: rows of different lengths";
  let shape =
    { Batch.pixels = n; s2_rows = n; s1_rows = n; slots = 1; width = l2 }
  in
  let ones r = Nx.ones Nx.int32 [| Nx.dim 0 (Nx_ragged.values r) |] in
  let b, _ = Batch.pack shape v { s2 = ones v.s2; s1 = ones v.s1 } ~from:0 in
  b

(* A group's keys: [key] folded with a site, then a view, then a stream
   position. *)
let draw_site = 0
let dropout_site = 1
let mix_site = 2

let group ~key ~length ~first px =
  let n = Pixels.length px in
  let positions = Nx.arange Nx.int64 first (first + n) 1 in
  let keys site view =
    Nx.Rng.fold_in_tensor
      (Nx.Rng.fold_in (Nx.Rng.fold_in key site) view)
      positions
  in
  let view v = pack (Pixels.draw ~keys:(keys draw_site v) ~length px) in
  let dropout =
    Nx.Rng.of_tensor
      (Nx.stack (List.init 3 (fun v -> Nx.Rng.to_tensor (keys dropout_site v))))
  in
  let mix = Nx.Rng.fold_in (Nx.Rng.fold_in key mix_site) first in
  let k = Nx.Rng.split mix in
  {
    a1 = view 0;
    a2 = view 1;
    keys = dropout;
    alpha = Nx.Rng.uniform ~key:k.(0) f32 [||];
    perm = Nx.Rng.permute ~key:k.(1) ~n (Nx.arange Nx.int64 0 n 1);
  }

(* [b] with its rows' tokens permuted by [perm]: a view's batch has one pixel a
   row, every row laid out alike. *)
let permuted perm (b : 'd Batch.t) =
  let rows (r : 'd Batch.rows) =
    { r with tokens = Nx.take ~axis:0 perm r.tokens }
  in
  { b with s2 = rows b.s2; s1 = rows b.s1 }

let loss w ~dropout ~remat m p v =
  let m = { m with Model.net = p.net } in
  let keys k = Nx.Rng.of_tensor (Nx.get [ k ] (Nx.Rng.to_tensor v.keys)) in
  let z k i =
    project p.projector (Model.train ~dropout ~remat ~keys:(keys k) m i)
  in
  let i1 = Model.inputs m v.a1 in
  let z1 = z 0 i1 and z2 = z 1 (Model.inputs m v.a2) in
  let mixed =
    Model.mixup ~alpha:v.alpha i1 (Model.inputs m (permuted v.perm v.a2))
  in
  let zm = z 2 mixed in
  objective w ~alpha:v.alpha ~perm:v.perm z1 z2 zm

(* Steps *)

type ('a, 'd) group = 'd views

module Group = struct
  type nonrec ('a, 'd) t = ('a, 'd) group

  let walk c v =
    let open Nx.Ptree.Walk in
    let a1 = field c "a1" Batch.walk v.a1 in
    let a2 = field c "a2" Batch.walk v.a2 in
    let keys = field c "keys" Nx.Rng.walk v.keys in
    let alpha = field c "alpha" tensor v.alpha in
    let perm = field c "perm" tensor v.perm in
    { a1; a2; keys; alpha; perm }
end

let stack vs =
  let stack f = Nx.stack (List.map f vs) in
  {
    a1 = Batch.stack (List.map (fun v -> v.a1) vs);
    a2 = Batch.stack (List.map (fun v -> v.a2) vs);
    keys = Nx.Rng.of_tensor (stack (fun v -> Nx.Rng.to_tensor v.keys));
    alpha = stack (fun v -> v.alpha);
    perm = stack (fun v -> v.perm);
  }

type ('a, 'd) state = {
  model : ('a, 'd) Model.t;
  projector : ('a, 'd) projector;
  opt : (('a, 'd) params, 'd) Vega.adam_state;
}

module State = struct
  type nonrec ('a, 'd) t = ('a, 'd) state

  let walk c st =
    let open Nx.Ptree.Walk in
    let model = field c "model" Model.walk st.model in
    let projector = field c "projector" walk_projector st.projector in
    let opt = field c "opt" (under Vega.Adam_state.walk Params.walk) st.opt in
    { model; projector; opt }
end

let init m pr =
  let p = Nx.Ptree.instantiate (module Params) in
  let opt = Vega.adamw_init p { net = m.Model.net; projector = pr } in
  { model = m; projector = pr; opt }

let decays path _ =
  match List.rev (Nx.Ptree.Path.segments path) with
  | Nx.Ptree.Path.Field "w" :: _ -> true
  | _ -> false

type recipe = {
  pixels : int;
  length : int;
  projector : int list;
  lr : float;
  warmup : float;
  weight_decay : float;
  clip : float;
  dropout : Model.dropout;
}

(* The released run's configuration (its checkpoint's config dict), and the
   decay it applied. *)
let v1_recipe =
  {
    pixels = 2048;
    length = 40;
    projector = List.init 6 (fun _ -> 16384);
    lr = 2e-3;
    warmup = 0.1;
    weight_decay = 0.;
    clip = 2.0;
    dropout = { layers = 0.1; reducer = 0.2 };
  }

let step ~compute w ~dropout ~remat ~lr ~weight_decay ?(decays = decays) ~clip
    ((gs : ('d Nx.float32_t, 'd) group), (st : ('d Nx.float32_t, 'd) state)) =
  let p = Nx.Ptree.instantiate (module Params) in
  let params = { net = st.model.net; projector = st.projector } in
  let one = Nx.full_at (Nx.beside gs.alpha) f32 [||] 1. in
  let trip (total, sum) (g : ('d Nx.float32_t, 'd) group) =
    let objective q =
      loss w ~dropout ~remat st.model
        (Nx.Ptree.cast (module Params) compute q)
        g
    in
    let l, grads = Rune.value_and_grad p objective params in
    ((Nx.add total l, Nx.Ptree.axpy p one grads sum), ())
  in
  let zeros = Nx.Ptree.map p (fun _ x -> Nx.zeros_like x) params in
  let zero = Nx.scalar_like one 0. in
  let (total, sum), () =
    Rune.scan
      Nx.Ptree.(pair tensor p)
      (Nx.Ptree.instantiate (module Group))
      Nx.Ptree.unit trip ~init:(zero, zeros) gs
  in
  let n = Float.of_int (Nx.dim 0 gs.alpha) in
  let grads = Nx.Ptree.scale p (Nx.scalar_like one (1. /. n)) sum in
  let grads = Vega.clip_by_global_norm p ~max_norm:clip grads in
  let params, opt =
    Vega.adamw_step p ~lr ~weight_decay ~decays st.opt ~params ~grads
  in
  ( Nx.div_s total n,
    {
      model = { st.model with net = params.net };
      projector = params.projector;
      opt;
    } )
