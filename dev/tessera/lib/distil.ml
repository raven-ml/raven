(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('a, 'd) head = int * ('a, 'd) Kaun.Linear.t list
type ('a, 'd) params = { net : ('a, 'd) Model.net; heads : ('a, 'd) head list }

let walk_head c (k, layers) =
  let open Nx.Ptree.Walk in
  let k = field c "prefix" int k in
  let layers = field c "layers" (list Kaun.Linear.walk) layers in
  (k, layers)

let walk_heads c hs = Nx.Ptree.Walk.list walk_head c hs

module Params = struct
  type nonrec ('a, 'd) t = ('a, 'd) params

  let walk c p =
    let open Nx.Ptree.Walk in
    let net = field c "net" Model.Net.walk p.net in
    let heads = field c "heads" walk_heads p.heads in
    { net; heads }
end

let heads ~key ?(hidden = []) ~prefixes d =
  if List.exists (fun k -> k <= 0) (prefixes @ hidden) then
    invalid_arg "Tessera.Distil.heads: a prefix or a width is not positive";
  List.mapi
    (fun i k ->
      let key = Nx.Rng.fold_in key i in
      let widths = (k :: hidden) @ [ d ] in
      let rec layers j = function
        | a :: (b :: _ as rest) ->
            Kaun.Linear.make ~key:(Nx.Rng.fold_in key j) ~inputs:a ~outputs:b
              Nx.float32
            :: layers (j + 1) rest
        | _ -> []
      in
      (k, layers 0 widths))
    prefixes

(* Loss *)

let f32 = Nx.float32

(* [F.cosine_similarity]'s eps, which bounds each norm from below. *)
let cosine_eps = 1e-8
let relu x = Nx.maximum x (Nx.scalar_like x 0.)

let apply_head layers x =
  List.fold_left
    (fun (i, x) l ->
      let x = if i > 0 then relu x else x in
      (i + 1, Nx.cast f32 (Product.apply l x)))
    (0, x) layers
  |> snd

let unit x =
  let n = Nx.sqrt (Nx.sum ~axes:[ -1 ] ~keepdims:true (Nx.square x)) in
  Nx.div x (Nx.maximum n (Nx.scalar_like n cosine_eps))

let cosines hs e ~targets =
  let t = unit (Nx.cast f32 targets) in
  Nx.stack ~axis:1
    (List.map
       (fun (k, layers) ->
         let s = Nx.slice [ Nx.A; Nx.R (0, k) ] e in
         Nx.sum ~axes:[ -1 ] (Nx.mul (unit (apply_head layers s)) t))
       hs)

let loss ?record ~dropout ~keys m p (b : 'd Batch.t) ~targets =
  let m = { m with Model.net = p.net } in
  let e =
    Model.train ?record ~dropout ~remat:false ~keys m (Model.inputs m b)
  in
  let c = cosines p.heads e ~targets in
  let filled = Nx.reshape [| Nx.dim 0 c; 1 |] (Nx.greater_equal_s b.pixel 0L) in
  let one = Nx.scalar_like c 1. in
  Nx.sum (Nx.where filled (Nx.sub one c) (Nx.scalar_like c 0.))

(* Steps *)

type ('a, 'd) group = { batch : 'd Batch.t; keys : 'd Nx.Rng.key; targets : 'a }

module Group = struct
  type nonrec ('a, 'd) t = ('a, 'd) group

  let walk c g =
    let open Nx.Ptree.Walk in
    let batch = field c "batch" Batch.walk g.batch in
    let keys = field c "keys" Nx.Rng.walk g.keys in
    let targets = field c "targets" leaf g.targets in
    { batch; keys; targets }
end

let keys ~key ~first (b : Nx.host Batch.t) =
  let p =
    Nx.add_s (Nx.maximum b.pixel (Nx.scalar Nx.int64 0L)) (Int64.of_int first)
  in
  Nx.Rng.fold_in_tensor key p

type ('a, 'd) state = {
  model : ('a, 'd) Model.t;
  heads : ('a, 'd) head list;
  opt : (('a, 'd) params, 'd) Vega.adam_state;
}

module State = struct
  type nonrec ('a, 'd) t = ('a, 'd) state

  let walk c st =
    let open Nx.Ptree.Walk in
    let model = field c "model" Model.walk st.model in
    let heads = field c "heads" walk_heads st.heads in
    let opt = field c "opt" (under Vega.Adam_state.walk Params.walk) st.opt in
    { model; heads; opt }
end

let init m hs =
  let p = Nx.Ptree.instantiate (module Params) in
  let opt = Vega.adamw_init p { net = m.Model.net; heads = hs } in
  { model = m; heads = hs; opt }

(* AdamW's second moment's decay in v2's distillation (App. J, Table 9). *)
let b2 = 0.95

let step ~compute ~dropout ~lr ~weight_decay ?decays ~clip
    ((gs : ((float, 't, 'd) Nx.t, 'd) group), (st : ('d Nx.float32_t, 'd) state))
    =
  let p = Nx.Ptree.instantiate (module Params) in
  let params = { net = st.model.net; heads = st.heads } in
  let one = Nx.full_at (Nx.beside gs.batch.pixel) f32 [||] 1. in
  let trip (total, sum) (g : ((float, 't, 'd) Nx.t, 'd) group) =
    let objective q =
      loss ~dropout ~keys:g.keys st.model
        (Nx.Ptree.cast (module Params) compute q)
        g.batch ~targets:g.targets
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
  (* A step of no filled place divides by 1: its gradient is zero. *)
  let filled = Nx.sum (Nx.cast f32 (Nx.greater_equal_s gs.batch.pixel 0L)) in
  let n = Nx.maximum filled one in
  let grads = Nx.Ptree.scale p (Nx.recip n) sum in
  let grads = Vega.clip_by_global_norm p ~max_norm:clip grads in
  let params, opt =
    Vega.adamw_step p ~lr ~b2 ~weight_decay ?decays st.opt ~params ~grads
  in
  ( Nx.div total n,
    { model = { st.model with net = params.net }; heads = params.heads; opt } )
