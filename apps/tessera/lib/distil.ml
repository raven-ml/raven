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

(* An npz's entry [name] at float32, on the host. *)
let entry fn file a name =
  match Nx_io.Archive.find name a with
  | None -> failwith (Printf.sprintf "Tessera.%s: %s: no entry %s" fn file name)
  | Some (Nx.P x) -> Nx_io.Archive.float ~shape:(Nx.shape x) Nx.float32 name a

let dims s = String.concat "; " (Array.to_list (Array.map string_of_int s))

let load_heads file prefixes =
  let a = Nx_io.load_npz file in
  let entry = entry "Distil.load_heads" file a in
  let fail name x need =
    failwith
      (Printf.sprintf "Tessera.Distil.load_heads: %s: %s [%s], [%s] expected"
         file name (dims (Nx.shape x)) need)
  in
  let read name need =
    let x = entry name in
    if Nx.shape x <> need then fail name x (dims need);
    x
  in
  (* Every head's d is the first weight's. *)
  let d =
    match prefixes with
    | [] -> 0
    | k :: _ ->
        let name = Printf.sprintf "%d.weight" k in
        let w = entry name in
        if Nx.ndim w <> 2 then fail name w (Printf.sprintf "d; %d" k);
        Nx.dim 0 w
  in
  List.map
    (fun k ->
      let w = read (Printf.sprintf "%d.weight" k) [| d; k |] in
      let b = read (Printf.sprintf "%d.bias" k) [| d |] in
      (k, [ { Kaun.Linear.w = w; b = Some b } ]))
    prefixes

(* Loss *)

let f32 = Nx.float32

(* [F.cosine_similarity]'s eps, which bounds each norm from below. *)
let cosine_eps = 1e-8

let apply_head layers x =
  List.fold_left
    (fun (i, x) l ->
      (i + 1, Model.linear l (if i > 0 then Kaun.Fn.relu x else x)))
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

let loss ~dropout ~keys m p (b : 'd Batch.t) ~targets =
  let m = { m with Model.net = p.net } in
  let e = Model.train ~dropout ~keys m (Model.inputs m b) in
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

let targets file (Nx.P t) ~first indices =
  let k = Nx.numel indices in
  if first + k <= Nx.dim 0 t then
    Nx.cast Nx.float32 (Nx.take ~axis:0 indices t)
  else
    let row = Array.map Int64.to_int (Nx.to_array indices) in
    match Nx_io.load_npy_rows file ~row ~rows:(Array.make k 1) with
    | Nx.P x -> Nx.cast Nx.float32 x

let keys ~key ~first (b : Nx.host Batch.t) =
  let p =
    Nx.add_s (Nx.maximum b.pixel (Nx.scalar Nx.int64 0L)) (Int64.of_int first)
  in
  Nx.Rng.fold_in_tensor key p

(* The rows of 256 tokens of [pixels] pixels of the mean tokens of the
   corpus's first pixels, 5 % to spare, and a row more. *)
let shape ~pixels ~slots px =
  let n = Pixels.length px in
  if n = 0 then invalid_arg "Tessera.Distil.shape: a corpus of no pixel";
  let sample = Pixels.sub px ~pos:0 ~len:(min 4096 n) in
  let _, c = Batch.expand sample (Batch.v2_counts sample) in
  let width = 256 in
  let rows x =
    let mean =
      Float.of_int (Nx.numel x) /. Float.of_int (Pixels.length sample)
    in
    Float.to_int
      (Float.ceil (1.05 *. mean *. Float.of_int pixels /. Float.of_int width))
    + 1
  in
  { Batch.pixels; s2_rows = rows c.s2; s1_rows = rows c.s1; slots; width }

let groups ~key ~shape ~groups ~first px targets =
  if Nx.dim 0 targets <> Pixels.length px then
    invalid_arg "Tessera.Distil.groups: targets of other rows than the pixels";
  let px, counts = Batch.expand px (Batch.v2_counts px) in
  let rec go g from acc =
    if g = groups then (List.rev acc, from)
    else
      let b, next = Batch.pack shape px counts ~from in
      let places = Nx.maximum b.pixel (Nx.scalar Nx.int64 0L) in
      let t = Nx.take ~axis:0 places targets in
      let g' = { batch = b; keys = keys ~key ~first b; targets = t } in
      go (g + 1) next (g' :: acc)
  in
  let gs, taken = go 0 0 [] in
  let stack f = Nx.stack (List.map f gs) in
  ( {
      batch = Batch.stack (List.map (fun g -> g.batch) gs);
      keys = Nx.Rng.of_tensor (stack (fun g -> Nx.Rng.to_tensor g.keys));
      targets = stack (fun g -> g.targets);
    },
    taken )

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

type recipe = { lr : float; warmup : float; weight_decay : float; clip : float }

let v2_recipe = { lr = 8e-4; warmup = 0.05; weight_decay = 1e-4; clip = 1.0 }

(* AdamW's second moment's decay in v2's distillation (App. J, Table 9). *)
let b2 = 0.95

let step ~compute ~dropout ~lr ~weight_decay ?decays ~clip
    ((gs : ((float, 't, 'd) Nx.t, 'd) group), (st : ('d Nx.float32_t, 'd) state))
    =
  let p = Nx.Ptree.instantiate (module Params) in
  let params = { net = st.model.net; heads = st.heads } in
  let group_loss (g : ((float, 't, 'd) Nx.t, 'd) group) q =
    let l =
      loss ~dropout ~keys:g.keys st.model
        (Nx.Ptree.cast (module Params) compute q)
        g.batch ~targets:g.targets
    in
    (l, l)
  in
  let sum, losses =
    Grads.sum p (Nx.Ptree.instantiate (module Group)) group_loss params gs
  in
  (* A step of no filled place divides by 1: its gradient is zero. *)
  let filled = Nx.sum (Nx.cast f32 (Nx.greater_equal_s gs.batch.pixel 0L)) in
  let n = Nx.maximum filled (Nx.scalar_like filled 1.) in
  let grads = Nx.Ptree.scale p (Nx.recip n) sum in
  let grads = Vega.clip_by_global_norm p ~max_norm:clip grads in
  let params, opt =
    Vega.adamw_step p ~lr ~b2 ~weight_decay ?decays st.opt ~params ~grads
  in
  ( Nx.div (Nx.sum losses) n,
    { model = { st.model with net = params.net }; heads = params.heads; opt } )
