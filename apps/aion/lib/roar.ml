(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid fmt = Printf.ksprintf invalid_arg fmt

type 'd temperature = Greedy | Tempered of 'd Nx.float64_t
[@@deriving ptree]

(* [keys] holds a key per object, [order] each object's order rotated so that
   the positions not yet revealed come first, and [step], 0-d, the index the
   next step's keys fold in. *)
type 'd draw = {
  keys : 'd Nx.Rng.key;
  inputs : 'd Tokens.t list;
  target : 'd Tokens.t;
  order : 'd Nx.int64_t;
  step : 'd Nx.int64_t;
  temperature : 'd temperature;
}
[@@deriving ptree]

let start ~keys ?(temperature = 1.) inputs target =
  if temperature < 0. || not (Float.is_finite temperature) then
    invalid "Aion.Roar.start: temperature %g, not in [0, inf)" temperature;
  if List.exists (fun (i : _ Tokens.t) -> i.modality = target) inputs then
    invalid "Aion.Roar.start: %s is an input" (Modality.key target);
  let first =
    match inputs with
    | (i : _ Tokens.t) :: _ -> i.tokens
    | [] -> invalid "Aion.Roar.start: no input"
  in
  let b = Nx.dim 0 first in
  (match Nx.shape (Nx.Rng.to_tensor keys) with
  | [| b'; 2 |] when b' = b -> ()
  | s ->
      invalid "Aion.Roar.start: keys of shape %s, [%d; 2] expected"
        (Format.asprintf "%a" Nx.pp_shape s)
        b);
  let at = Nx.beside first in
  let n = Modality.tokens target in
  let u =
    Nx.Rng.uniform_at at ~key:(Nx.Rng.fold_in keys 0) Nx.float64 [| n |]
  in
  let shape = [| b; n |] in
  let tokens = Nx.zeros_at at Nx.int64 shape in
  let present = Nx.zeros_at at Nx.bool shape in
  let temperature =
    if temperature = 0. then Greedy
    else Tempered (Nx.scalar_at at Nx.float64 temperature)
  in
  {
    keys;
    inputs;
    target = { modality = target; tokens; present };
    order = Nx.argsort ~axis:1 u;
    step = Nx.scalar_at at Nx.int64 1L;
    temperature;
  }

(* A step's tokens from its logits [l], [b; k; v]. Each row is shifted so its
   largest logit is 0 before the division by the temperature, so a small
   temperature sends the others toward -inf and never the largest to +inf,
   which [categorical] refuses. Where the temperature rounds to 0 in the
   logits' dtype, [0 / 0] would be NaN: the largest logits keep their 0, and the
   draw is uniform among them. *)
let pick ~key temperature l =
  match temperature with
  | Greedy -> Nx.argmax ~axis:(-1) l
  | Tempered t ->
      let d = Nx.sub l (Nx.max ~axes:[ -1 ] ~keepdims:true l) in
      let top = Nx.equal_s d 0. in
      let t = Nx.cast (Nx.dtype d) t in
      Nx.Rng.categorical ~key (Nx.where top d (Nx.div d t))

let step t k d =
  let n = Modality.tokens d.target.modality in
  if k < 1 || k > n then
    invalid "Aion.Roar.step: a count of %d, not in [1, %d]" k n;
  let p = Nx.slice [ A; R (0, k) ] d.order in
  let hidden =
    Nx.logical_not (Nx.take_along_axis ~axis:1 p d.target.present)
  in
  Nx.check Nx.Ptree.tensor hidden p (fun i p ->
      Invalid_argument
        (Format.asprintf
           "Aion.Roar.step: position %Ld at %a is already revealed"
           (Nx.item [] p) Nx.pp_shape i));
  let e = Transformer.encode t (d.inputs @ [ d.target ]) in
  let next : _ Transformer.target =
    { modality = d.target.modality; positions = Some p }
  in
  let l = List.hd (Transformer.decode t e [ next ]) in
  (* Keys of batch shape [b; 1; 1] against the [b; k; v] logits: each object
     draws its logits from its own key, as it would alone. *)
  let key =
    Nx.Rng.reshape [| Nx.dim 0 l; 1; 1 |] (Nx.Rng.fold_in_tensor d.keys d.step)
  in
  let x = pick ~key d.temperature l in
  let shown = Nx.ones_at (Nx.beside x) Nx.bool (Nx.shape x) in
  let target : _ Tokens.t =
    {
      d.target with
      tokens = Nx.scatter ~axis:1 p x d.target.tokens;
      present = Nx.scatter ~axis:1 p shown d.target.present;
    }
  in
  let order = Nx.roll ~axis:1 (-k) d.order in
  { d with target; order; step = Nx.add_s d.step 1L }

let tokens d = d.target.tokens

let sample ~keys ?temperature ~reveal t inputs target =
  let check k =
    if k < 1 then invalid "Aion.Roar.sample: a count of %d in reveal" k
  in
  List.iter check reveal;
  let total = List.fold_left ( + ) 0 reveal and n = Modality.tokens target in
  if total <> n then
    invalid "Aion.Roar.sample: reveal sums to %d, %s has %d tokens" total
      (Modality.key target) n;
  let d = start ~keys ?temperature inputs target in
  tokens (List.fold_left (fun d k -> step t k d) d reveal)
