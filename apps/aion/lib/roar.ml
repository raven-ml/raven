(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* A step's tokens from its logits [l], [b; k; v], at the temperature [t], a
   0-d float64 array, or the largest logit's without one. Each row is shifted so
   its largest logit is 0 before the division, so a small temperature sends the
   others toward -inf and never the largest to +inf, which [categorical]
   refuses. Where the temperature rounds to 0 in the logits' dtype, [0 / 0]
   would be NaN: the largest logits keep their 0, and the draw is uniform among
   them. *)
let pick ~key t l =
  match t with
  | None -> Nx.argmax ~axis:(-1) l
  | Some t ->
      let d = Nx.sub l (Nx.max ~axes:[ -1 ] ~keepdims:true l) in
      let top = Nx.equal_s d 0. in
      let t = Nx.cast (Nx.dtype d) t in
      Nx.Rng.categorical ~key (Nx.where top d (Nx.div d t))

let order at ~key b n =
  Nx.argsort ~axis:1
    (Nx.Rng.uniform_at at ~key:(Nx.Rng.fold_in key 0) Nx.float64 [| b; n |])

(* Each object's order of [n] positions, at the placement of [first], an
   input's tokens. *)
let orders n first key = order (Nx.beside first) ~key (Nx.dim 0 first) n

(* Step [s], a 0-d index ([s + 1] for its key), draws [target]'s tokens at the
   positions [p] at the temperature [temp] and makes them present. *)
let step ~target t key s temp inputs tokens revealed p =
  let known : _ Tokens.t = { modality = target; tokens; present = revealed } in
  let e = Transformer.encode t (inputs @ [ known ]) in
  let next : _ Transformer.target = { modality = target; positions = Some p } in
  let l = List.hd (Transformer.decode t e [ next ]) in
  let x = pick ~key:(Nx.Rng.fold_in_tensor key s) temp l in
  let shown = Nx.ones_at (Nx.beside x) Nx.bool (Nx.shape x) in
  (Nx.scatter ~axis:1 p x tokens, Nx.scatter ~axis:1 p shown revealed)

let invalid fmt = Printf.ksprintf invalid_arg fmt

(* The checks of a draw's arguments, before any step: the batch's size and
   placement. *)
let checked fn ~temperature ~reveal inputs target =
  if temperature < 0. || not (Float.is_finite temperature) then
    invalid "Aion.%s: temperature %g, not in [0, inf)" fn temperature;
  let name = Modality.key target and n = Modality.tokens target in
  List.iter
    (fun k -> if k < 1 then invalid "Aion.%s: a count of %d in reveal" fn k)
    reveal;
  let total = List.fold_left ( + ) 0 reveal in
  if total <> n then
    invalid "Aion.%s: reveal sums to %d, %s has %d tokens" fn total name n;
  if List.exists (fun (i : _ Tokens.t) -> i.modality = target) inputs
  then invalid "Aion.%s: %s is an input" fn name;
  match inputs with
  | (i : _ Tokens.t) :: _ -> (Nx.dim 0 i.tokens, Nx.beside i.tokens)
  | [] -> invalid "Aion.%s: no input" fn

(* The draw's loop: the orders, then each count's step, from no token
   revealed. *)
let run ~orders ~step ~key ~temperature ~reveal ~b ~at inputs target =
  let n = Modality.tokens target in
  let tokens = Nx.zeros_at at Nx.int64 [| b; n |] in
  let revealed = Nx.zeros_at at Nx.bool [| b; n |] in
  let first = (List.hd inputs : _ Tokens.t).tokens in
  let order = orders first key in
  let temp =
    if temperature = 0. then None
    else Some (Nx.scalar_at at Nx.float64 temperature)
  in
  let go (s, o, tokens, revealed) k =
    let p = Nx.slice [ A; R (o, o + k) ] order in
    let s' = Nx.scalar_at at Nx.int64 (Int64.of_int (s + 1)) in
    let tokens, revealed = step key s' temp inputs tokens revealed p in
    (s + 1, o + k, tokens, revealed)
  in
  let _, _, tokens, _ = List.fold_left go (0, 0, tokens, revealed) reveal in
  tokens

let sample ~key ?(temperature = 1.) ~reveal t inputs target =
  let b, at = checked "Roar.sample" ~temperature ~reveal inputs target in
  run
    ~orders:(orders (Modality.tokens target))
    ~step:(step ~target t) ~key ~temperature ~reveal ~b ~at inputs target

(* Samplers *)

type ('b, 'd) sampler = {
  target : Modality.t;
  orders : 'd Nx.int64_t -> 'd Nx.Rng.key -> 'd Nx.int64_t;
  draws :
    'd Nx.Rng.key ->
    'd Nx.int64_t ->
    'd Nx.float64_t ->
    'd Tokens.t list ->
    'd Nx.int64_t ->
    'd Nx.bool_t ->
    'd Nx.int64_t ->
    'd Nx.int64_t * 'd Nx.bool_t;
  argmaxes :
    'd Nx.Rng.key ->
    'd Nx.int64_t ->
    'd Tokens.t list ->
    'd Nx.int64_t ->
    'd Nx.bool_t ->
    'd Nx.int64_t ->
    'd Nx.int64_t * 'd Nx.bool_t;
}

let sampler t target =
  let open Nx.Ptree in
  let inputs = list (v Tokens.walk) and result = pair tensor tensor in
  let step = step ~target t in
  {
    target;
    orders =
      Rune.jit
        (tensor @-> Nx.Rng.ptree @-> returns tensor)
        (orders (Modality.tokens target));
    draws =
      Rune.jit
        (Nx.Rng.ptree @-> tensor @-> tensor @-> inputs @-> tensor @-> tensor
       @-> tensor @-> returns result)
        (fun key s temp -> step key s (Some temp));
    argmaxes =
      Rune.jit
        (Nx.Rng.ptree @-> tensor @-> inputs @-> tensor @-> tensor @-> tensor
        @-> returns result)
        (fun key s -> step key s None);
  }

let draw s ~key ?(temperature = 1.) ~reveal inputs =
  let b, at = checked "Roar.draw" ~temperature ~reveal inputs s.target in
  let step key st temp =
    match temp with
    | None -> s.argmaxes key st
    | Some temp -> s.draws key st temp
  in
  run ~orders:s.orders ~step ~key ~temperature ~reveal ~b ~at inputs s.target
