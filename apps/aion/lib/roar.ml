(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let invalid fmt =
  Printf.ksprintf (fun m -> invalid_arg ("Aion.Roar.sample: " ^ m)) fmt

(* A step's tokens from its logits [l], [b; k; v]. Each row is shifted so its
   largest logit is 0 before the division, so a small temperature sends the
   others toward -inf and never the largest to +inf, which [categorical]
   refuses. Where the temperature rounds to 0 in the logits' dtype, [0 / 0]
   would be NaN: the largest logits keep their 0, and the draw is uniform among
   them. *)
let draw ~key temperature l =
  if temperature = 0. then Nx.argmax ~axis:(-1) l
  else
    let d = Nx.sub l (Nx.max ~axes:[ -1 ] ~keepdims:true l) in
    let top = Nx.equal_s d 0. in
    Nx.Rng.categorical ~key (Nx.where top d (Nx.div_s d temperature))

let order at ~key b n =
  Nx.argsort ~axis:1
    (Nx.Rng.uniform_at at ~key:(Nx.Rng.fold_in key 0) Nx.float64 [| b; n |])

let sample ~key ?(temperature = 1.) ~reveal t inputs target =
  if temperature < 0. || not (Float.is_finite temperature) then
    invalid "temperature %g, not in [0, inf)" temperature;
  let name = Modality.key target and n = Modality.tokens target in
  List.iter (fun k -> if k < 1 then invalid "a count of %d in reveal" k) reveal;
  let total = List.fold_left ( + ) 0 reveal in
  if total <> n then invalid "reveal sums to %d, %s has %d tokens" total name n;
  if List.exists (fun (i : _ Transformer.input) -> i.modality = target) inputs
  then invalid "%s is an input" name;
  let first =
    match inputs with
    | (i : _ Transformer.input) :: _ -> i.tokens
    | [] -> invalid "no input"
  in
  let b = Nx.dim 0 first and at = Nx.beside first in
  let order = order at ~key b n in
  (* Step [s] reveals the [k] positions of each order from [o]. *)
  let step (s, o, tokens, revealed) k =
    let p = Nx.slice [ A; R (o, o + k) ] order in
    let known : _ Transformer.input =
      { modality = target; tokens; present = Some revealed }
    in
    let next : _ Transformer.target =
      { modality = target; positions = Some p }
    in
    let e = Transformer.encode t (inputs @ [ known ]) in
    let l = List.hd (Transformer.decode t e [ next ]) in
    let x = draw ~key:(Nx.Rng.fold_in key (s + 1)) temperature l in
    let shown = Nx.ones_at (Nx.beside x) Nx.bool (Nx.shape x) in
    ( s + 1,
      o + k,
      Nx.scatter ~axis:1 p x tokens,
      Nx.scatter ~axis:1 p shown revealed )
  in
  let tokens = Nx.zeros_at at Nx.int64 [| b; n |] in
  let revealed = Nx.zeros_at at Nx.bool [| b; n |] in
  let _, _, tokens, _ = List.fold_left step (0, 0, tokens, revealed) reveal in
  tokens
