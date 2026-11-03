(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('a, 'd) weight = Float of 'a | Quant of 'd Nx_quant.t

type ('a, 'd) t = {
  gate_up : ('a, 'd) weight;
  gate_up_bias : 'a;
  down : ('a, 'd) weight;
  down_bias : 'a;
}

let walk_weight c =
  let open Nx.Ptree.Walk in
  function
  | Float w ->
      case c "float";
      Float (leaf c w)
  | Quant w ->
      case c "quant";
      Quant (Nx_quant.walk c w)

let walk c p =
  let open Nx.Ptree.Walk in
  let gate_up = field c "gate_up" walk_weight p.gate_up in
  let gate_up_bias = field c "gate_up_bias" leaf p.gate_up_bias in
  let down = field c "down" walk_weight p.down in
  let down_bias = field c "down_bias" leaf p.down_bias in
  { gate_up; gate_up_bias; down; down_bias }

let route ~k logits =
  let logits, experts = Nx.top_k ~k logits in
  (experts, Nx.softmax logits)

let activation ~limit h =
  let shape = Nx.shape h in
  let last = Array.length shape - 1 in
  let half = shape.(last) / 2 in
  let out = Array.copy shape in
  out.(last) <- half;
  let pairs = Nx.reshape [| -1; half; 2 |] h in
  let feature i = Nx.reshape out (Nx.slice [ A; A; I i ] pairs) in
  let gate = Nx.clamp ~max:limit (feature 0) in
  let linear = Nx.clamp ~min:(-.limit) ~max:limit (feature 1) in
  Nx.mul (Nx.mul gate (Nx.sigmoid (Nx.mul_s gate 1.702))) (Nx.add_s linear 1.0)

(* [linear ~sum w b ids x] is the projection [w] and bias [b] of each
   position's expert, its id in [ids], [[| s… |]], applied to its row of
   [x], [[| s'…; inputs |]]: one product whose rows the kernels group by
   expert. A weight cut along its inputs or its experts gives, with [sum],
   the sum of the devices' terms ({!Nx.Placement.partial}). *)
let linear ~sum w b ids x =
  let y =
    match w with
    | Quant w -> Nx_quant.apply_experts ~sum ~ids w x
    | Float w -> Nx.matmul_segments ~sum ids x w
  in
  Nx.add y (Nx.take ~axis:0 ids b)

(* The activation reads [gate_up]'s product closed; [down]'s stays a sum,
   which the top-k combine keeps and closes once. *)
let apply ~limit p (ids, weights) x =
  let x = Nx.unsqueeze ~axes:[ -2 ] x in
  let h = linear ~sum:false p.gate_up p.gate_up_bias ids x in
  let y = linear ~sum:true p.down p.down_bias ids (activation ~limit h) in
  Nx.sum ~axes:[ -2 ] (Nx.mul y (Nx.unsqueeze ~axes:[ -1 ] weights))
