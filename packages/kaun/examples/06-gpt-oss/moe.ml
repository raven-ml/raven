(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type 'a weight = Float of 'a | Quant of Nx_quant.t

type 'a t = {
  gate_up : 'a weight;
  gate_up_bias : 'a;
  down : 'a weight;
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
      Quant (structure Nx_quant.ptree c w)

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

(* [linear w b e x] is the projection [w] and bias [b] of each block's expert
   [e], [[| g |]], applied to the block's rows [x], [[| g; c; inputs |]]. *)
let linear w b e x =
  let y =
    match w with
    | Quant w -> Nx_quant.apply (Nx_quant.take ~axis:0 ~indices:e w) x
    | Float w -> Nx.matmul x (Nx.take ~axis:0 ~indices:e w)
  in
  Nx.add y (Nx.unsqueeze ~axes:[ 1 ] (Nx.take ~axis:0 ~indices:e b))

let expert ~limit p e x =
  linear p.down p.down_bias e
    (activation ~limit (linear p.gate_up p.gate_up_bias e x))

let apply ~limit p (ids, weights) x =
  let experts = Nx.dim 0 p.down_bias in
  let y =
    Nx.map_segments ~segments:experts ids (expert ~limit p)
      (Nx.unsqueeze ~axes:[ -2 ] x)
  in
  Nx.sum ~axes:[ -2 ] (Nx.mul y (Nx.unsqueeze ~axes:[ -1 ] weights))
