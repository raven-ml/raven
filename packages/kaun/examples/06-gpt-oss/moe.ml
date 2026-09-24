(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun

type 'a weight = Float of 'a | Quant of Nx_quant.t

type 'a t = {
  gate_up : 'a weight;
  gate_up_bias : 'a;
  down : 'a weight;
  down_bias : 'a;
}

let map_weight f = function Float w -> Float (f w) | Quant w -> Quant w

let map f p =
  let gate_up = map_weight f p.gate_up in
  let gate_up_bias = f p.gate_up_bias in
  let down = map_weight f p.down in
  let down_bias = f p.down_bias in
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

let take_rows ids t =
  let rows = Nx.take ~axis:0 ~indices:(Nx.reshape [| -1 |] ids) t in
  let rest = Array.sub (Nx.shape t) 1 (Nx.ndim t - 1) in
  Nx.reshape (Array.append (Nx.shape ids) rest) rows

(* [product ids w x] is each token's experts [ids], [[| tokens; k |]], applied
   to its row of [x], [[| tokens; 1; 1; inputs |]]: [[| tokens; k; 1; outputs
   |]]. *)
let product ids w x =
  match w with
  | Quant w -> Nx_quant.apply ~ids w x
  | Float w -> Nx.matmul x (Nx.contiguous (take_rows ids w))

(* The gathered float rows and the activation are materialised: a product of two
   buffers is what tolk's heuristics take for a matrix product. *)
let apply ~limit p (ids, weights) x =
  let shape = Nx.shape x in
  let width = shape.(Array.length shape - 1) in
  let k = Nx.dim (Nx.ndim ids - 1) ids in
  let ids = Nx.reshape [| -1; k |] ids in
  let weights = Nx.reshape [| -1; k |] weights in
  let bias b = Nx.unsqueeze ~axes:[ 2 ] (take_rows ids b) in
  let x = Nx.reshape [| -1; 1; 1; width |] x in
  let h = Nx.add (product ids p.gate_up x) (bias p.gate_up_bias) in
  let h = Nx.contiguous (activation ~limit h) in
  let y = Nx.add (product ids p.down h) (bias p.down_bias) in
  Nx.reshape shape
    (Nx.sum ~axes:[ 1; 2 ] (Nx.mul y (Nx.unsqueeze ~axes:[ 2; 3 ] weights)))
