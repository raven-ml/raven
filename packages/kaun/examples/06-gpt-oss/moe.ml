(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Kaun

type 'a weight =
  | Float of 'a
  | Mxfp4 of { blocks : Mxfp4.blocks; scales : Mxfp4.scales }

type 'a t = {
  router : 'a Linear.t;
  gate_up : 'a weight;
  gate_up_bias : 'a;
  down : 'a weight;
  down_bias : 'a;
}

type form = Gather | Dense

let route ~k p x =
  let logits, experts = Nx.top_k ~k (Linear.apply p.router x) in
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

(* Weights as [...; inputs; outputs]: packed ones dequantise to the checkpoint's
   [...; outputs; inputs]. *)

let all_rows dt = function
  | Float w -> w
  | Mxfp4 { blocks; scales } ->
      Nx.matrix_transpose (Mxfp4.dequant blocks scales dt)

let selected_rows dt ids = function
  | Float w -> take_rows ids w
  | Mxfp4 { blocks; scales } ->
      Nx.matrix_transpose (Mxfp4.dequant_rows blocks scales ids dt)

let experts ~limit ~gate_up ~gate_up_bias ~down ~down_bias x =
  let h = Nx.add (Nx.matmul x gate_up) gate_up_bias in
  Nx.add (Nx.matmul (activation ~limit h) down) down_bias

let gather ~limit p dt ids weights x =
  let tokens = Nx.dim 0 x and width = Nx.dim 1 x in
  let row b = Nx.unsqueeze ~axes:[ 2 ] (take_rows ids b) in
  let y =
    experts ~limit
      ~gate_up:(selected_rows dt ids p.gate_up)
      ~gate_up_bias:(row p.gate_up_bias)
      ~down:(selected_rows dt ids p.down)
      ~down_bias:(row p.down_bias)
      (Nx.reshape [| tokens; 1; 1; width |] x)
  in
  Nx.sum ~axes:[ 1; 2 ] (Nx.mul y (Nx.unsqueeze ~axes:[ 2; 3 ] weights))

let dense ~limit p dt ids weights x =
  let n = Nx.dim 0 p.gate_up_bias in
  let row b = Nx.unsqueeze ~axes:[ 1 ] b in
  let y =
    experts ~limit ~gate_up:(all_rows dt p.gate_up)
      ~gate_up_bias:(row p.gate_up_bias) ~down:(all_rows dt p.down)
      ~down_bias:(row p.down_bias)
      (Nx.unsqueeze ~axes:[ 0 ] x)
  in
  let selected = Nx.cast dt (Nx.one_hot ~num_classes:n ids) in
  let per_expert =
    Nx.sum ~axes:[ 1 ] (Nx.mul selected (Nx.unsqueeze ~axes:[ 2 ] weights))
  in
  Nx.sum ~axes:[ 0 ]
    (Nx.mul y (Nx.unsqueeze ~axes:[ 2 ] (Nx.transpose per_expert)))

let apply form ~k ~limit p x =
  let shape = Nx.shape x in
  let width = shape.(Array.length shape - 1) in
  let x = Nx.reshape [| -1; width |] x in
  let ids, weights = route ~k p x in
  let dt = Nx.dtype x in
  let y =
    match form with
    | Gather -> gather ~limit p dt ids weights x
    | Dense -> dense ~limit p dt ids weights x
  in
  Nx.reshape shape y
