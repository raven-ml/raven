(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Gradient descent on a typed parameter record: fit y = x @ w + b by
   differentiating an ordinary OCaml function of an ordinary OCaml record. *)

type 'a params = { w : 'a; b : 'a }

(* The record's structure: one [field] per field, which names it. *)
module Params = struct
  type 'a t = 'a params

  let walk c { w; b } =
    let open Nx.Ptree.Walk in
    let w = field c "w" leaf w in
    let b = field c "b" leaf b in
    { w; b }
end

let params = Nx.Ptree.instantiate (module Params)

let () =
  Nx.Rng.with_key (Nx.Rng.key 0) @@ fun () ->
  (* Synthetic data: y = x @ w_true + b_true + noise. *)
  let w_true = Nx.create Nx.float32 [| 3; 1 |] [| 2.0; -1.0; 0.5 |] in
  let b_true = 0.3 in
  let x = Nx.randn Nx.float32 [| 64; 3 |] in
  let noise = Nx.mul_s (Nx.randn Nx.float32 [| 64; 1 |]) 0.01 in
  let y = Nx.add_s (Nx.add (Nx.matmul x w_true) noise) b_true in

  (* The objective: an ordinary function of the record. *)
  let loss p =
    let pred = Nx.add (Nx.matmul x p.w) p.b in
    Nx.mean (Nx.square (Nx.sub pred y))
  in

  (* One step: gradients have the same record type as the parameters. *)
  let lr = 0.1 in
  let step p =
    let l, g = Rune.value_and_grad params loss p in
    let p =
      { w = Nx.sub p.w (Nx.mul_s g.w lr); b = Nx.sub p.b (Nx.mul_s g.b lr) }
    in
    (p, Nx.item [] l)
  in

  let p =
    ref { w = Nx.zeros Nx.float32 [| 3; 1 |]; b = Nx.zeros Nx.float32 [| 1 |] }
  in
  for i = 1 to 200 do
    let p', l = step !p in
    p := p';
    if i mod 40 = 0 then Printf.printf "step %3d  loss %.6f\n%!" i l
  done;

  Printf.printf "\nw (expected ~[2.0; -1.0; 0.5]):\n  %s\n" (Nx.to_string !p.w);
  Printf.printf "b (expected ~0.3):\n  %s\n" (Nx.to_string !p.b)
