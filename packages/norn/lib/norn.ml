(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type ('u, 'f) density = 'u -> (float, 'f) Nx.t

(* [rows_dot u lp g dx] is, for each chain, the inner product of the chain's
   rows of [g] and [dx] over every float tensor, at [lp]'s dtype. It is linear
   in [dx]. *)
let rows_dot (type f) u (lp : (float, f) Nx.t) g dx =
  let dt = Nx.dtype lp in
  let row_sum (type a b) (x : (a, b) Nx.t) =
    let c = (Nx.shape x).(0) in
    Nx.cast dt (Nx.sum ~axes:[ 1 ] (Nx.reshape [| c; -1 |] x))
  in
  let products =
    Nx.Ptree.map2 u
      (fun _ g dx -> if Nx_dtype.is_float (Nx.dtype g) then Nx.mul g dx else g)
      g dx
  in
  (* The sum starts at the first float tensor's term: a tangent map adds no
     value to a tangent, zero included. *)
  let sum =
    Nx.Ptree.fold u
      (fun _ x acc ->
        if not (Nx_dtype.is_float (Nx.dtype x)) then acc
        else
          match acc with
          | None -> Some (row_sum x)
          | Some acc -> Some (Nx.add acc (row_sum x)))
      products None
  in
  match sum with Some s -> s | None -> Nx.zeros_like lp

let with_gradient u f =
  Rune.custom_jvp u Nx.Ptree.tensor (fun x ->
      let lp, g = f x in
      (lp, fun dx -> rows_dot u lp g dx))

module Support = Support
module Bij = Bij
module Stats = Stats
module Draws = Draws
module Diag = Diag
module Summary = Summary
module Dist = Dist
module Gaussian = Gaussian
module Nuts = Nuts
module Hmc = Hmc
