(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Thomas's elimination as scans.

   The forward sweep's pivot ratios [c'_i = sup_i / (diag_i − sub_i c'_(i−1))]
   follow a Möbius map of [c'_(i−1)], so they are the products of the 2×2
   matrices [[0, sup_i]; [−sub_i, diag_i]] applied to (0, 1), each product
   scaled by its largest entry so that none overflows. The eliminated right-hand
   sides [d'_i = (r_i − sub_i d'_(i−1)) / den_i] and the back substitution [x_i
   = d'_i − c'_i x_(i+1)] are affine maps, composed as pairs [(α, β)] for [x ↦ α
   x + β]. *)

let matrix (a00, a01, a10, a11) (b00, b01, b10, b11) =
  (* [b · a], then scaled. *)
  let open Nx in
  let p00 = add (mul b00 a00) (mul b01 a10)
  and p01 = add (mul b00 a01) (mul b01 a11)
  and p10 = add (mul b10 a00) (mul b11 a10)
  and p11 = add (mul b10 a01) (mul b11 a11) in
  let s = maximum (maximum (abs p00) (abs p01)) (maximum (abs p10) (abs p11)) in
  let s = where (equal s (zeros_like s)) (ones_like s) s in
  (div p00 s, div p01 s, div p10 s, div p11 s)

let affine (a1, b1) (a2, b2) =
  (* The map applied first is [(a1, b1)]. *)
  (Nx.mul a1 a2, Nx.add (Nx.mul a2 b1) b2)

(* [v] of shape [[n]] reshaped to broadcast against [[n] @ rest]. *)
let column r v =
  Nx.reshape (Array.append [| Nx.dim 0 v |] (Array.make (Nx.ndim r - 1) 1)) v

let shift_down v =
  let n = Nx.dim 0 v in
  Nx.concatenate ~axis:0
    [
      Nx.zeros_like (Nx.slice [ Nx.R (0, 1) ] v); Nx.slice [ Nx.R (0, n - 1) ] v;
    ]

let solve ~sub ~diag ~sup r =
  let pair = Nx.Ptree.(pair tensor tensor) in
  let quad =
    Nx.Ptree.iso
      (fun ((a, b), (c, d)) -> (a, b, c, d))
      (fun (a, b, c, d) -> ((a, b), (c, d)))
      (Nx.Ptree.pair pair pair)
  in
  let zero = Nx.zeros_like diag in
  let _, p01, _, p11 =
    Nx.associative_scan quad matrix (zero, sup, Nx.neg sub, diag)
  in
  let ratio = Nx.div p01 p11 in
  let den = Nx.sub diag (Nx.mul sub (shift_down ratio)) in
  let _, d =
    Nx.associative_scan pair affine
      (column r (Nx.neg (Nx.div sub den)), Nx.div r (column r den))
  in
  let flip v = Nx.flip ~axes:[ 0 ] v in
  let _, x =
    Nx.associative_scan pair affine (flip (column r (Nx.neg ratio)), flip d)
  in
  flip x
