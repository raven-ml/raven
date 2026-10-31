(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Type conversion and preprocessing *)

let to_float img = Nx.div_s (Nx.cast Nx.float32 img) 255.0

let to_uint8 img =
  let clipped = Nx.clamp ~min:0.0 ~max:1.0 img in
  Nx.cast Nx.uint8 (Nx.mul_s clipped 255.0)

let normalize ~mean ~std img =
  let shape = Nx.shape img in
  let rank = Array.length shape in
  let c = shape.(rank - 1) in
  if List.length mean <> c || List.length std <> c then
    invalid_arg
      (Printf.sprintf
         "normalize: mean/std length (%d/%d) does not match channels (%d)"
         (List.length mean) (List.length std) c);
  let ones = Array.make rank 1 in
  ones.(rank - 1) <- c;
  let per_channel l =
    Nx.reshape ones
      (Helpers.beside img (Nx.create Nx.float32 [| c |] (Array.of_list l)))
  in
  let mean_t = per_channel mean and std_t = per_channel std in
  Nx.div (Nx.sub img mean_t) std_t

let threshold t img = Nx.greater_s img t

(* Re-export private modules *)

include Color
include Transform
include Filter
include Morphology
include Edge
