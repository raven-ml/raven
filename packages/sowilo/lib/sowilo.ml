(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Type conversion and preprocessing *)

let to_float img = Nx.div_s (Nx.astype Nx.float32 img) 255.0

let to_uint8 img =
  let clipped = Nx.clip ~min:0.0 ~max:1.0 img in
  Nx.astype Nx.uint8 (Nx.mul_s clipped 255.0)

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
  let mean_t =
    Nx.reshape ones (Nx.create Nx.float32 [| c |] (Array.of_list mean))
  in
  let std_t =
    Nx.reshape ones (Nx.create Nx.float32 [| c |] (Array.of_list std))
  in
  Nx.div (Nx.sub img mean_t) std_t

let threshold t img =
  let t_s = Nx.scalar_like img t in
  let one = Nx.ones_like img in
  let zero = Nx.zeros_like img in
  Nx.where (Nx.greater img t_s) one zero

(* Re-export private modules *)

include Color
include Transform
include Filter
include Morphology
include Edge
