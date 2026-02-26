(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Type conversion and preprocessing *)

let to_float img = Rune.div_s (Rune.astype Rune.float32 img) 255.0

let to_uint8 img =
  let clipped = Rune.clip ~min:0.0 ~max:1.0 img in
  Rune.astype Rune.uint8 (Rune.mul_s clipped 255.0)

let normalize ~mean ~std img =
  let shape = Rune.shape img in
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
    Rune.reshape ones (Rune.create Rune.float32 [| c |] (Array.of_list mean))
  in
  let std_t =
    Rune.reshape ones (Rune.create Rune.float32 [| c |] (Array.of_list std))
  in
  Rune.div (Rune.sub img mean_t) std_t

let threshold t img =
  let t_s = Rune.scalar_like img t in
  let one = Rune.ones_like img in
  let zero = Rune.zeros_like img in
  Rune.where (Rune.greater img t_s) one zero

(* Re-export private modules *)

include Color
include Transform
include Filter
include Morphology
include Edge
