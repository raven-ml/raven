(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let err_rank n =
  Printf.sprintf "expected rank 3 [H;W;C] or 4 [N;H;W;C], got %d" n

let with_batch f img =
  match Array.length (Rune.shape img) with
  | 3 ->
      let batched = Rune.unsqueeze_axis 0 img in
      Rune.squeeze_axis 0 (f batched)
  | 4 -> f img
  | n -> invalid_arg (err_rank n)

let with_batch_pair f img =
  match Array.length (Rune.shape img) with
  | 3 ->
      let batched = Rune.unsqueeze_axis 0 img in
      let a, b = f batched in
      (Rune.squeeze_axis 0 a, Rune.squeeze_axis 0 b)
  | 4 -> f img
  | n -> invalid_arg (err_rank n)

let convolve_per_channel kernel img =
  let shape = Rune.shape img in
  let c = shape.(3) in
  let kshape = Rune.shape kernel in
  let kh = kshape.(0) and kw = kshape.(1) in
  let img_nchw = Rune.transpose ~axes:[ 0; 3; 1; 2 ] img in
  let k4d =
    Rune.tile [| c; 1; 1; 1 |] (Rune.reshape [| 1; 1; kh; kw |] kernel)
  in
  let out = Rune.correlate2d ~groups:c ~padding_mode:`Same img_nchw k4d in
  Rune.transpose ~axes:[ 0; 2; 3; 1 ] out
