(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let err_rank n =
  Printf.sprintf "expected rank 3 [H;W;C] or 4 [N;H;W;C], got %d" n

let with_batch f img =
  match Array.length (Nx.shape img) with
  | 3 ->
      let batched = Nx.unsqueeze_axis 0 img in
      Nx.squeeze_axis 0 (f batched)
  | 4 -> f img
  | n -> invalid_arg (err_rank n)

let with_batch_pair f img =
  match Array.length (Nx.shape img) with
  | 3 ->
      let batched = Nx.unsqueeze_axis 0 img in
      let a, b = f batched in
      (Nx.squeeze_axis 0 a, Nx.squeeze_axis 0 b)
  | 4 -> f img
  | n -> invalid_arg (err_rank n)

let convolve_per_channel kernel img =
  (* img: (N, H, W, C), kernel: (kH, kW) *)
  let shape = Nx.shape img in
  let n = shape.(0) in
  let h = shape.(1) in
  let w = shape.(2) in
  let c = shape.(3) in
  (* NCHW then merge N*C into leading: (N*C, H, W) *)
  let img_nchw = Nx.transpose ~axes:[ 0; 3; 1; 2 ] img in
  let merged = Nx.reshape [| n * c; h; w |] img_nchw in
  (* correlate on last 2 dims with Same padding *)
  let result = Nx.correlate ~padding:`Same merged kernel in
  let out_shape = Nx.shape result in
  let oh = out_shape.(1) in
  let ow = out_shape.(2) in
  (* Reshape back: (N, C, H_out, W_out) -> (N, H_out, W_out, C) *)
  let result = Nx.reshape [| n; c; oh; ow |] result in
  Nx.transpose ~axes:[ 0; 2; 3; 1 ] result
