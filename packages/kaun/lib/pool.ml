(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let check name ~kernel_size ~stride x =
  let kh, kw = kernel_size and sh, sw = stride in
  if kh <= 0 || kw <= 0 then
    Printf.ksprintf invalid_arg
      "Pool.%s: kernel_size must be positive, got (%d, %d)" name kh kw;
  if sh <= 0 || sw <= 0 then
    Printf.ksprintf invalid_arg "Pool.%s: stride must be positive, got (%d, %d)"
      name sh sw;
  let xs = Nx.shape x in
  let r = Array.length xs in
  if r < 2 then
    Printf.ksprintf invalid_arg
      "Pool.%s: input must have at least 2 axes, got rank %d" name r;
  let h = xs.(r - 2) and w = xs.(r - 1) in
  if h < kh || w < kw then
    Printf.ksprintf invalid_arg
      "Pool.%s: kernel (%d, %d) does not fit input (%d, %d)" name kh kw h w

(* The windows of the last two axes: [[...; oh; ow; kh; kw]], a view. *)
let windows ~kernel_size:(kh, kw) ~stride:(sh, sw) x =
  let r = Nx.ndim x in
  x
  |> Nx.sliding_window ~axis:(r - 2) ~window:kh ~step:sh
  |> Nx.sliding_window ~axis:(r - 1) ~window:kw ~step:sw

let window_axes x = [ Nx.ndim x; Nx.ndim x + 1 ]

let max_pool2d ~kernel_size ?stride x =
  let stride = Option.value stride ~default:kernel_size in
  check "max_pool2d" ~kernel_size ~stride x;
  Nx.max ~axes:(window_axes x) (windows ~kernel_size ~stride x)

let avg_pool2d ~kernel_size ?stride x =
  let stride = Option.value stride ~default:kernel_size in
  check "avg_pool2d" ~kernel_size ~stride x;
  Nx.mean ~axes:(window_axes x) (windows ~kernel_size ~stride x)
