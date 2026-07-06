(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** 2-D pooling.

    Pure, parameter-free window reductions over the last two axes — the spatial
    axes of the NCHW layout [[| batch; channels; height; width |]] used by
    {!Conv}. Every function is differentiable through Rune-next.

    Windows are placed wherever they fit entirely inside the input (no padding):
    pooling [height] with a window of [kh] and a stride of [sh] yields
    [(height - kh) / sh + 1] positions, and likewise for the width. All leading
    axes are preserved. *)

val max_pool2d :
  kernel_size:int * int ->
  ?stride:int * int ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [max_pool2d ~kernel_size:(kh, kw) x] is the maximum of each [kh × kw] window
    over [x]'s last two axes.

    [stride] is the [(vertical, horizontal)] step between windows and defaults
    to [kernel_size] (non-overlapping windows).

    Raises [Invalid_argument] if a kernel or stride component is not positive,
    if [x] has rank below 2, or if the window does not fit ([height < kh] or
    [width < kw]). *)

val avg_pool2d :
  kernel_size:int * int ->
  ?stride:int * int ->
  (float, 'b) Nx.t ->
  (float, 'b) Nx.t
(** [avg_pool2d ~kernel_size x] is like {!max_pool2d} except each window is
    reduced to its mean. Same defaults and errors as {!max_pool2d}. *)
