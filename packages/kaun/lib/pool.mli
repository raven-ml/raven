(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Pooling, the parameter-free reductions over windows of the spatial axes of a
    [[| batch; s…; channels |]] input, the layout of {!Conv}. A window of
    extents [window] steps by [stride] (default [window]) along the [K]
    spatial axes, [K] the length of [window], over [x] padded by [padding]'s
    [(lo, hi)] per axis (default none), placed as {!Nx.conv} places a kernel
    without dilation. A larger [hi] keeps a last, partial window, and
    {!Nx.ceil_padding}, with [window] as its kernel, gives the padding that
    keeps [⌈s / stride⌉] windows. Differentiable through Rune. *)

val max :
  window:int array ->
  ?stride:int array ->
  ?padding:(int * int) array ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t
(** [max ~window x] is the greatest element of each window. Padding holds
    [-∞], or the dtype's least finite value where it has no infinity, so it
    is the result only of a window that lies in padding alone.

    Raises [Invalid_argument] if [window] is empty, [x] does not have [K + 2]
    axes, [stride] or [padding] has other than [K] entries, a window or
    stride entry is below [1], a padding is negative, or no window fits. *)

val avg :
  window:int array ->
  ?stride:int array ->
  ?padding:(int * int) array ->
  (float, 'b, 'd) Nx.t ->
  (float, 'b, 'd) Nx.t
(** [avg ~window x] is the mean of each window, padding counted as zeros.
    Each window's sum is divided by the product of [window]. The mean of a
    window's elements of [x] alone is
    [Nx.div (avg ~padding x) (avg ~padding (Nx.ones_like x))]. Raises as
    {!max}. *)
