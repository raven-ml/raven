(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A polygon's area in a cell, edge by edge, for the modules that compute it
    ([Warp]'s area rules and [Rasterize.areas]).

    Coordinates are pixel points [(row, col)] relative to the cell's corner, so
    the cell is [\[0, 1\] × \[0, 1\]]. By vertical slices, a polygon's area in
    the cell is the signed sum over its edges of {!term}, the width each edge
    spans within a window of the cell's column times its mean height clipped
    to the cell's rows. Summed over windows that tile [\[0, 1\]], with each
    edge's sign, the terms give the polygon's area in the cell. *)

val term :
  lo:'d Nx.float64_t ->
  hi:'d Nx.float64_t ->
  'd Nx.float64_t * 'd Nx.float64_t ->
  'd Nx.float64_t * 'd Nx.float64_t ->
  'd Nx.float64_t * 'd Nx.float64_t
(** [term ~lo ~hi (yp, xp) (yq, xq)] is, for the edge from [(yp, xp)] to
    [(yq, xq)] and the window of columns [\[lo, hi\]] within [\[0, 1\]], the
    signed width [b - a] the edge spans within the window, its ends [xp] and
    [xq] clamped to it, times the mean over that span of the edge's height
    clipped to [\[0, 1\]], and the width's magnitude. [lo] and [hi] broadcast
    against the coordinates. At [lo = 0] and [hi = 1] the window is the cell's
    column.

    A vertical edge spans no width; its two ends are taken as its heights, so
    that a width it gains has its derivative. The clamps have derivative 1
    inside the window and 0 elsewhere, the right-hand derivative at its ends,
    and the edge's parameter at an end, clamped to \[0, 1\], keeps its
    derivative at those bounds, which is the right-hand one too.

    Each term errs by at most (15M + 5) u |b - a| to first order, with u =
    2{^-53} and M ≥ 1 the largest magnitude of the edge's coordinates. Above
    the cell its heights are 0, exact; below it they are within u (3M + 3) of
    1. The heights' ends err by at most 12 u M: the coordinates' subtraction
    from the cell's corner u M, the edge's parameter 3 u times [|dy| ≤ 2M],
    [dy] 2 u M, the product 2 u M and the sum u M. The mean adds 3 u M + 3 u,
    and the width and the product 2 u. *)
