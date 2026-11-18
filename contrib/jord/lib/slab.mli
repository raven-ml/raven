(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** An even-odd region by vertical slabs, for the modules that integrate one
    ([Rasterize.areas], [Polygons.area]).

    Edges are segments of a plane, [k]'s endpoints [(xp, yp)] and [(xq, yq)]
    at [4k] to [4k + 3] of a float array. Over an interval of abscissae
    \[lo, hi\], the interval is cut at every abscissa where an edge ends and
    where two edges cross, each computed once. Within a {e slab} between two
    cuts no edge ends and, but for rounding, none crosses another, so the
    edges spanning it are ordered by height, sorted once, and the even-odd
    region over the slab lies between the first and the second, the third and
    the fourth, and so on. An edge's {e sign} in a slab is -1 where it is the
    lower side of the region, at an even rank from the least height, and +1
    where it is the upper side; the region's measure over the slab is the
    signed sum of the edges' heights there. A {e window} of an edge is a run
    of consecutive slabs where its sign stays the same. *)

type t
(** The type for scratch space, reused across calls. *)

val create : unit -> t
(** [create ()] is empty scratch space. *)

val windows : t -> float array -> int -> lo:float -> hi:float -> int
(** [windows s e n ~lo ~hi] is the number of windows of the [n] edges of [e]
    over \[lo, hi\], which {!edges}, {!ends} and {!signs} hold until the next
    call. The windows of the edges spanning a slab tile it, and an edge
    spanning no width of \[lo, hi\], as a vertical one, has none.

    Each cut and each sort key is a rounded number. A slab's edges are sorted
    by the sum of their heights at its two ends, twice the height at its
    midpoint, so that two edges it misorders lie within the keys' rounding of
    each other over the whole slab when they do not cross in it. A crossing is
    found from the edges' height difference at the ends of their common span,
    where it changes sign, and placed by interpolating it. *)

val edges : t -> int array
(** [edges s] holds window [i]'s edge at [i]. *)

val ends : t -> float array
(** [ends s] holds window [i]'s ends, where it starts and ends, at [4i] and
    [4i + 1], and its edge's height there at [4i + 2] and [4i + 3]: [yp + t
    (yq - yp)] for its parameter [t = (x - xp) / (xq - xp)] clamped to
    \[0, 1\], or an end's own height at that end. *)

val signs : t -> float array
(** [signs s] holds window [i]'s sign, -1. or 1., at [i]. *)
