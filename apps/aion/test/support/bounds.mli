(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Where a float32 computation may land, by nx's error bounds.

    nx.mli states each function's error at float32 as its distance, in floats,
    from the exact value rounded to float32: within [k] floats means one of the
    [2k + 1] floats centred on that rounding. This metric needs no ulp, so it
    holds next to powers of two, where the floats on either side of a value are
    spaced differently. A composition's bound follows from its outer function's
    monotonicity: an increasing [f] within [k] floats, applied to any input of
    an interval, lands within [k] floats of the rounded [f] of the interval's
    ends. *)

val f32 : float -> float
(** [f32 x] is [x] rounded to float32, ties to even. *)

val move : int -> float -> float
(** [move k x] is the float32 [k] floats above the float32 [x], below for a
    negative [k], [-0.] and [+0.] counting as one float. *)

val around : int -> float -> float * float
(** [around k y] is the interval of the floats within [k] of [f32 y]. *)

val through : int -> (float -> float) -> float * float -> float * float
(** [through k f (lo, hi)] is the interval where an increasing [f] within [k]
    floats lands from an input in [[lo, hi]]. *)

val inside : float * float -> float -> bool
(** [inside (lo, hi) x] is [lo <= x && x <= hi]. *)
