(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Floats and the decimals that read as them, decided exactly.

    The C library rounds some decimals wrong both ways, as msvcrt does under
    mingw-w64, so reading and writing check its answers against exact
    arithmetic on naturals and move them to the nearest. *)

val split : string -> (bool * string * int) option
(** [split t] is [(negative, digits, e)] with the real text [t] (FITS 4.0
    §4.2.4) equal to [digits × 10^e], [digits] without leading or trailing
    zeros, and zero as [(negative, "", 0)]. It is [None] when the exponent
    does not fit an [int]. [t] matches the real grammar, which [split] does
    not check. *)

val read : string -> int -> float option
(** [read digits e] is the float nearest [digits × 10^e], ties to even, or
    [None] past the largest float. [digits] are decimal digits without a
    sign, as {!split} gives them. *)

val shortest : float -> int * int
(** [shortest x] is [(m, e)], [m × 10^e] the decimal of the fewest
    significant digits that reads as the positive finite [x], the nearest of
    them, ties to even. *)
