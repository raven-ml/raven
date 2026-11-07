(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Floats to a number of digits, the same text on every system.

    OCaml's [Printf] writes a float with the C library's printf, whose text
    differs between systems: under mingw-w64, msvcrt writes three exponent
    digits, keeps 17 digits and rounds from those, and rounds an exact tie
    away from zero. {!g} and {!f} lay out the digits {!Float_text} computes
    from a float's exact decimal value, rounded once, to nearest, ties to
    even, as C's printf does where it is correctly rounded. A NaN is [nan]
    whatever its sign, and the infinities [inf] and [-inf]; a negative zero
    keeps its sign. *)

val g : int -> float -> string
(** [g p x] is [x] as C's [%.*g] writes it, with [p] significant digits, at
    least [1]: positional for a decimal exponent from [-4] to below [p],
    otherwise with an exponent of at least two digits, without trailing zeros,
    as [135.562], [1e+16] and [1.5e-05] at six. *)

val f : int -> float -> string
(** [f decimals x] is [x] as C's [%.*f] writes it, with [decimals] digits
    after the point, at least [0]: [0.12] for [0.125] at two. *)
