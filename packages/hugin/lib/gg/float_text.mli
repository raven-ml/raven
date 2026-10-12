(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Floats as decimal text, the same on every system.

    OCaml's [Printf] writes a float with the C library's printf, whose text
    differs between systems: under mingw-w64, msvcrt writes three exponent
    digits, keeps 17 digits and rounds from those, and rounds an exact tie
    away from zero. These functions take a float's exact decimal value and
    round it once, to nearest, ties to even, as C's printf does where it is
    correctly rounded: [135.5625] to six digits is [135.562]. A NaN is [nan]
    whatever its sign, and the infinities [inf] and [-inf]; a negative zero
    keeps its sign.

    hugin's libraries each compile this module from [gg/], privately. *)

val g : ?precision:int -> float -> string
(** [g ~precision x] is [x] as C's [%.*g] writes it, with [precision]
    significant digits, [6] by default and at least [1]: positional for a
    decimal exponent from [-4] to below [precision], otherwise with an
    exponent of at least two digits, without trailing zeros, as [135.562],
    [1e+16] and [1.5e-05]. *)

val pp : Format.formatter -> float -> unit
(** [pp ppf x] formats [g x]. *)

val f : int -> float -> string
(** [f decimals x] is [x] as C's [%.*f] writes it, with [decimals] digits
    after the point, at least [0]: [0.12] for [0.125] at two. *)

val exact : float -> string
(** [exact x] is the first of [g ~precision:15 x], [g ~precision:16 x] and
    [g ~precision:17 x] that reads back as [x], to nearest, ties to even: the
    fewest digits from 15 that name [x], as [0.1] and [0.30000000000000004]. *)

val pp_exact : Format.formatter -> float -> unit
(** [pp_exact ppf x] formats [exact x]. *)
