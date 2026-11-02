(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Floats as decimal text, the same on every system.

    OCaml's [Printf] writes a float with the C library's printf, whose text
    differs between systems: under mingw-w64, msvcrt writes three exponent
    digits, keeps 17 digits and rounds from those, and rounds an exact tie away
    from zero. These functions take a float's exact decimal value and round it
    once, to nearest, ties to even, as C's printf does where it is correctly
    rounded: [135.5625] to six digits is [135.562]. A NaN is [nan] whatever its
    sign, and the infinities [inf] and [-inf]; a negative zero keeps its sign.
*)

val g : ?precision:int -> float -> string
(** [g ~precision x] is [x] as C's [%.*g] writes it, with [precision]
    significant digits, [6] by default and at least [1]: positional for a
    decimal exponent from [-4] to below [precision], otherwise with an exponent
    of at least two digits, without trailing zeros, as [135.562], [1e+16] and
    [1.5e-05]. *)

val f : int -> float -> string
(** [f decimals x] is [x] as C's [%.*f] writes it, with [decimals] digits after
    the point, at least [0]: [0.12] for [0.125] at two. *)

val exact : float -> string
(** [exact x] is the first of [g ~precision:15 x], [g ~precision:16 x] and
    [g ~precision:17 x] that reads back as [x], to nearest, ties to even: the
    fewest digits from 15 that name [x], as [0.1] and [0.30000000000000004]. *)

val exact_float : float -> string
(** [exact_float x] is [exact x] with [.0] added when it has neither a fraction
    nor an exponent, as [1.0] and [-0.0], so that a reader that tells integers
    from floats by their text reads a float. *)

val of_string : string -> float option
(** [of_string s] is the float nearest the decimal number [s] writes, ties to
    even, the same on every system, or [None] if [s] is not one number in JSON's
    syntax ({!number_end}). A number too large for a float is an infinity. *)

val number_end : string -> int -> int -> int
(** [number_end s i stop] is the index after the number in JSON's syntax that
    starts at [i] in [s], read no further than [stop], or [-1] if none starts
    there: an optional minus, then a zero alone or digits not starting with one,
    an optional point and digits, and an optional [e] or [E], sign and digits.
    The longest such number is taken: [number_end "0123" 0 4] is [1]. *)

val of_substring : string -> int -> int -> float option
(** [of_substring s pos len] is [of_string (String.sub s pos len)], without the
    copy.

    Raises [Invalid_argument] if [pos] and [len] do not name a substring of [s].
*)

(**/**)

val read : string -> float option
(** [read s] is [of_string s] without the reading of short decimals by sums of
    floats, which relies on [Float.fma]: tests check that reading against this
    one. *)
