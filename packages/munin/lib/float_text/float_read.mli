(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Decimal text as floats, the same on every system.

    The C library's [strtod], behind [float_of_string], is not correctly
    rounded everywhere: under mingw-w64 it reads [3.839223843572815e-239] one
    float above its nearest. These functions read a decimal as the float
    nearest its exact value, ties to even. *)

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
