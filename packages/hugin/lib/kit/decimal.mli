(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Exact decimal numbers, and how the kit writes them.

    {!Number} and {!Ticks} write their numbers through {!write}. *)

(* Numbers *)

type t = { neg : bool; digits : string; exp : int }
(** The type for the decimal [±digits × 10]{^ [exp]}. [digits] has no
    leading and no trailing zero, and is empty for zero, which is never
    negative and has [exp = 0]. Each number has one representation, which a
    value built by hand must respect. *)

val is_zero : t -> bool
val abs : t -> t

val shift : int -> t -> t
(** [shift k d] is [d × 10]{^ [k]}. *)

val pow10 : bool -> int -> t
(** [pow10 neg k] is [10]{^ [k]}, negative iff [neg]. *)

val mag : t -> int
(** [mag d] is the [e] of [10]{^ [e]}[ <= |d| < 10]{^ [e+1]}. [d] must not be
    zero. *)

val of_float : float -> t
(** [of_float x] is the exact value of the finite [x]. [-0.] is zero. *)

val compare_mag : t -> t -> int
(** [compare_mag a b] compares [|a|] with [|b|]. *)

val compare : t -> t -> int
val sub : t -> t -> t

(* Rounding *)

(** The type for rounding directions. [Nearest] rounds halves away from
    zero. *)
type direction = Nearest | Toward_zero | Away_from_zero

val round : direction -> int -> t -> t
(** [round dir pos d] is [d] rounded in direction [dir] to a multiple of
    [10]{^ [pos]}. *)

(* Binary formats *)

type format = { precision : int; emin : int; max : float; saturates : bool }
(** The type for binary float formats: [precision] bits of significand
    counting the implicit one, [emin] the exponent of the least normal
    value, [max] the largest finite value, and [saturates] whether a value
    past [max] is stored as [max] instead of infinity. *)

val binary64 : format
(** [binary64] is the format of OCaml's [float]. *)

val round_to_format : format -> float -> float
(** [round_to_format f x] is the finite [x] rounded to [f], ties to even. *)

val rounds_to : format -> t -> float -> bool
(** [rounds_to f c y] is [true] iff [c] rounds to [y] in [f], ties to even.
    [y] must be a finite value of [f], and [c] zero or of the sign of [y]. *)

val nearest : int -> int -> float -> float
(** [nearest n k near] is the float nearest [n × 10]{^ [k]}, ties to even.
    It moves [near] one float at a time until it is that float, so [near]
    must be within a few floats of it. *)

val shortest : float -> t
(** [shortest x] is the decimal with the fewest significant digits that
    rounds to the finite [x], the nearer to [x] of two. *)

(* Writing *)

(** {!Number.notation}. *)
type notation = Plain | Exponent | Si | Percent

(** {!Number.precision}. *)
type precision = Decimals of int | Significant of int

val superscript : int -> string
(** [superscript n] is [n] in Unicode superscript digits, with a
    superscript minus if [n < 0]. *)

val floor_div : int -> int -> int
(** [floor_div a b] is [a / b] rounded toward negative infinity. [b > 0]
    must hold. *)

val write :
  Locale.t -> group:bool -> trim:bool -> notation -> precision -> t -> string
(** [write locale ~group ~trim n p d] is [d] written in notation [n] to
    precision [p] with the strings of [locale], as {!Number.to_string}
    writes a finite number. *)
