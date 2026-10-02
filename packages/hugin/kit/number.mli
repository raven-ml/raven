(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Number formats.

    A format says how to write a float: in which {!type-notation}, to which
    {!type-precision}, whether trailing zeros are trimmed and whether integer
    digits are grouped. {!to_string} writes a number in a format with the
    strings of a {!Locale.t}. A format writes each number on its own; what
    depends on a set of numbers, such as the decimals the labels of an axis
    share, is decided by {!Ticks}, which writes those labels.

    {1:writing How numbers are written}

    - {b Digits} come from the exact decimal expansion of the binary value,
      rounded to the precision with halves away from zero: [0.125] to two
      decimals is [0.13], [2.5] to no decimal is [3], and [0.1] to twenty
      decimals is [0.10000000000000000555]. They are the same on every platform.
    - {b Signs.} A negative number starts with the locale's minus sign unless
      every digit written is zero: [-0.] and [-0.001] to two decimals are both
      [0.00]. A positive number has no sign.
    - {b Trimming} removes the zeros that end the digits after the decimal
      separator, then the separator if no digit follows it: [1.50] becomes [1.5]
      and [2.00] becomes [2].
    - {b Grouping} separates the integer digits with the locale's group
      separator and group sizes: [1234567] becomes [1,234,567].
    - {b Non-finite numbers.} [nan] is written [NaN], and the infinities [∞] and
      the locale's minus sign followed by [∞], in every notation. *)

(** {1:formats Formats} *)

(** The type for notations. *)
type notation =
  | Plain  (** Positional digits: [1234.5], [0.00125]. *)
  | Exponent
      (** A mantissa, then [×10] and the exponent in superscript digits, the
          mantissa in \[[1];[10]\[ once rounded: [1.2345×10³], [5×10⁻⁷]. A
          mantissa written [1] is left out, as in [10³]. Zero is written as in
          [Plain]. *)
  | Si
      (** The number divided by [1000^K], then the SI prefix of [1000^K], for
          the [K] in \[[-10];[10]\] that puts the rounded quotient in
          \[[1];[1000]\[, or the nearer end of that range if none does. The
          prefixes from [1000^-10] to [1000^10] are [q], [r], [y], [z], [a],
          [f], [p], [n], [µ] (U+00B5 MICRO SIGN), [m], none, [k], [M], [G], [T],
          [P], [E], [Z], [Y], [R] and [Q]. The prefix is chosen after rounding,
          so [999.96] to one decimal is [1.0k]. Zero is written as in [Plain].
      *)
  | Percent  (** A hundred times the number, then [%]: [12.5%]. *)

(** The type for precisions. *)
type precision =
  | Decimals of int
      (** The number of digits after the decimal separator of what the notation
          writes: the number, the mantissa, the quotient or the percentage. *)
  | Significant of int
      (** The number of significant digits of what the notation writes, counted
          from its first nonzero digit: [1234.5] to three significant digits is
          [1230] in [Plain] and [1.23k] in [Si]. Zero, which has no significant
          digit, is written with one decimal fewer than this number. *)

type t
(** The type for number formats. *)

val v : ?trim:bool -> ?group:bool -> notation -> precision -> t
(** [v ~trim ~group n p] is the format writing numbers in notation [n] to
    precision [p], where:
    - [trim], if [true], trims trailing zeros ({!section-writing}). Defaults to
      [false].
    - [group], if [true], groups the integer digits of what the notation writes
      ({!section-writing}). Defaults to [false].

    Raises [Invalid_argument] if [p] is [Decimals d] with [d < 0] or
    [Significant s] with [s < 1]. *)

(** {1:writing_numbers Writing numbers} *)

val to_string : ?locale:Locale.t -> t -> float -> string
(** [to_string ~locale f x] is [x] written in [f] with the strings of [locale],
    which defaults to {!Locale.default}, by the rules of {!section-writing}. *)

val decimals : ('a, 'b) Nx.dtype -> float -> int
(** [decimals dtype x] is the least [d >= 0] such that [x] written in [Plain]
    notation with [d] decimals, read as a decimal and rounded directly to
    [dtype], is [x] rounded to [dtype]: the fewest decimals that reproduce [x]
    in its source dtype. It is [1] for [0.1] in [Nx.float64] and [4] for the
    float32 nearest [0.9234] (the float64 [0.92339998483657836...]) in
    [Nx.float32]. It is [0] for integer values, [nan], the infinities and every
    value of an integer dtype.

    Raises [Invalid_argument] if [dtype] is a complex or boolean dtype. *)

(** {1:comparing Comparing and formatting} *)

val equal : t -> t -> bool
(** [equal f f'] is [true] iff [f] and [f'] have equal notations, precisions,
    trimming and grouping. Equal formats write every number the same in every
    locale. *)

val pp : Format.formatter -> t -> unit
(** [pp ppf f] formats [f] for debugging, such as [plain 2 decimals trim]. *)

val pp_notation : Format.formatter -> notation -> unit
(** [pp_notation ppf n] formats [n] in lowercase, such as [percent]. *)
