(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Arbitrary-precision natural numbers.

    The exact arithmetic that units and their conversions need: coefficients of
    a few thousand bits and the bounds of an evaluation. Values are immutable.
*)

type t
(** The type for natural numbers. *)

val zero : t
(** [zero] is [0]. *)

val one : t
(** [one] is [1]. *)

val of_int : int -> t
(** [of_int n] is [n]. Raises [Invalid_argument] if [n < 0]. *)

val to_int : t -> int option
(** [to_int n] is [Some n] if [n <= max_int] and [None] otherwise. *)

val to_int64 : t -> int64
(** [to_int64 n] is [n]'s low 64 bits as an [int64] bit pattern. *)

val is_zero : t -> bool
(** [is_zero n] is [true] iff [n] is [0]. *)

val compare : t -> t -> int
(** [compare m n] orders [m] and [n] by value. *)

val equal : t -> t -> bool
(** [equal m n] is [true] iff [m] and [n] are equal. *)

val bit_length : t -> int
(** [bit_length n] is the number of bits of [n]: [0] for [0], otherwise
    [1 + floor (log2 n)]. *)

val low_bits_zero : t -> int -> bool
(** [low_bits_zero n k] is [true] iff bits [0] to [k - 1] of [n] are zero. *)

val add : t -> t -> t
(** [add m n] is [m + n]. *)

val sub : t -> t -> t
(** [sub m n] is [m - n]. Raises [Invalid_argument] if [m < n]. *)

val mul : t -> t -> t
(** [mul m n] is [m * n]. *)

val pow : t -> int -> t
(** [pow n k] is [n] to the power [k], with [pow n 0] is [1]. Raises
    [Invalid_argument] if [k < 0]. *)

val shift_left : t -> int -> t
(** [shift_left n k] is [n * 2{^k}], for [k >= 0]. *)

val shift_right : t -> int -> t
(** [shift_right n k] is [floor (n / 2{^k})], for [k >= 0]. *)

val div_rem : t -> t -> t * t
(** [div_rem m n] is [(q, r)] with [m = q * n + r] and [r < n]. Raises
    [Division_by_zero] if [n] is [0]. *)

val rem_int : t -> int -> int
(** [rem_int n d] is [n mod d], for [0 < d < 2{^31}]. *)

val rem_ints : t -> int array -> int array
(** [rem_ints n ds] is [n mod d] for each [d] of [ds], each [0 < d < 2{^31}]. *)

val div_int : t -> int -> t
(** [div_int n d] is [floor (n / d)], for [0 < d < 2{^31}]. *)

val root : t -> int -> t
(** [root n k] is [floor (n{^1/k})], for [k >= 1]. *)

val of_digits : string -> t
(** [of_digits s] is the natural that the decimal digits [s] denote. [s] is
    non-empty and holds only ['0'] to ['9']. *)

val to_string : t -> string
(** [to_string n] is [n] in decimal, without leading zeros. *)
