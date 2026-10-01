(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Arbitrary-precision integers.

    Integers are exact: no operation wraps or rounds unless it says so. An
    integer that fits an [int] is held as one, so arithmetic on such integers
    allocates nothing while its result fits an [int] too.

    Each integer has one representation: polymorphic equality and [Hashtbl.hash]
    apply to integers, and an integer that fits an [int] hashes as that [int].
    Polymorphic comparison does not order them: use {!compare}. *)

type t
(** The type for integers. *)

exception Overflow
(** Raised when a conversion's result does not fit its type. *)

(** {1:consts Constants} *)

val zero : t
(** [zero] is [0]. *)

val one : t
(** [one] is [1]. *)

val minus_one : t
(** [minus_one] is [-1]. *)

(** {1:conv Conversions} *)

val of_int : int -> t
(** [of_int n] is [n]. *)

val of_int32 : int32 -> t
(** [of_int32 n] is [n]. *)

val of_int64 : int64 -> t
(** [of_int64 n] is [n]. *)

val of_nativeint : nativeint -> t
(** [of_nativeint n] is [n]. *)

val of_int32_unsigned : int32 -> t
(** [of_int32_unsigned n] is [n] read as unsigned, in \[[0];[2{^32}-1]\]. *)

val of_int64_unsigned : int64 -> t
(** [of_int64_unsigned n] is [n] read as unsigned, in \[[0];[2{^64}-1]\]. *)

val of_float : float -> t
(** [of_float x] is [x] rounded toward zero.

    Raises {!Overflow} if [x] is infinite or NaN. *)

val of_string : string -> t
(** [of_string s] is the integer [s] writes: an optional [-] or [+], an optional
    [0x], [0o] or [0b] for base 16, 8 or 2 (base 10 otherwise), then digits of
    the base, where underscores may follow the first digit.

    Raises [Invalid_argument] if [s] is anything else. *)

val fits_int : t -> bool
(** [fits_int n] is [true] iff [n] is in [int]'s range. *)

val fits_int64 : t -> bool
(** [fits_int64 n] is [true] iff [n] is in [int64]'s range. *)

val to_int : t -> int
(** [to_int n] is [n].

    Raises {!Overflow} if [n] is out of [int]'s range. *)

val to_int32 : t -> int32
(** [to_int32 n] is [n].

    Raises {!Overflow} if [n] is out of [int32]'s range. *)

val to_int64 : t -> int64
(** [to_int64 n] is [n].

    Raises {!Overflow} if [n] is out of [int64]'s range. *)

val to_int32_unsigned : t -> int32
(** [to_int32_unsigned n] is [n] stored as an unsigned [int32].

    Raises {!Overflow} if [n] is out of \[[0];[2{^32}-1]\]. *)

val to_int64_unsigned : t -> int64
(** [to_int64_unsigned n] is [n] stored as an unsigned [int64].

    Raises {!Overflow} if [n] is out of \[[0];[2{^64}-1]\]. *)

val to_float : t -> float
(** [to_float n] is the float nearest to [n], ties to even: an infinity if [n]
    is past the greatest float. *)

val to_string : t -> string
(** [to_string n] is [n] in decimal, with a [-] if negative. *)

val pp_print : Format.formatter -> t -> unit
(** [pp_print] formats integers as {!to_string} does. *)

(** {1:order Predicates and comparisons} *)

val sign : t -> int
(** [sign n] is [-1], [0] or [1] as [n] is negative, zero or positive. *)

val compare : t -> t -> int
(** [compare] orders integers by value. *)

val equal : t -> t -> bool
(** [equal n0 n1] is [compare n0 n1 = 0]. *)

val leq : t -> t -> bool
(** [leq n0 n1] is [n0 <= n1]. *)

val geq : t -> t -> bool
(** [geq n0 n1] is [n0 >= n1]. *)

val lt : t -> t -> bool
(** [lt n0 n1] is [n0 < n1]. *)

val gt : t -> t -> bool
(** [gt n0 n1] is [n0 > n1]. *)

val min : t -> t -> t
(** [min n0 n1] is the least of [n0] and [n1]. *)

val max : t -> t -> t
(** [max n0 n1] is the greatest of [n0] and [n1]. *)

(** {1:arith Arithmetic} *)

val neg : t -> t
(** [neg n] is [-n]. *)

val abs : t -> t
(** [abs n] is the absolute value of [n]. *)

val succ : t -> t
(** [succ n] is [n + 1]. *)

val pred : t -> t
(** [pred n] is [n - 1]. *)

val add : t -> t -> t
(** [add n0 n1] is [n0 + n1]. *)

val sub : t -> t -> t
(** [sub n0 n1] is [n0 - n1]. *)

val mul : t -> t -> t
(** [mul n0 n1] is [n0 * n1]. *)

val div : t -> t -> t
(** [div a b] is [a / b] rounded toward zero.

    Raises [Division_by_zero] if [b] is zero. *)

val rem : t -> t -> t
(** [rem a b] is [a - b * div a b]: it has [a]'s sign and is smaller than [b] in
    magnitude.

    Raises [Division_by_zero] if [b] is zero. *)

val fdiv : t -> t -> t
(** [fdiv a b] is [a / b] rounded toward negative infinity.

    Raises [Division_by_zero] if [b] is zero. *)

val cdiv : t -> t -> t
(** [cdiv a b] is [a / b] rounded toward positive infinity.

    Raises [Division_by_zero] if [b] is zero. *)

val divisible : t -> t -> bool
(** [divisible a b] is [true] iff [b] divides [a]: [a = b * q] for an integer
    [q]. Only [zero] is divisible by [zero]. *)

val ediv : t -> t -> t
(** [ediv a b] is the Euclidean quotient [q] of [a] by [b], the one for which
    [0 <= a - b * q < abs b].

    Raises [Division_by_zero] if [b] is zero. *)

val erem : t -> t -> t
(** [erem a b] is [a - b * ediv a b], in \[[0];[abs b - 1]\].

    Raises [Division_by_zero] if [b] is zero. *)

val gcd : t -> t -> t
(** [gcd a b] is the greatest common divisor of [a] and [b], never negative:
    [gcd a zero] is [abs a]. *)

val pow : t -> int -> t
(** [pow b e] is [b] to the power [e].

    Raises [Invalid_argument] if [e] is negative. *)

val sqrt : t -> t
(** [sqrt n] is the square root of [n] rounded toward zero.

    Raises [Invalid_argument] if [n] is negative. *)

(** {1:bits Bits}

    Bitwise operations read integers in two's complement, extended to infinitely
    many bits: a negative integer has infinitely many ones on its left. *)

val logand : t -> t -> t
(** [logand n0 n1] is the bitwise and of [n0] and [n1]. *)

val logor : t -> t -> t
(** [logor n0 n1] is the bitwise or of [n0] and [n1]. *)

val logxor : t -> t -> t
(** [logxor n0 n1] is the bitwise exclusive or of [n0] and [n1]. *)

val lognot : t -> t
(** [lognot n] is the bitwise complement of [n], [-n - 1]. *)

val shift_left : t -> int -> t
(** [shift_left n k] is [n * 2{^k}].

    Raises [Invalid_argument] if [k] is negative. *)

val shift_right : t -> int -> t
(** [shift_right n k] is [n / 2{^k}] rounded toward negative infinity.

    Raises [Invalid_argument] if [k] is negative. *)

val extract : t -> int -> int -> t
(** [extract n off len] is the non-negative integer of bits [off] to
    [off + len - 1] of [n].

    Raises [Invalid_argument] if [off] is negative or [len] is not positive. *)

val signed_extract : t -> int -> int -> t
(** [signed_extract n off len] is [extract n off len] read as a [len]-bit two's
    complement integer, in \[[-2{^len-1}];[2{^len-1}-1]\].

    Raises [Invalid_argument] if [off] is negative or [len] is not positive. *)

val numbits : t -> int
(** [numbits n] is the number of bits of [abs n]: [0] for zero, and otherwise
    the [k] for which [2{^k-1} <= abs n < 2{^k}]. *)

val trailing_zeros : t -> int
(** [trailing_zeros n] is the greatest [k] for which [2{^k}] divides [n], or
    [max_int] if [n] is zero. *)

val popcount : t -> int
(** [popcount n] is the number of one bits of [n].

    Raises {!Overflow} if [n] is negative: it has infinitely many. *)

(** {1:ops Operators}

    For local opens: [Bigint.(a * b + one)]. *)

val ( + ) : t -> t -> t
(** [( + )] is {!add}. *)

val ( - ) : t -> t -> t
(** [( - )] is {!sub}. *)

val ( * ) : t -> t -> t
(** [( * )] is {!mul}. *)

val ( / ) : t -> t -> t
(** [( / )] is {!div}. *)

val ( lsl ) : t -> int -> t
(** [( lsl )] is {!shift_left}. *)

val ( = ) : t -> t -> bool
(** [( = )] is {!equal}. *)

val ( < ) : t -> t -> bool
(** [( < )] is {!lt}. *)

val ( <= ) : t -> t -> bool
(** [( <= )] is {!leq}. *)

val ( > ) : t -> t -> bool
(** [( > )] is {!gt}. *)

val ( >= ) : t -> t -> bool
(** [( >= )] is {!geq}. *)
