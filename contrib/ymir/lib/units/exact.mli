(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Correct rounding of exact positive numbers.

    A number is a product of primes and π raised to rational exponents. It is
    evaluated with natural-number arithmetic, π bounded by Machin's formula, and
    rounded once. *)

(** The type for the bases of a number. *)
type base = Prime of int | Pi

(** The type for the reasons a number has no value in a dtype. *)
type error =
  | Zero  (** It rounds to 0. *)
  | Subnormal  (** It rounds to a subnormal value. *)
  | Overflow  (** It rounds past the largest finite value. *)
  | Not_integer  (** It is not an integer, for an integer dtype. *)
  | Out_of_range  (** It is an integer the integer dtype does not hold. *)
  | Too_wide  (** Its evaluation needs a natural wider than {!budget} bits. *)
  | Boolean  (** The dtype is [bool] or [bit], which hold no factor. *)

val budget : int
(** [budget] is the width in bits of the widest natural an evaluation computes:
    2{^ 16}. *)

val round : ('a, 'b) Nx_dtype.t -> (base * int * int) list -> ('a, error) result
(** [round d v] is [v] rounded once to [d], where [v] is the product of each
    [(b, num, den)]'s [b] to the power [num/den].

    A float dtype rounds as IEEE 754 roundTiesToEven does in [d], with [d]'s
    subnormals and an exponent unbounded above; the result is an error if it is
    0, subnormal or above [d]'s largest finite value. A complex dtype rounds as
    its component and has a zero imaginary part. An integer dtype holds [v] only
    if it is an integer in its range; unsigned 32 and 64-bit values are their
    bit patterns. The rounded value is exact in float64.

    The exponents are reduced, [den >= 1] and [num <> 0], and each base occurs
    once. *)
