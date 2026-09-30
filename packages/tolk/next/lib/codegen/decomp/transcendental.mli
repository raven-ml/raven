(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Transcendental functions as arithmetic.

    A target without instructions for [exp2], [log2], [sin] or [sqrt] computes
    them from additions, multiplications, comparisons, selections, casts and bit
    reinterpretations. The functions below build those computations as nodes:
    polynomial approximations on a reduced argument, with the reduction and the
    reconstruction done on the bits of the floating-point representation.

    [exp2], [log2] and [sin] are defined in {!Dtype.Float16}, {!Dtype.Float32}
    and {!Dtype.Float64}, and return IEEE's special values at infinities, NaNs
    and zeros. Elsewhere their results are within a relative error of [5e-3]
    plus an absolute error of [1e-2] of the exact value in {!Dtype.Float16},
    [1e-5] plus [2e-5] in {!Dtype.Float32}, and [1e-5] plus [3e-2] in
    {!Dtype.Float64}, whose sine loses absolute precision as its argument grows.

    {b Errors.} A function given a node of another type than it is defined for
    raises [Invalid_argument]. *)

(** {1:bits Bit manipulation} *)

val exponent_bias : Dtype.t -> int
(** [exponent_bias dt] is the bias of the exponent field of the float [dt]:
    [2]{^ [e-1]}[ - 1] for [e] exponent bits ({!Dtype.finfo}), and one more for
    {!Dtype.Fp8e4m3fnuz} and {!Dtype.Fp8e5m2fnuz}.

    Raises [Invalid_argument] if [dt] is not in {!Dtype.floats}. *)

val shl : Ops.t -> int -> Ops.t
(** [shl x n] is [x * 2]{^ [n]}: [x] shifted left by [n] bits if [x] is an
    integer.

    Raises [Invalid_argument] if [n] is negative. *)

val shr : Ops.t -> int -> Ops.t
(** [shr x n] is [x // 2]{^ [n]}, the division rounding towards negative
    infinity: [x] shifted right by [n] bits if [x] is an integer.

    Raises [Invalid_argument] if [n] is negative. *)

val rintk : Ops.t -> Ops.t
(** [rintk d] is the float [d] rounded to the nearest integer, halves away from
    zero, as the signed integer of [d]'s width: {!Dtype.Int16}, {!Dtype.Int32}
    or {!Dtype.Int64}. *)

val pow2if : Ops.t -> Dtype.t -> Ops.t
(** [pow2if q dt] is 2{^ [q]} as a float, for [q] an integer in the range of
    normal exponents of that float: {!Dtype.Float64} for an {!Dtype.Int64} [q],
    {!Dtype.Float32} for an {!Dtype.Int32} [q], and [dt] for an {!Dtype.Int16}
    [q]. *)

val frexp : Ops.t -> Ops.t * Ops.t
(** [frexp v] is [(m, e)] with [|v| = |m| * 2]{^ [e]} and [0.5 <= |m| < 1], for
    a normal [v]. [m] has [v]'s type, and [v]'s sign in {!Dtype.Float16} and
    {!Dtype.Float32}; in {!Dtype.Float64} it is positive. [e] has the unsigned
    integer type of [v]'s width, and wraps around for [|v| < 0.5]. *)

(** {1:reductions Argument reductions}

    A reduction brings an angle [d >= 0] near zero, where a polynomial
    approximates the sine, and says which multiple of a half-turn or of a
    quarter-turn it removed. *)

val payne_hanek_reduction : Ops.t -> Ops.t * Ops.t
(** [payne_hanek_reduction d] is [(r, q)] with [d = q * pi/2 + r] and
    [|r| <= pi/4], for [1 <= d], infinities excluded. [r] has [d]'s type and [q]
    is a {!Dtype.Int32} whose value modulo 4 is the quadrant of [d]. It
    multiplies [d] by 190 bits of [2/pi], which keeps [r] exact to [d]'s
    precision whatever [d]'s magnitude. *)

val cody_waite_reduction : Ops.t -> Ops.t * Ops.t
(** [cody_waite_reduction d] is [(r, q)] with [d = q * pi + r] and
    [|r| <= pi/2], for [|d| <= 39800]. [r] has [d]'s type and [q] is a
    {!Dtype.Int32}. It subtracts [q * pi] in several parts whose products are
    exact, which is precise only while [q] is small. *)

(** {1:functions Functions} *)

val xsin : ?fast:bool -> ?switch_over:float -> Ops.t -> Ops.t
(** [xsin ~fast ~switch_over d] is the sine of [d], NaN at infinities and NaN.
    Angles below [switch_over] (default [30.]) in magnitude are reduced with
    {!cody_waite_reduction}, and larger ones with {!payne_hanek_reduction},
    unless [fast] (default [false]), which assumes every angle is below
    [switch_over] and builds only the first. *)

val xexp2 : Ops.t -> Ops.t
(** [xexp2 d] is [2]{^ [d]}: [inf] where it overflows [d]'s type, [0] where it
    underflows past the subnormals, and NaN at NaN. *)

val xlog2 : Ops.t -> Ops.t
(** [xlog2 d] is the base-2 logarithm of [d], subnormals included: [inf] at
    [inf], [-inf] at both zeros, and NaN below zero and at NaN. *)

val xpow : Ops.t -> Ops.t -> Ops.t
(** [xpow base exponent] is [base] to the power [exponent], computed as
    [exp2 (exponent * log2 |base|)]. A negative [base] gives NaN for an exponent
    that is not an integer, except for [-inf], and the result's sign follows the
    exponent's parity otherwise. Any [base] to the power [0] is [1], NaN and
    infinities included. *)

(** {1:patterns Rewriting} *)

val patterns : force:bool -> Op.Set.t -> (unit, Ops.t) Ops.Pattern_matcher.t
(** [patterns ~force ops] rewrites each {!Op.Exp2}, {!Op.Log2} and {!Op.Sin}
    whose operation is not in [ops], or every one if [force], to {!xexp2},
    {!xlog2} and {!xsin}. On the other floats, {!Dtype.Bfloat16} and the 8-bit
    ones, the operation is computed in {!Dtype.Float32} and cast back. An
    {!Op.Sqrt} under the same condition becomes [xpow d 0.5]. *)
