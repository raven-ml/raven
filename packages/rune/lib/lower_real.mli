(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The float kinds on [float32] and [float64] nodes.

    Each kind of {!Nx_kernel.Prog} that floats take is a function of nodes of
    one float type, [float32] or [float64], applied to each element. The
    {e unit roundoff} [u] of a type is [2{^-24}] for [float32] and [2{^-53}] for
    [float64]. An {e ulp} is one rank in the order of a type's bit patterns,
    [-0] and [+0] one rank apart, and an error of [n] ulps is a distance of [n]
    ranks from the correctly rounded result. A kind is {e exact} when it gives
    the bits {!Nx_kernel.Prog} states for it, a NaN result as nx's NaN rule
    allows ({!Nx_array.Dtype.section-stores}), here and in the kinds
    {!Lower_complex} and {!Lower_arith} build from these. These rules hold for
    every operand on a target that keeps subnormals. On a target that flushes
    them, as Metal's [float32] does, a kind may be inexact or break its bound
    where an operand, a result or an intermediate is below the least normal.
    This module's kinds are the real functions of {!Rune.section-numerics}.

    {1:exact Exact kinds}

    [Neg], [Abs], [Sign], [Floor], [Ceil], [Round], [Trunc], [Mod], [Maximum],
    [Minimum], the comparisons, [Add], [Sub], [Mul] and {!fma} are exact.
    {!Nx_kernel.Prog} states the last four as IEEE 754's correctly rounded
    operations. [Fdiv], [Recip] (as [1 / x]) and [Sqrt] are exact on a target
    that rounds division and the square root correctly.

    {1:transcendental Transcendental kinds}

    [Exp], [Exp2], [Expm1], [Log], [Log2], [Log1p], [Sin], [Cos], [Tan],
    [Asin], [Acos], [Atan], [Atan2], [Sinh], [Cosh], [Tanh], [Erf] and [Pow]
    compute the sequence of operations nx's kernel header, [nx_kinds.h],
    gives them, each fused multiply-add a {!Lower_bits.mulacc}. The
    exponentials and hyperbolic kinds reduce [x] to [k ln 2 + r], [k] an
    integer, and take nx's polynomial in [r], [2{^k}] applied by exact
    products. The logarithms reduce [x] to [2{^k} (1 + f)], [1 + f] in
    [\[√½, √2)], and take nx's polynomial in [(f / (2 + f)){^2}]; [Pow]
    multiplies such a [log2] in two parts by [y] and takes [Exp2]'s
    polynomial of the product. The trigonometric kinds reduce by [π/2], Cody
    and Waite's reduction below [2{^15}] ([float32]) or [2{^28}] ([float64])
    and Payne and Hanek's from there, then take nx's polynomials. The inverse
    trigonometric kinds take nx's polynomials of [arcsin] and [arctan] on
    reduced arguments, summed with [π/2] or [π] in two parts. A branch of the
    header is a selection, both sides computed. On a target that keeps
    subnormals, rounds division and the square root correctly and rounds a
    fused multiply-add once, they give the bits nx's kernels give, a NaN
    result as nx's NaN rule allows: they are exact.

    Where every operand is [±0], [±inf] or a NaN, a kind gives the zero,
    infinity, [±1] or NaN that C99 Annex F (ISO/IEC 9899, F.9) gives, with the
    zero's sign. So [sin], [tan], [asin], [atan], [sinh], [tanh], [erf], [expm1]
    and [log1p] keep [-0], [exp (-inf)] is [+0], [log 0] is [-inf] and
    [tanh inf] is [1]. [pow] gives F.9.4.4's [1] where [y] is [±0] or [x] is
    [1], NaN included, and where [x] is [-1] and [y] is infinite. A kind gives a
    NaN wherever Annex F does. *)

open Tolk

(** {1:kinds Lowering the kinds} *)

val unary : Nx_kernel.Prog.unary -> Ops.t -> Ops.t
(** [unary k x] is [k] of each element of the [float32] or [float64] node [x].

    Raises [Invalid_argument] for [Asinh], [Acosh] and [Atanh], which floats do
    not take. *)

val binary : Nx_kernel.Prog.binary -> Ops.t -> Ops.t -> Ops.t
(** [binary k x y] is [k] of the elements of [x] and [y], two nodes of one float
    type.

    Raises [Invalid_argument] for [Idiv], [And], [Or], [Xor] and [Threefry],
    which floats do not take. *)

val compare : Nx_kernel.Prog.compare -> Ops.t -> Ops.t -> Ops.t
(** [compare k x y] is the boolean comparison [k] of the elements of [x] and
    [y]. *)

val fma : Ops.t -> Ops.t -> Ops.t -> Ops.t
(** [fma a b c] is {!Lower_bits.mulacc}[ a b c], [a * b + c] rounded once. [a],
    [b] and [c] must be nodes of one float type. *)

val sin_near : Ops.t -> Ops.t
(** [sin_near x] is {!unary}[ Sin x] wherever [|x|] is below the switch to
    Payne and Hanek's reduction, [2{^15}] at [float32] and [2{^28}] at
    [float64], or is not finite: [nx_kinds.h]'s [nx_sin_near], Cody and
    Waite's reduction alone, for a caller whose argument is bounded below the
    switch. Elsewhere its value is unspecified. *)

val cos_near : Ops.t -> Ops.t
(** [cos_near x] is {!unary}[ Cos x] where {!sin_near} is [Sin]'s:
    [nx_kinds_real.h]'s [nx_cos_near]. *)

(** {1:helpers Building blocks of the complex kinds}

    These values build {!Lower_complex}'s kinds on nodes of one float type,
    [x]'s. A constant is a pair, its value at [float32] then at [float64]. *)

val pick : Ops.t -> 'a * 'a -> 'a
(** [pick x p] is the member of [p] for [x]'s type. *)

val con : Ops.t -> float * float -> Ops.t
(** [con x p] is the constant {!pick}[ x p] of [x]'s type and shape. *)

val mbits : int * int
(** [mbits] is the number of fraction bits, [23] and [52]: [nx_kinds.h]'s
    [MBITS]. *)

val of_word : Ops.t -> int64 * int64 -> Ops.t
(** [of_word x p] is the float of [x]'s type and shape whose bits are
    {!pick}[ x p]. *)

val sqrt_half_bits : int64 * int64
(** [sqrt_half_bits] is the bits of √½ rounded: [nx_kinds.h]'s
    [SQRT_HALF_BITS]. *)

val ln2 : float * float
(** [ln2] is ln 2 rounded. *)

val log2e : float * float
(** [log2e] is 1 / ln 2 rounded. *)

val log2e_lo : float * float
(** [log2e_lo] is 1 / ln 2 less {!log2e}, rounded: [nx_kinds.h]'s
    [LOG2E_LO]. *)

val copysign : Ops.t -> Ops.t -> Ops.t
(** [copysign m s] is [m] with [s]'s sign bit. *)

val flip : Ops.t -> Ops.t -> Ops.t
(** [flip m s] is [m] with its sign bit flipped where [s]'s is set. *)

val scale : Ops.t -> Ops.t -> Ops.t
(** [scale y k] is [y × 2{^k}] for the [int32] node [k], as [y] times
    [2{^⌊k/2⌋}] times [2{^k - ⌊k/2⌋}]. Both powers must be normal floats. It
    rounds once where [y 2{^⌊k/2⌋}] is normal, and may round twice elsewhere. *)

val ilogb : Ops.t -> Ops.t
(** [ilogb x] is the [int32] exponent [e] of [x], [x = m 2{^e}] with
    [1 <= m < 2]. [x] must be finite and positive, a subnormal included. *)

val two_sum : Ops.t -> Ops.t -> Ops.t * Ops.t
(** [two_sum a b] is [(hi, lo)], [hi] the rounded sum and [a + b = hi + lo]
    exactly. [a + b] must not overflow. *)

val ln2_hi : float * float
(** [ln2_hi] is ln 2's head, few enough bits that its product by the
    reductions' [k] is exact: [nx_kinds.h]'s [LN2_HI]. *)

val ln2_lo : float * float
(** [ln2_lo] is ln 2 less {!ln2_hi}, rounded: [LN2_LO]. *)

val pio2_hi : float * float
(** [pio2_hi] is π/2's head: [nx_kinds.h]'s [PIO2_HI]. *)

val pio2_lo : float * float
(** [pio2_lo] is π/2 less {!pio2_hi}, rounded: [PIO2_LO]. *)

val exp2_lo : float * float
(** [exp2_lo] is [nx_kinds.h]'s [EXP2_LO], the clamp below which [2{^x}]
    rounds to [0]. *)

val shifter : Ops.t -> Ops.t
(** [shifter x] is [NX_SHIFT], [1.5 × 2{^p - 1}] for [p] the type's fraction
    bits, of [x]'s type and shape: adding it rounds a float below
    [2{^p - 2}] to the nearest integer. *)

val nearest : Ops.t -> Ops.t * Ops.t
(** [nearest t] is [nx_nearest]: from [t = v + ]{!shifter}, the integer
    nearest [v] as a float and as an [int32] node. *)

val exp_reduce : Ops.t -> Ops.t * Ops.t * Ops.t
(** [exp_reduce x] is [nx_exp_reduce]: [(r, rc, k)] with [x = k ln 2 + r],
    [k] the [int32] integer nearest [x / ln 2] and [rc] [r]'s rounding
    error. *)

val exp_q : Ops.t -> Ops.t
(** [exp_q r] is [nx_exp_q]: [Q(r)] with [expm1 r = r + r{^2} Q(r)]. *)

val exp2_poly : Ops.t -> Ops.t
(** [exp2_poly r] is [nx_exp2_poly]: [2{^r}] for [|r| <= 1/2]. *)

val trig_big : Ops.t -> Ops.t
(** [trig_big x] is [nx_trig_big]: whether [|x|] is finite and at or past
    the switch to Payne and Hanek's reduction. *)

val rem_ln2 : Ops.t -> Ops.t * Ops.t * Ops.t
(** [rem_ln2 y] is [nx_rem_pio2_big] of [nx_two_ln2_over_pi]: [y ln 2] as
    [q π/2 + hi + lo], [(hi, lo, q)] with [q] a [uint32] node, from the bits
    of [2 ln 2 / π]. [|y|] must be finite and past the switch {!trig_big}
    tests. *)

val sin_at : Ops.t -> Ops.t * Ops.t * Ops.t -> Ops.t
(** [sin_at x r] is [nx_sin_at]: the sine of [x] from its reduction [r], as
    {!rem_ln2} gives one. *)

val cos_at : Ops.t -> Ops.t * Ops.t * Ops.t -> Ops.t
(** [cos_at x r] is [nx_cos_at], as {!sin_at}. *)
