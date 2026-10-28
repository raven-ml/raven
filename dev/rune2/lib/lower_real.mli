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
    the bits {!Nx_kernel.Prog} states for it, except that a NaN result of a
    float kind has an unspecified sign and payload, here and in the kinds
    {!Lower_complex} and {!Lower_arith} build from these. These rules hold for
    every operand on a target that keeps subnormals. On a target that flushes
    them, as Metal's [float32] does, a kind may be inexact or break its bound
    where an operand, a result or an intermediate is below the least normal.
    This module's kinds and bounds are the float rows of
    {!Rune2.section-numerics}.

    {1:exact Exact kinds}

    [Neg], [Abs], [Sign], [Floor], [Ceil], [Round], [Trunc], [Mod], [Maximum],
    [Minimum], the comparisons, [Add], [Sub], [Mul] and {!fma} are exact.
    {!Nx_kernel.Prog} states the last four as IEEE 754's correctly rounded
    operations. [Fdiv], [Recip] (as [1 / x]) and [Sqrt] are exact on a target
    that rounds division and the square root correctly.

    {1:transcendental Transcendental kinds}

    [Log] is computed from the exact kinds and [Fdiv] alone. The other kinds
    compose tolk's [exp2], [log2] and [sin] ({!Tolk.Op.Exp2}, {!Tolk.Op.Log2},
    {!Tolk.Op.Sin}) and its argument reductions ({!Tolk.Transcendental}) with
    the exact kinds and [Log]. Where a target computes [exp2], [log2] and [sin]
    within 4 ulps, and rounds division and the square root correctly, each kind
    is within these bounds in ulps, to first order in [u]:

    {v
      kind                        float32  float64
      log                            1        1
      exp2, log2, sin                4        4
      erf                            2        3
      atan                           2        7
      asin, acos                     5        9
      atan2                          6        8
      exp, cos                       9        9
      cosh                          10       10
      log1p                         12       12
      tanh                          18       18
      tan                           19       19
      expm1, sinh                   21       21
      pow                     9 + 6 |y log2 |x||
    v}

    [pow]'s bound grows with its result's exponent [y log2 |x|]. [log], which
    reads none of tolk's kinds, holds its bound on every target that rounds
    division correctly. On the CPU host target, tolk computes [exp2], [log2] and
    [sin] by {!Tolk.Transcendental}, within one ulp, and [exp], [sin] and [cos]
    are within 2 ulps at [float32]. [exp2] at an integer and [log2] at a power
    of two may be inexact; their bounds still hold there. The bounds hold at
    tolk's default {!Tolk.Setting.transcendental}.

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

(** {1:helpers Building blocks of the complex kinds}

    These values build {!Lower_complex}'s kinds on nodes of one float type,
    [x]'s. A constant is a pair, its value at [float32] then at [float64]. *)

val pick : Ops.t -> 'a * 'a -> 'a
(** [pick x p] is the member of [p] for [x]'s type. *)

val con : Ops.t -> float * float -> Ops.t
(** [con x p] is the constant {!pick}[ x p] of [x]'s type and shape. *)

val ln2 : float * float
(** [ln2] is ln 2 rounded. *)

val log2e : float * float
(** [log2e] is 1 / ln 2 rounded. *)

val hyp_far : float
(** [hyp_far] is the [|x|] from which [cosh] and [|sinh|] are [e{^|x|} / 2], and
    [|tanh|] is [1], to both types' precision. *)

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

val exp_parts : Ops.t -> Ops.t * Ops.t
(** [exp_parts x] is [(hi, lo)] with [x / ln 2 = hi + lo] to twice the type's
    precision, [hi] rounded. *)

val exp2_split : Ops.t -> Ops.t -> Ops.t
(** [exp2_split hi lo] is [2{^hi + lo}]. [|lo|] must be at most the gap from
    [|hi|] to the next float above it. *)
