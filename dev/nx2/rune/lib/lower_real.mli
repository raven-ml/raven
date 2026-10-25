(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The float kinds on [float32] and [float64] nodes.

    {1:exact Exact kinds}

    [Neg], [Abs], [Sign], [Floor], [Ceil], [Round], [Trunc], [Mod], [Maximum],
    [Minimum] and the comparisons give the bits {!Nx_kernel.Prog} states. [Add],
    [Sub], [Mul], [Fdiv], [Recip] (as [1 / x]), [Sqrt] and {!fma} are IEEE 754's
    operations, correctly rounded. A NaN result of any of them is a NaN, of no
    promised sign or payload.

    {1:transcendental Transcendental kinds}

    The other kinds compose tolk's [exp2], [log2] and [sin] ({!Tolk.Ops.exp2},
    {!Tolk.Ops.log2}, {!Tolk.Op.Sin}) and its argument reductions
    ({!Tolk.Transcendental}) with the exact kinds, in forms that cancel nowhere.
    Taking [exp2], [log2] and [sin] within 4 ulps, as tolk's targets compute
    them, and division and the square root correctly rounded, each kind is
    within these bounds in ulps, the distance in ordered-bit ranks from the
    correctly rounded result, [-0] and [+0] one rank apart:

    {v
      kind                        float32  float64
      exp2, log2, sin                4        4
      erf                            2        3
      atan                           2        7
      asin, acos                     5        9
      atan2                          6        8
      exp, log, cos                  9        9
      cosh                          10       10
      log1p                         12       12
      tanh                          18       18
      tan                           19       19
      expm1, sinh                   21       21
      pow                     9 + 6 |y log2 |x||
    v}

    The bounds are derived to first order in the unit roundoff [u] ([2{^-24}] or
    [2{^-53}]), an error of [n] ulps taken as a relative error of [2 n u] and a
    relative error [d] as [d / u + 1/2] ulps after the last rounding; erf's
    polynomials and float32 atan's carry their measured bounds. pow is
    [2{^y log2 |x|}], so the bound grows with its exponent: [log2 |x|]'s error
    is that of a [y] perturbed by a few [u]. [exp2] is not promised exact at
    integers, nor [log2] at powers of two: they hold their bounds there as
    elsewhere. The bounds hold at tolk's default
    {!Tolk.Setting.transcendental}, for every operand on a target that keeps
    subnormals; a target that flushes them, as Metal's float32 does, differs
    where an operand, a result or an intermediate is below the least normal.

    Where every operand is [±0], [±inf] or NaN and C99 Annex F (ISO/IEC 9899,
    F.9) gives a zero, an infinity, [±1] or a NaN, a kind gives it, a zero's
    sign included: [sin], [tan], [asin], [atan], [sinh], [tanh], [erf], [expm1]
    and [log1p] keep [-0]; [exp (-inf)] is [+0], [log 0] is [-inf] and
    [tanh inf] is [1]. [pow] gives F.9.4.4's [1] where [y] is [±0] or [x] is
    [1], NaN included, and where [x] is [-1] and [y] is infinite. Elsewhere a
    result that should be NaN is a NaN, of no promised sign or payload. *)

open Tolk

val unary : Nx_kernel.Prog.unary -> Ops.t -> Ops.t
(** [unary k x] is [k] of each element of the [float32] or [float64] node [x].
    Raises [Invalid_argument] for a kind floats do not take. *)

val binary : Nx_kernel.Prog.binary -> Ops.t -> Ops.t -> Ops.t
(** [binary k x y] is [k] of the elements of [x] and [y], two nodes of one float
    type. Raises [Invalid_argument] for a kind floats do not take. *)

val compare : Nx_kernel.Prog.compare -> Ops.t -> Ops.t -> Ops.t
(** [compare k x y] is the boolean comparison [k] of the elements of [x] and
    [y]. *)

val fma : Ops.t -> Ops.t -> Ops.t -> Ops.t
(** [fma a b c] is [a * b + c] rounded once. *)

(** {1:parts Parts of the complex kinds}

    A constant is a pair: its value at [float32], then at [float64]. *)

val pick : Ops.t -> 'a * 'a -> 'a
(** [pick x p] is the member of [p] for [x]'s type. *)

val con : Ops.t -> float * float -> Ops.t
(** [con x p] is the constant {!pick}[ x p] of [x]'s type and shape. *)

val ln2 : float * float
(** [ln2] is ln 2 rounded. *)

val log2e : float * float
(** [log2e] is 1 / ln 2 rounded. *)

val hyp_far : float
(** [hyp_far] is where [cosh], [sinh] and [tanh] are [e{^|x|} / 2] and [±1] to
    both types' precision. *)

val copysign : Ops.t -> Ops.t -> Ops.t
(** [copysign m s] is [m] with [s]'s sign bit. *)

val flip : Ops.t -> Ops.t -> Ops.t
(** [flip m s] is [m] with its sign bit flipped where [s]'s is set. *)

val scale : Ops.t -> Ops.t -> Ops.t
(** [scale y k] is [y 2{^k}] for the [int32] node [k], rounded once where
    [y 2{^k/2}] is normal. *)

val ilogb : Ops.t -> Ops.t
(** [ilogb x] is the [int32] exponent [e] of a finite [x > 0], [x = m 2{^e}]
    with [m] in \[1, 2). *)

val two_sum : Ops.t -> Ops.t -> Ops.t * Ops.t
(** [two_sum a b] is [(hi, lo)], [hi] the rounded sum and [a + b = hi + lo]
    exactly. *)

val exp_parts : Ops.t -> Ops.t * Ops.t
(** [exp_parts x] is [(hi, lo)] with [x / ln 2 = hi + lo] to twice the type's
    precision, [hi] rounded. *)

val exp2_split : Ops.t -> Ops.t -> Ops.t
(** [exp2_split hi lo] is [2{^hi + lo}] for [|lo|] at most an ulp of [hi]. *)
