(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The complex kinds on [complex64] and [complex128] numbers.

    A number is a pair of nodes of one shape, both [float32] or both [float64]:
    its real part, then its imaginary part. Each kind is composed from
    {!Lower_real}'s float kinds.

    {1:exact Exact kinds}

    [Neg], [Add] and [Sub] are the parts' kinds, [Maximum] and [Minimum] the
    extremes and the comparisons the order {!Nx_kernel.Prog} states, bit for
    bit; a NaN part they make is a NaN of no promised sign or payload.

    {1:bounded Other kinds}

    [Mul] is the textbook product, each part one fused multiply-add over the
    other product. [Fdiv] is Smith's algorithm (1962), the divisor's larger part
    dividing its smaller, and [Recip] is [(1 + 0i) / z]. The transcendental
    kinds are the principal values of C99 7.3 (ISO/IEC 9899), each part's sign
    choosing the side of a branch cut as there, and [Sign] is [z / |z|]: [Sqrt]
    is Kahan's (1987), [Log]'s real part keeps its digits near the unit circle,
    [Exp] scales by a power of two after its products so that it overflows only
    where its result does, [Tanh] is Kahan's form, and the trigonometric kinds
    are the hyperbolic ones turned by [i].

    Each holds a normwise bound, [|z' - z| / |z|] for a result [z'] of the exact
    [z], in the unit roundoff [u] of the part type ([2{^-24}] or [2{^-53}]),
    derived to first order from {!Lower_real}'s bounds:

    {v
      mul                                      2
      fdiv, recip, sqrt, sign                  4
      log                                     15
      log2                                    16
      atan, atanh                             18
      exp                                     20
      asin, acos, asinh, acosh                28
      sin, cos, sinh, cosh                    33
      tan, tanh                               77
      exp2                                    19 + |y|
      pow, z^w                                20 + 17 |w log z|
    v}

    exp2's phase [y ln2] is rounded once, so its error grows with [y]; pow's
    with its exponent, as the float [pow]'s does. A bound holds where every part
    of the operands is finite and the result's modulus is a normal float: for
    [Fdiv] and [Recip], where the operands' parts are below half the greatest
    float and the divisor's larger part is normal; for [Pow], where
    [|w| |log z|] is below the greatest float. There, a finite result has no NaN
    part.

    An operand with a NaN part and none infinite gives a result with a NaN
    part, but [z{^0}], which is [1]. At an operand with an infinite part, each
    unary kind gives, part by part, the class of the value C99 Annex G (G.6)
    gives, as nx's kinds do: a NaN, an infinity of Annex G's sign, a zero or
    another number. [Sign] gives the direction among the eight a pair of parts
    names, and [Recip] the class of Smith's quotient. Zeros' signs, and an
    infinity's sign that comes from a NaN part or that Annex G leaves open,
    are unspecified. At [x + iy], [x = +∞] and [y] finite, [Exp2]'s infinite
    parts take the signs of the cosine and sine of its phase [y ln2] rounded
    to the part type, within a relative [2u] of it: where [y ln2] lies that
    close to a non-zero multiple of [π/2], as some [y] do from [|y| ≈ 2.27]
    and every [y] does past [|y| ≈ 0.57 / u], a sign may differ from the
    exact phase's. The binary kinds state no value at an infinite part. *)

open Tolk

val unary : Nx_kernel.Prog.unary -> Ops.t * Ops.t -> Ops.t * Ops.t
(** [unary k z] is [k] of each element of [z].

    Raises [Invalid_argument] for a kind complex numbers do not take. *)

val binary :
  Nx_kernel.Prog.binary -> Ops.t * Ops.t -> Ops.t * Ops.t -> Ops.t * Ops.t
(** [binary k a b] is [k] of the elements of [a] and [b].

    Raises [Invalid_argument] for a kind complex numbers do not take. *)

val compare : Nx_kernel.Prog.compare -> Ops.t * Ops.t -> Ops.t * Ops.t -> Ops.t
(** [compare k a b] is the boolean comparison [k] of the elements of [a] and
    [b]: by real part, then imaginary part, false where a part of either is a
    NaN but for [Not_equal]. *)
