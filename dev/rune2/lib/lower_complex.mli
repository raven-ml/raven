(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The complex kinds, on [complex64] and [complex128] values held as pairs of
    nodes.

    A {e number} is a pair of nodes of one shape, both [float32] or both
    [float64], its real part then its imaginary part. A number [z] is written
    [x + iy], and [w] is the second operand of a binary kind. Each kind is
    composed from {!Lower_real}'s float kinds, and [u] is the unit roundoff
    ({!Lower_real}) of the parts' type. The kinds and bounds here are the
    complex rows of {!Rune2.section-numerics}.

    {1:exact Exact kinds}

    [Neg], [Add] and [Sub] apply the float kind to each part, and [Maximum],
    [Minimum] and the comparisons follow the order {!Nx_kernel.Prog} states.
    Each is exact ({!Lower_real}).

    {1:bounded Other kinds}

    [Fdiv] is Smith's algorithm, which divides by the divisor's larger part, and
    [Recip] is [(1 + 0i) / z] by [Fdiv]. The transcendental kinds are the
    principal values of C99 7.3 (ISO/IEC 9899), each part's sign choosing the
    side of a branch cut as there. [Sign] is [z / |z|]. [Exp] overflows only
    where its result does.

    Each holds a {e normwise} bound, in the sense that a result [z'] of exact
    value [z] has [|z' - z| / |z|] at most the table's multiple of [u], to first
    order:

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

    [exp2]'s bound grows with [|y|], since its phase [y ln2] is computed in the
    parts' type, and [pow]'s with its exponent, as the float [pow]'s does. A
    bound holds where every part of the operands is finite and the result's
    modulus is a normal float. [Fdiv] and [Recip] also need the operands' parts
    below half the greatest float and the divisor's larger part normal, and
    [Pow] needs [|w| |log z|] below the greatest float. There, the result has no
    NaN part.

    {1:special Special values}

    A part's {e sort} is whether it is a NaN, an infinity, a zero or another
    number. An operand with a NaN part and none infinite gives a result with a
    NaN part, except [z{^w}] at a [w] whose parts are zeros, which is [1]. At an
    operand with an infinite part, each part of a unary kind's result has the
    sort of that part of the value C99 Annex G (G.6) gives, as nx's kinds do,
    and an infinite part has Annex G's sign. There, where no part is a NaN,
    [Sign] gives the direction of the infinite parts, one of the eight [±1],
    [±i] and [(±1 ± i) / √2], and [Recip] the sorts of [(1 + 0i) / z] as [Fdiv]
    computes it. Zeros' signs are unspecified, and so is an infinity's sign that
    comes from a NaN part or that Annex G leaves open. A result of [Mul], [Fdiv]
    or [Pow] at an operand with an infinite part is unspecified.

    At [x + iy] with [x = +∞] and [y] finite, [Exp2]'s infinite parts take the
    signs of the cosine and sine of its phase [y ln2] rounded to the parts'
    type, which is within a relative [2u] of [y ln2]. Where [y ln2] lies that
    close to a non-zero multiple of [π/2], a sign may differ from the exact
    phase's. That happens for some [y] from [|y| ≈ 2.27] and for every [y] past
    [|y| ≈ 0.57 / u]. *)

open Tolk

(** {1:kinds Lowering the kinds} *)

val unary : Nx_kernel.Prog.unary -> Ops.t * Ops.t -> Ops.t * Ops.t
(** [unary k z] is [k] of each element of [z].

    Raises [Invalid_argument] for [Abs], [Log1p], [Expm1], [Erf], [Floor],
    [Ceil], [Round] and [Trunc], which complex numbers do not take. *)

val binary :
  Nx_kernel.Prog.binary -> Ops.t * Ops.t -> Ops.t * Ops.t -> Ops.t * Ops.t
(** [binary k a b] is [k] of the elements of [a] and [b].

    Raises [Invalid_argument] for [Idiv], [Mod], [Atan2], [And], [Or], [Xor] and
    [Threefry], which complex numbers do not take. *)

val compare : Nx_kernel.Prog.compare -> Ops.t * Ops.t -> Ops.t * Ops.t -> Ops.t
(** [compare k a b] is the boolean comparison [k] of the elements of [a] and
    [b], by real part then imaginary part. It is false where a part of either is
    a NaN, and true there for [Not_equal]. *)
