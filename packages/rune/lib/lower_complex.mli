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
    complex rows of {!Rune.section-numerics}.

    {1:exact Exact kinds}

    [Neg], [Add] and [Sub] apply the float kind to each part, and [Maximum],
    [Minimum] and the comparisons follow the order {!Nx_kernel.Prog} states.
    Each is exact ({!Lower_real}).

    {1:nx Kinds of nx's sequence}

    These kinds compute the sequence of operations nx's kernel header,
    [nx_kinds_complex.h], gives them, each fused multiply-add a
    {!Lower_bits.mulacc}, a branch of the header a selection with every side
    computed. A transcendental kind among them first sets the quiet bit of its
    operands' NaN parts.

    - [Mul] is the textbook product, each part one fused multiply-add over the
      other product.
    - [Fdiv] is Smith's algorithm, which divides by the divisor's larger part,
      over a dividend scaled by [2{^25}] or [2{^54}] where its parts are below
      the least normal over [u]; [Recip] is [(1 + 0i) / z] by [Fdiv].
    - [Sqrt] is Kahan's, of [z] scaled by an even power of two, and [Sign] is
      [z / |z|] of [z] scaled by a power of two.
    - [Log] is [log |z| + i atan2(y, x)], [log |z|] by [log1p] of exact
      products near the unit circle, and [Log2] is [Log] times [1/ln 2] in two
      parts, the float [log2] of the modulus on the axes.
    - [Exp] is [e{^x} cis y], [e{^x}] past [86] ([float32]) or [708]
      ([float64]) taken in two halves around each product so that no part
      overflows before the result. [Exp2] is [2{^x} cis(y ln 2)], its phase in
      two parts below the trigonometric kinds' switch and reduced from the bits
      of [2 ln 2 / π] past it. [Pow] is [e{^w log z}], a real or imaginary [w]
      multiplying [log z]'s parts alone, [1] at [w = 0] and the float [pow] on
      the positive real axis.

    On a target that keeps subnormals, rounds division and the square root
    correctly and rounds a fused multiply-add once, they give the bits nx's
    kernels give, a NaN part as nx's NaN rule allows: they are exact.
    Exactness has one exception: at an operand with an infinite or NaN part,
    where nx's [Pow] gives [+0 ± 0i], a kernel on an NVIDIA GPU, or one whose
    constants tolk folds, may give the zero imaginary part the opposite sign
    to nx's on the host: nx takes that sign, at some of these operands, from
    a NaN it computes in [w log z], whose sign the target or the compiler
    sets. On a target that rounds division and the square root correctly and
    rounds a fused multiply-add once, each but [Mul], [Fdiv] and [Recip] is
    within the normwise bound nx states for it where no value computed for it
    is subnormal: [3u] for [Sqrt] and [Sign], [4u] for [Log], [Log2], [Exp]
    and [Exp2], and [(3 + 3 |w log z|) u] for [Pow]; on another target no
    bound is stated for them.

    {1:bounded Other kinds}

    The transcendental kinds are the principal values of C99 7.3 (ISO/IEC
    9899), each part's sign choosing the side of a branch cut as there.

    On every target, [Mul], [Fdiv], [Recip] and the transcendental kinds below
    each hold a {e normwise} bound, in the sense that a result [z'] of exact
    value [z] has [|z' - z| / |z|] at most the table's multiple of [u], to first
    order. The bounds are derived from {!Lower_real}'s, an error of [n] ulps
    there being one of at most [(2n + 1) u]:

    {v
      mul                                      2
      fdiv, recip                              4
      atan, atanh, acos, acosh                21
      asin, asinh                             23
      cos, cosh                               57
      sin, sinh                               67
      tan, tanh                               73
    v}

    A bound holds where every part of the operands is finite, the result's
    modulus is a normal float and no value computed for it, a part of it
    included, is subnormal. [Fdiv] and [Recip] also need the operands' parts
    below half the greatest float and the divisor's larger part normal. There,
    the result has no NaN part.

    {1:special Special values}

    A part's {e sort} is whether it is a NaN, an infinity, a zero or another
    number. An operand with a NaN part and none infinite gives a result with a
    NaN part, except [z{^w}] at a [w] whose parts are zeros, which is [1], and
    at [z = 1 ± 0i] with [w]'s imaginary part zero, which is [1 ± 0i] as the
    float [pow] gives [1] at [x = 1]. At an operand with an infinite part, each
    part of a unary kind's result has the sort of that part of the value C99
    Annex G (G.6) gives, as nx's kinds do, and an infinite part has Annex G's
    sign. There, where no part is a NaN,
    [Sign] gives the direction of the infinite parts, one of the eight [±1],
    [±i] and [(±1 ± i) / √2], and [Recip] the sorts of [(1 + 0i) / z] as [Fdiv]
    computes it. Zeros' signs are unspecified, and so is an infinity's sign that
    comes from a NaN part or that Annex G leaves open. On a target where they
    are not exact, a result of [Mul], [Fdiv] or [Pow] at an operand with an
    infinite part is unspecified, and at [x + iy] with [x = +∞] and [y] finite
    an infinite part of [Exp2] may take the wrong sign where [y ln2] lies
    within a relative [2u] of a non-zero multiple of [π/2]. *)

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
