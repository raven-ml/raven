(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Reductions, scans and contractions as tolk graphs.

    Each function takes nodes as {!Lower_arith} holds elements and computes what
    {!Nx_kernel.type-reduction} and {!Nx_kernel.S.contract} state, to the class
    each states:

    - {b Extremes} give nx's bits, but for a NaN: [-0] below [+0], any and all
      on booleans, complex numbers by their real parts, then their imaginary
      parts.
    - {b Integer sums and products} wrap, exactly.
    - {b Float sums and products} add or multiply their terms in the accumulator
      in an association tolk chooses: a sum of [n] terms is within
      [γ(n - 1) Σ|x|] of the exact one, a product within a relative [γ(n - 1)],
      barring overflow and underflow, [γ(k) = k u / (1 - k u)] for the
      accumulator's unit roundoff [u]. A sum whose terms are all [-0] is [+0];
      the sum of no term is [+0] and the product [1]. A complex sum adds each
      part as a float sum.
    - {b Arg} gives the extreme's position exactly, that of the first NaN term
      where the extreme is a NaN.
    - {b NaNs}: a sum, a product or an extreme with a NaN term is a NaN, of
      no promised sign or payload, and a complex extreme then has two NaN
      parts.

    {1:bounds Logsumexp and Moments}

    Their special values are {!Nx_kernel.type-reduction}'s: a [Logsumexp] is
    [-∞] for no term or terms all [-∞], [+∞] where a term is [+∞] and none
    is a NaN, and a NaN where a term is one; [Moments] are NaNs for no term
    or a NaN term, and with an infinite term and no NaN the mean is the sum
    of the infinities and the variance a NaN. A variance is never negative.

    Their numbers hold these bounds on the result in the carrier
    ({!Nx_kernel.carrier}) before its rounding into the destination, which
    adds its own, to first order in the carrier's unit roundoff [u], barring
    overflow and underflow. [E] and [L] are the bounds in ulps {!Lower_real}
    states for [Exp] and [Log], ulps from the correctly rounded result, so
    each is within a relative [(2E + 1) u] or [(2L + 1) u]; they hold at
    tolk's default {!Tolk.Setting.transcendental}.

    A reduction of [n] terms:

    {t
      | result | bound |
      |:--|:--|
      | [Logsumexp] [r] | [(1.37 (n - 1) + 2E + 1 + (2L + 1) ln n) u + u|r|] |
      | mean [μ] | [B = (n + 1) u M] |
      | variance [V] | [(n + 4) u V + 2 n u B √V + 3 (n + 2) u B²] | }

    [M] is the mean of the terms' magnitudes. A [Logsumexp] is [c + log s],
    [s = Σ exp (x - c)], [c] the terms' maximum: [s] takes [(n - 1) u] from
    its sum, [(2E + 1) u] from [exp], and at most [(n - 1) u / e] from the
    rounding of the shifts [x - c], as [|x - c| exp (x - c) <= 1/e], so
    [1 + 1/e < 1.37]; its log, of a sum within [[1, n]], [(2L + 1) u ln n].
    The variance is the corrected two-pass form of Chan, Golub and LeVeque
    ("Algorithms for computing the sample variance: analysis and
    recommendations", 1983), whose correction removes the mean's error to
    second order: [B] enters only through its products with [u].

    A scan's result [i], of [N = i + 1] terms, passes through at most
    [D = max 0 (2⌈log₂ N⌉ - 1)] combinations of {!Nx_kernel.section-scans}'
    network, [R] being the range of its terms and [N] exact in the carrier:

    {t
      | result | bound |
      |:--|:--|
      | [Logsumexp] [r] | [((2E + 3) D + (2L + 2) ln N) u + u|r|] |
      | mean [μ] | [(4 D + 1) u R + u|μ|] |
      | variance [V] | [(7 D + 1) u V + (4 D² - 2 D) u R √V] | }

    A [Logsumexp] state is a maximum [m] and a sum [s] of [exp (x - c)], [c]
    being [m] where finite and [0] elsewhere; a combination rescales each
    sum by an [exp], within [(2E + 1) u], or by [exp 0 = 1] exactly,
    multiplies, within [u], and adds, within [u]. The rescales' arguments
    round within [u (M - x)] along a term's path, [M] the terms' maximum,
    which moves [s] by a relative [u ln N] at most; [c + log s] adds the
    [log]'s [(2L + 1) u ln N] and its own rounding. A [Moments] state is a
    count, the mean of terms shifted by the slice's first and its sum of
    squared deviations, combined by Chan, Golub and LeVeque's (1.5b) with
    the means weighted by the counts: a combination's mean errs by its
    states' largest error and [3 u R] at most, its sum by a relative [7 u],
    and the means' errors reach the variance through each [δ²], [δ] the
    difference of the states' means, whose weighted sum at each depth
    Cauchy and Schwarz's inequality bounds by [N √V / 2]. *)

open Tolk

val reduce :
  Nx_kernel.reduction ->
  int list ->
  src:Nx_array.Dtype.any ->
  Nx_array.Dtype.any ->
  Ops.t ->
  Ops.t list
(** [reduce r axes ~src dt x] is the reduction [r] of the node [x], a value of
    [src], along [axes], in [src]'s carrier ({!Nx_kernel.carrier}) and rounded
    once to [dt], as {!Nx_kernel.S.reduce}'s destinations: one node, or the
    mean then the variance for [Moments], or the extreme then its [int64]
    position for [Arg]. Each drops [axes].

    Raises [Invalid_argument] for a complex [Prod]. *)

val scan :
  Nx_kernel.reduction ->
  int ->
  carry:Ops.t option ->
  src:Nx_array.Dtype.any ->
  Nx_array.Dtype.any ->
  Ops.t ->
  Ops.t list
(** [scan r axis ~carry ~src dt x] is the inclusive running reduction [r] of the
    node [x], a value of [src], along [axis], in [src]'s carrier and rounded
    once to [dt], its nodes as {!reduce}'s, each of [x]'s shape. With [Some c],
    of a [Monoid], each slice starts from its element of [c], a value of the
    carrier of extent [1] along [axis].

    An [Arg] scan finds the running extreme by its [Max] or [Min] scan, and
    each position as the last position up to it where the extreme changes.
    A [Logsumexp] or [Moments] scan combines states by the network of
    {!Nx_kernel.section-scans}, within its {{!bounds}bounds}: about
    [2 log₂ n] kernels and [O(n)] work for a slice of [n] terms.

    Raises [Invalid_argument] for a complex [Prod], and a [complex128] [Max],
    [Min] or [Arg]. *)

val contract :
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  acc:Nx_array.Dtype.any ->
  out:Nx_array.Dtype.any ->
  Nx_array.Dtype.any * Ops.t ->
  Nx_array.Dtype.any * Ops.t ->
  Ops.t option ->
  Ops.t
(** [contract ~batch ~contracting ~acc ~out (da, a) (db, b) init] is the
    contraction of [a], a value of [da], with [b], a value of [db], by the pairs
    [batch] and [contracting], as {!Nx_kernel.S.contract} states: each operand
    stored into [acc], each product in [acc], their sum along the contracted
    axes and [init], a value of [out] where present, added in [acc], then
    rounded once to [out]. With no contracted term the result is [init]
    in [acc], rounded to [out], or [+0]. An integer [acc] wraps exactly. A float [acc]'s
    result is within [γ(K + 1) (|init| + Σ|a·b|)] of the exact one for [K]
    contracted terms, barring overflow and underflow: tolk may fuse a product
    into the sum that takes it. A complex [acc]'s real part is within
    [γ(K + 2) (|init| + Σ(|ar br| + |ai bi|))] of the exact one, its imaginary
    part within [γ(K + 2) (|init| + Σ(|ar bi| + |ai br|))], [|init|] that
    part's: each takes two products and a difference or a sum per term. *)
