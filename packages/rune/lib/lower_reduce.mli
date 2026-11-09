(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Reductions, scans and contractions as tolk nodes.

    Each function takes nodes as {!Lower_arith} holds elements and computes what
    {!Nx_kernel.type-reduction}, {!Nx_kernel.S.scan} and {!Nx_kernel.S.contract}
    state, within the bounds below. These bounds implement the reduction rows of
    {!Rune.section-numerics}.

    A float result computes in its {e accumulator}, which is the carrier
    ({!Nx_kernel.carrier}) of a reduction or a scan and [acc] of a contraction.
    [u] is the accumulator's unit roundoff, its parts' for a complex one, and
    [γ(k) = k u / (1 - k u)], as {!Nx_kernel.section-bounds} defines it. Each
    bound holds barring overflow and underflow, before the rounding into the
    destination, which adds its own.

    {1:monoids Monoids and Arg}

    - {b Extremes} give nx's bits except at a NaN. They order [-0] below [+0],
      and complex numbers by real part, then imaginary part. On booleans they
      are any and all.
    - {b Integer sums and products} wrap and are exact.
    - {b Float sums and products} add or multiply their terms in the accumulator
      in an association tolk chooses. A sum of [n] terms, those of a reduction
      or those up to a scan's result with its carry, is within [γ(n - 1) Σ|x|]
      of the exact one, and a product within a relative [γ(n - 1)]. A sum whose
      terms are all [-0] is [+0], except a scan's result with a carry, which is
      [-0] where the carry and every term up to it are [-0]. The sum of no term
      is [+0] and the product [1]. A complex sum adds each part as a float sum.
    - {b Arg} gives the extreme's position exactly, that of the first NaN term
      where the extreme is a NaN.
    - {b NaNs}: a sum, a product or an extreme with a NaN term is a NaN with an
      unspecified sign and payload, and a complex extreme then has two NaN
      parts.

    {1:bounds Logsumexp and Moments}

    Their special values are {!Nx_kernel.type-reduction}'s: a [Logsumexp] is
    [-∞] for no term or terms all [-∞], [+∞] where a term is [+∞] and none
    is a NaN, and a NaN where a term is one; [Moments] are NaNs for no term
    or a NaN term, and with an infinite term and no NaN the mean is the sum
    of the infinities and the variance a NaN. A variance is never negative.

    Their bounds are to first order in [u]. [E] and [L] are the bounds in ulps
    {!Lower_real} states for [Exp] and [Log], which hold at tolk's default
    {!Tolk.Setting.transcendental}, and an error of [n] ulps counts as a
    relative error of [(2n + 1) u] from the exact result. [M] is the mean of the
    terms' magnitudes. A reduction of [n] terms:

    {t
      | result | bound |
      |:--|:--|
      | [Logsumexp] [r] | [(1.37 (n - 1) + 2E + 1 + (2L + 1) ln n) u + u|r|] |
      | mean [μ] | [B = (n + 1) u M] |
      | variance [V] | [(n + 4) u V + 2 n u B √V + 3 (n + 2) u B²] | }

    A [Logsumexp] or [Moments] scan combines its terms by the odd/even network
    ({!Nx_kernel.section-network}). The scan's output at index [i], of
    [N = i + 1] terms, passes through at most [D = max 0 (2⌈log₂ N⌉ - 1)] of its
    combinations. [R] is the range of the terms. Where [N] is exact in the
    carrier:

    {t
      | result | bound |
      |:--|:--|
      | [Logsumexp] [r] | [((2E + 3) D + (2L + 2) ln N) u + u|r|] |
      | mean [μ]        | [(4 D + 1) u R + u|μ|]                  |
      | variance [V]    | [(7 D + 1) u V + (4 D² - 2 D) u R √V]   |
    } *)

open Tolk

(** {1:reductions Reductions and scans} *)

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

    Raises {!Lower_arith.Unlowered} for a complex [Prod]. *)

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
    each slice starts from its element of [c], a value of the carrier of extent
    [1] along [axis]. Only a [Monoid] uses a carry; another [r] ignores it. A
    slice of [n] terms takes [O(n)] work, in about [2 log₁₆ n] kernels for a
    [Monoid] and [2 log₂ n] for a [Logsumexp] or [Moments].

    Raises {!Lower_arith.Unlowered} for a complex [Prod], and a [complex128]
    [Max], [Min] or [Arg]. *)

(** {1:contractions Contractions}

    A contraction stores each operand into [acc] and, where the operand has a
    block scale ({!type-scale}), multiplies each element by its block's scale
    read into [acc]. It sums the products and [init] in [acc] and rounds the sum
    once to [out]. An integer [acc] wraps and is exact. A float32 contraction
    of integer codes whose [Value] scale runs along its first contracted axes
    with a narrow float of a type that holds every code, where
    {!Tolk.Heuristic.core_fits} holds, sums the products along its last
    contracted axis, multiplies each partial sum by its block's scale, then
    sums the blocks: a tensor core multiplies only its narrow operands.

    A float [acc]'s result of [K] products is within
    [γ(2K + 2 + 2s) (|init| + Σ|a||b|)] of the exact sum, where [s] is the
    number of scaled operands: {!Nx.contract}'s bound where [s] is [0]. A complex [acc]'s real part is within
    [γ(K + 2) (|init| + Σ(|ar br| + |ai bi|))] of the exact one, and its
    imaginary part within [γ(K + 2) (|init| + Σ(|ar bi| + |ai br|))], [|init|]
    that part's. *)

(** The type for a contraction operand's block scale ({!Nx_kernel.type-scale}):
    the node of an array of its rank whose extents divide the operand's. *)
type scale =
  | Exponent of Ops.t
      (** A [uint8] node of codes. Code [e] scales its block by [2{^e - 127}],
          and [255] makes it NaN. *)
  | Value of Nx_array.Dtype.any * Ops.t
      (** A node of that dtype. Each scale is read into [acc] before it
          multiplies its block. *)

(** The type for a contraction operand: its dtype, its node and its scale. *)
type operand = {
  dtype : Nx_array.Dtype.any;
  node : Ops.t;
  scale : scale option;
}

val contract :
  renderer:Tolk.Renderer.t Lazy.t ->
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  segments:Ops.t Nx_kernel.segments option ->
  acc:Nx_array.Dtype.any ->
  out:Nx_array.Dtype.any ->
  operand ->
  operand ->
  Ops.t option ->
  Ops.t
(** [contract ~renderer ~batch ~contracting ~segments ~acc ~out a b init] is
    the contraction of [a] with [b] by the pairs [batch] and [contracting], as
    {!Nx_kernel.S.contract} states, in the form the section above states for
    [renderer]'s tensor cores. [init], where present, must be a value of
    [out]. Each group of [segments], by a node of [int64] ids, is as
    {!Nx_kernel.type-segments} states. An output without products is [init] in
    [acc], rounded to [out], or [+0].

    Of the segmented forms, a [Rows] output's [K] in the bounds above is the
    contracted extent, and a [Contracting] output's [K] counts its group's
    indices alone. The other indices add exact zeros, which can change only a
    zero result's sign.

    A [Rows] contraction of [n] rows over [g] groups contracts each row with
    its own group's slice of [b] alone. Where [n < g], or [g] is [0] or more
    than a slice's elements, each row loads its slice at its id, the
    products of the contraction of one slice, in one kernel. Otherwise it
    routes the rows by counting, in work [n g] and [S ⌈log2 (n + 1)⌉], into
    [S = B c] slots of blocks of [c] rows of one group,
    [B = m + ⌊(n - m) / c⌋] for [m = min n g], [c] 4 below 5 rows a group on
    average, 8 below 16 and 64 from 16: each block loads its group's slice,
    the rows enter at their slots, and each row reads its slot's stored
    output. A row's sum runs over the contracted extent in the kernel's
    order whatever its slot, and the form is chosen from the shapes, so a
    row's bits are a function of the
    shapes, its row and its group's slice; with [b]'s groups cut over
    devices, [g] is each device's, so the order follows the count of
    groups.

    A [Contracting] contraction of [g] groups over a pair of [n] indices
    routes the indices as [Rows] does and cuts each group into blocks of [c]
    indices from its first: at most [B] blocks, [B c] products for each
    index of the other axes into [B] partial results stored apart, and
    [g ⌈n / c⌉] loads of them, each group summing its own blocks from its
    first. [c] is [⌈n / g⌉] or [⌈√(n - g)⌉], with [B = g + ⌊(n - g) / c⌋],
    where that takes fewer products and sums than [g n]. Otherwise [c] is
    [n] and each group takes one block: [g n] products. A group's bits so do
    not depend on the other groups' counts.

    Raises {!Lower_arith.Unlowered} for a scale with an [acc] other than
    [float32] or [float64]. *)
