(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Operations whose route computes by partials ({!Route.Partials}), as
    operations applied through [apply]: on one device each, or a [Place].

    A [Reduce] of several reductions computes each one alone. One computes
    per group of its results' devices that hold one window of its results. A
    device's window along the reduced axes is its {e tile}; a device whose
    tile is empty reduces nothing. A group's tiles are in C order of their
    first indices. The first device of a group holding a
    tile reduces its terms into a partial result per output, by its kernels
    in their own order. Each device of the group takes the partials of each
    tile it does not hold from that device and combines them, one per tile
    in the tiles' order, rounding once into the result's dtype. So a result
    depends on the terms, the placement and the devices' kernels alone, and
    is the same on every run:

    - A [Sum], [Prod], [Max] or [Min] reduces into the carrier of its
      output's dtype ({!Nx_kernel.carrier}); its partials combine by [Add],
      [Mul], [Maximum] or [Minimum] in the carrier. Over integers and
      booleans this is exact: the result has the bits of the reduction on
      one device. A float or complex [Max] or [Min] has them too where it is
      not a NaN: {!Nx_kernel.Prog}'s [Maximum] and [Minimum] round nothing,
      put [-0] below [+0] and give a NaN where either operand is one.
    - An [Arg] reduces into its output's dtype, each position renumbered
      from C order of the window's reduced indices into C order of the
      whole's, and its partials move and combine as an integer's. Two
      partials combine into the first extreme: a NaN before any number and
      the smaller position of two NaNs, else the extreme in
      {!Nx_kernel.Prog}'s order, [-0] below [+0] and complex numbers by
      their parts, and the smaller position of two of one bits. So the
      result is the whole's first extreme and its position, whatever the
      tiles' order.
    - A [Logsumexp] is the logsumexp of its tiles' logsumexps, each into the
      carrier, laid out as [[outputs; tiles]] on each device.
    - [Moments] compute in the carrier in two passes. Each tile's sum
      combines by [Add] in the tiles' order into [s], and the mean [c] is
      [s / n]. Each tile then sums its terms'
      deviations [d = x - c] and their squares, which combine so, and the
      variance is [(Σ d² - Σ d (Σ d / n)) / n], [+0] where that is
      negative or [Σ d²] where that is [+∞]. Both are NaNs where [c] is.

    Any other float or complex result does not keep the bits of the
    reduction on one device, and a NaN result is a NaN whose sign and
    payload it does not state. Let [n] be an output's terms, [t] the tiles holding terms and [u]
    the carrier's unit roundoff. Where each device's kernels keep the bounds
    of {!Nx_kernel.section-reductions}, and barring overflow and underflow,
    a result before the rounding into its dtype, which adds its own, is:

    - For a float sum, within [γ(n - 1) Σ|x|] of the exact sum, as on one
      device, and a complex sum so part by part: a tile of [m] terms sums
      within [γ(m - 1)] of its magnitudes, the partials within [γ(t - 1)] of
      theirs, and [γ(j) + γ(k) + γ(j) γ(k) <= γ(j + k)] (Higham, {e Accuracy
      and Stability of Numerical Algorithms}, Lemma 3.3) with
      [(m - 1) + (t - 1) <= n - 1]. A float product, so, within a relative
      [γ(n - 1)]. A complex product within a normwise relative
      [γ(2n - 2)], each complex multiplication by the kernels erring by at
      most [2u] ({!Nx_kernel.section-reductions}).
    - For a [Logsumexp], within [ε + ε{_t}] of the exact one, [ε] the
      largest of the bounds the kernels state for its tiles' logsumexps and
      [ε{_t}] theirs for one of [t] terms: a logsumexp moves by no more than
      its terms do.
    - For [Moments] on nx.cpu's kernels, to first order in [u] and for [n]
      exact in the carrier: a lane-order sum ({!Nx_cpu}) of [m] terms
      rounds each at most [L + D] times, [L = ⌈min (m, 1024) / 16⌉ + 3] in
      its block and [D = ⌈log₂ ⌈m / 1024⌉⌉] across blocks, and the
      combination adds [t - 1]. Let [K = L + D + t - 1] for the tile of
      most terms, [M] the mean of the terms' magnitudes and [V] their
      variance. The mean is within [A = K u M + u|μ|] of the exact [μ] and
      the variance within [(K + 5) u V + 2 (K + 1) u A √V + (3K + 7) u A²],
      nx.cpu's bound on one device with [K] for its [L + D]. For any [c],
      [Σ (x - c)² - (Σ (x - c))² / n] is [n V], so [c]'s error reaches the
      variance only through the roundings: [Σ d²] within a relative
      [(K + 3) u], and [Σ d] within [(K + 1) u Σ|d|], where
      [Σ|d| <= n (√V + A)] by Cauchy-Schwarz.

    Each device moves one partial per output for each tile of its group it
    does not hold, two for an [Arg]: of [t] tiles, [t - 1] rows of its
    window of the results, where reading the loads whole moves
    [(t - 1) / t] of each load's part on its group. A [Logsumexp] then
    assembles them and reduces [t] terms per output. [Moments] move one
    row per tile, then two, and map after each. A [Reduce] of [k]
    reductions runs [k] loops over each device's loads, [Moments] two over
    its terms, laid out once where they are not a load.

    A [Contract] computes per group of its result's devices that hold one
    window of it. A device's windows of [a] and [b] along the contracting
    pairs, alike, are its {e tile}, and a group's tiles are in C order of
    their first indices. A scale is read whole along the pairs. Along each
    pair, a tile is cut at its scales' block edges into {e parts}, each
    within one block of every scale there or from an edge they all share to
    another: a tile of no scale is one part. The first device of a group
    holding a tile contracts each of its parts, in C order of their first
    indices, by its kernels in their own order, from no [init] into the
    accumulator, each scale read at the blocks the part spans. Each device
    of the group takes the parts' sums, adds them in order, tile after tile,
    then [init] as the accumulator holds it over its window of the result,
    and rounds once into the result's dtype. A segmented contraction never
    computes so.

    An integer contraction so has the bits it has on one device. A float or
    complex one does not, and a NaN result is a NaN whose sign and payload
    it does not state. Let [K] be an output's products and [u] the
    accumulator's unit roundoff, its parts' for a complex one. Where each
    device's kernels keep the bound of {!Nx_kernel.S.contract}, and barring
    overflow and underflow, a result before the rounding into its dtype is
    within [γ(2K + 2) (|init| + Σ|a||b|)] of the exact sum, a complex one
    in modulus, plus the roundings each library states its scaling adds: the
    bound on one device ({!Nx.contract}). A kernel's addition errs by at most
    [2u] and nx's [Add] by [u]. A part of [m] products sums them and [+0],
    so each product passes at most [m] additions there; the sums of [p]
    parts pass at most [p - 1] more and [init] one, and [m + p <= K + 1] as
    each part holds a product. A term passing [k] additions is within
    [γ(2k)] of itself (Higham, {e Accuracy and Stability of Numerical
    Algorithms}, Lemma 3.1).

    Each device takes the sum of each part it does not contract, over its
    window of the result: [p - 1] such windows in the accumulator where it
    contracts one part. No part of [a] or [b] moves. A tile cut into parts
    reads each of its scales once per part.

    A [Scan] computes per group of its results' devices that hold one window
    of the axes off its axis; a device's window along its axis is its
    {e tile}, and the group's tiles are in their order along it. A device
    scans its tile from a carry, the partials of the tiles before it
    combined in order on that device:

    - A [Sum], [Prod], [Max] or [Min]: a tile's partial is its terms reduced
      into their carrier. Partials combine by [Add], [Mul], [Maximum] or
      [Minimum], and the tile scans from the carry by {!Nx_kernel.S.scan}.
      Over integers and booleans, and for a float or complex [Max] or [Min]
      that is not a NaN, each result has the bits of the scan on one device.
      A float or complex sum's or product's result of
      [n] terms is within the bound of a reduction of them above: the carry
      is the reduction of the terms before the tile, and the tile's scan
      takes it as one more term.
    - An [Arg]: a tile's partial is its terms' first extreme and its
      position from the axis's start. The tile scans alone, and one map
      combines each result with the carry into the first extreme, as a
      reduction's partials combine, with the bits of the scan on one device.
    - A device whose tile is empty scans its window alone.

    Each device takes, as rows of its window of the results off the axis,
    the partials of the tiles before its: one row per tile, two for an
    [Arg]. No device waits for another's results. An [Arg] maps twice over
    each tile's results but the first's. *)

val run :
  ('q. by:string -> 'q Value.prim -> 'q) ->
  by:string ->
  unit Route.t ->
  'r Value.prim ->
  'r
(** [run apply ~by route op] is [op], its operands lying where [route], its
    route, reads them, computed through [apply] as stated above.

    Raises [Invalid_argument] as an internal fault for an operation other than
    a [Reduce], a [Scan] of no carry or a [Contract]: no route computes
    another by partials. *)
