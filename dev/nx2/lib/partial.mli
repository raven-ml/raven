(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Folds over a cut axis: operations whose route computes by partials
    ({!Route.Partials}).

    A {e fold} is a reduction, a scan without carry or a contraction, and
    its {e folded} axes are those it reduces, scans or contracts, and [b]'s
    axis of [Rows]' groups ({!Nx_kernel.type-segments}). Its route
    reads an operand cut along a folded axis, so no device holds every term
    of its window of the results. Each device computes a {e partial}, the
    fold of the terms its operands' windows hold, and the partials
    {e close}: they move between devices and combine into the results at
    the route's result. Every step is one operation computed by windows
    ({!Exec.windows}) or one move ({!Place.value}). A partial computes on
    its device alone, over a route this module builds that reads every
    operand on that device and puts every result there. This module lies
    above {!Exec}; {!Check} and {!Eval} route operations and give it those
    whose route computes by partials.

    A device's windows of the operands along the folded axes are its
    {e tile}. The devices that hold one window of the results form a
    {e group}. The grid axes that cut the folded axes number a group's
    tiles in C order of their first indices, one per device of a group of
    [N] devices. A tile is empty only where a folded axis has fewer
    elements than tiles, and the tiles that {e hold terms} come first along
    each cut axis. A group's partials are {e stacked} into one value of a
    new leading axis per cut folded axis, in the order of the first
    operand's axes, or one for [b]'s groups, each of as many elements as
    its folded axis has tiles holding terms. A device whose tile holds
    terms holds its partial, its {e slot}, at its tile's number along each
    cut folded axis: the stack's placement cuts each leading axis over the
    grid axes that cut its folded axis and every other axis as the result
    is cut, so a device whose tile is empty holds no slot.

    A partial is in the {e carrier} of the result's terms
    ({!Nx_kernel.carrier}), a contraction's accumulator, or an [Arg]'s
    term's dtype, and the closing rounds once into the result's dtype. So
    a result depends on the terms, the placement and the devices' kernels
    alone, and is the same on every run. Barring overflow and underflow,
    the bounds below are on a result before that rounding, which adds its
    own, where each device's kernels keep the bounds of
    {!Nx_kernel.section-bounds}. Let [n] be an output's terms, [t] the
    tiles holding terms and [u] the carrier's unit roundoff. A NaN result
    is a NaN whose sign and payload this module does not state. *)

(** {1:reductions Reductions and contractions}

    A reduction or a contraction closes so. The stack moves to a placement
    whole along its leading axes, the grid axes that cut them moved to the
    result's first axis that the result keeps whole and that holds at least
    [N] elements, or holding copies where it has none. One [Reduce] over
    the leading axes then gives each device its {e part}, its window of the
    results there: the fold's monoid over the slots in C order of their
    indices, by the kernels in their own order. Where an axis took the
    leading axes' cut, the parts move to the route's result
    ({!Place.value}). A [Reduce] of several reductions computes each one
    alone.

    A [Sum], [Prod], [Max] or [Min] partial reduces its tile into the
    carrier, and the closing reduces the slots by the same monoid. Over
    integers and booleans this is exact: the result has the bits of the
    reduction on one device. A float or complex [Max] or [Min] has them too
    where it is not a NaN, since {!Nx_kernel.Prog}'s [Maximum] and
    [Minimum] round nothing and put [-0] below [+0]. Any other float or
    complex result has the kernels' association over its tile and over the
    slots, and does not keep the bits of the reduction on one device. A
    float sum is within [γ(n - 1) Σ|x|] of the exact sum, as on one device,
    and a complex sum so part by part: a tile of [m] terms sums within
    [γ(m - 1)] of its magnitudes, the slots within [γ(t - 1)] of theirs, and
    [γ(j) + γ(k) + γ(j) γ(k) <= γ(j + k)] (Higham, {e Accuracy and
    Stability of Numerical Algorithms}, Lemma 3.3) with
    [(m - 1) + (t - 1) <= n - 1]. A float product is so within a relative
    [γ(n - 1)], and a complex product within a normwise relative
    [γ(2n - 2)], each complex multiplication by the kernels erring by at
    most [2u] ({!Nx_kernel.section-bounds}).

    A [Logsumexp] closes as the [Logsumexp] of its tiles' logsumexps,
    within [ε + ε{_t}] of the exact one, [ε] the largest of the bounds the
    kernels state for its tiles' logsumexps and [ε{_t}] theirs for one of
    [t] terms: a logsumexp moves by no more than its terms do.

    An [Arg] partial is its tile's first extreme in its term's dtype and its
    position, renumbered from C order of the tile's reduced indices into C
    order of the whole's. The extremes and the positions are stacked and
    moved as above, and each device combines the slots, in order, one map
    per slot: of two, the first extreme, a NaN before any number and the
    smaller position of two NaNs, else the extreme in {!Nx_kernel.Prog}'s
    order, [-0] below [+0] and complex numbers by their parts, and the
    smaller position of two of one bits. So the result is the whole's first
    extreme and its position, whatever the tiles' order.

    [Moments] close in two passes, each of whose sums closes as a [Sum]:
    the tiles' sums give [s] and the mean [c = s / n], which every device
    of the group then holds; each tile then sums its terms' deviations
    [d = x - c] and their squares, and the variance is
    [(Σ d² - Σ d (Σ d / n)) / n], [+0] where that is negative or [Σ d²]
    where that is [+∞]. Both are NaNs where [c] is. On nx.cpu's kernels,
    to first order in [u] and for [n] exact in the carrier, a lane-order
    sum of [m] terms rounds each at most [L + D] times, [L] and [D] as
    {!Nx_cpu.section-states} defines them for [m] terms, and the closing,
    a lane-order sum of [t] terms, adds [L{_t} + D{_t}], those of [t]
    terms. Let
    [K = L + D + L{_t} + D{_t}] for the tile of most terms, [M] the mean of
    the terms' magnitudes and [V] their variance. The mean is within
    [A = K u M + u|μ|] of the exact [μ] and the variance within
    [(K + 5) u V + 2 (K + 1) u A √V + (3K + 7) u A²], nx.cpu's bound on one
    device with [K] for its [L + D]. For any [c],
    [Σ (x - c)² - (Σ (x - c))² / n] is [n V], so [c]'s error reaches the
    variance only through the roundings: [Σ d²] within a relative
    [(K + 3) u], and [Σ d] within [(K + 1) u Σ|d|], where
    [Σ|d| <= n (√V + A)] by Cauchy-Schwarz.

    A contraction's tile is its windows of [a] and [b] along the contracting
    pairs, alike, or, for [Rows] whose [b] is cut along its groups, its
    window of the groups, the pairs whole. A scale is read whole along the
    pairs and the groups. Along each, a tile is cut at its scales' block
    edges into {e pieces}, each within one block of every scale there or
    from an edge they all share to another: a tile of no scale is one
    piece, and a library that sums blocks of its own sums them within each
    piece. A device contracts its tile's pieces in C order of their first
    indices by its kernels, the first from no [init] and each other from
    the sum of those before it as its [init], into the accumulator, each
    scale read at the blocks the piece spans. The closing sums the slots;
    [init], as the accumulator holds it, is added once to each part, and
    the result rounds once into its dtype. A segmented contraction cut
    along a pair never computes so, nor one of [Contracting] groups.

    A piece of groups [lo] to [hi - 1] of [Rows]' offsets [o], [g + 1] of
    them, is bounded by fences [lo] to [hi] of [F e], the least of
    [max o.(0) o.(i)] for [i] from [e] to [g], which its device computes
    from the offsets it holds whole. Group [e] of
    {!Nx_kernel.type-segments} is the rows [r] with
    [F e <= r < F (e + 1)], so each group of the piece is the whole's,
    whatever the offsets, and offsets that do not decrease are their own
    [F]. A device multiplies the rows of its groups. Every other row has
    no products there and holds [+0], or the sum before it in a later
    piece. An output's slots so hold its sum on its group's device and
    [+0] on the others, and a sum from [+0] is never [-0]. Without
    [init] the closing adds exact zeros, and each output has the bits it
    has with [b]'s groups whole on each device, NaNs and infinities
    included, where the kernels' association does not depend on the count
    of [b]'s groups, as nx.cpu's ({!Nx_cpu}). With [init], an output whose
    slots sum to zero takes [init] alone, so that one without products
    keeps [init]'s bits.

    An integer contraction has the bits it has on one device. A float or
    complex one is within
    [γ(2K + 2) (|init| + Σ|a||b|)] of the exact sum, a complex one in
    modulus, [K] an output's products and [u] the accumulator's unit
    roundoff, that of its parts' dtype for a complex one,
    plus the roundings each library states its scaling adds: the bound on
    one device ({!Nx.contract}). A kernel's addition errs by at most [2u],
    a reduction's and nx's [Add] by [u]. A product passes at most the
    products of its tile in additions there, at most [t - 1] in the
    closing and one for [init], and the products of a tile and [t - 1] are
    at most [K], as each tile holding terms holds a product. A term passing
    [k] additions is within [γ(2k)] of itself (Higham, Lemma 3.1).

    {1:scans Scans}

    A [Scan] closes per group of its results' devices that hold one window
    of the axes off its axis; a device's window along its axis is its tile,
    and the group's tiles are in their order along it. A device scans its
    tile alone. Its tile's {e total} is the last result of that scan. The
    totals, stacked along a new leading axis cut as the axis is, of one
    slot per tile holding terms, move to every device of the group. Each
    device but the first then combines the totals of the tiles before its
    own with each result of its tile. The results keep the axis's cut.

    - A [Sum], [Prod], [Max] or [Min] tile scans into the carrier. One
      [Scan] of the monoid along the slots gives each device, at slot [i],
      the totals of tiles [0] to [i] combined, and a tile's results combine
      with the slot of the tile before it by [Add], [Mul], [Maximum] or
      [Minimum], the slot's first, rounding once into the result's dtype.
      Over integers and booleans, and for a float or complex [Max] or
      [Min] that is not a NaN, each result has the bits of the scan on one
      device. A float or complex sum's or product's result of [n] terms is
      within the bound of a reduction of them above: each of its terms
      passes at most the terms of its tile in additions there, at most the
      tiles before in the totals' scan, and one in the combination, and
      every tile before holds a term.
    - An [Arg] tile scans with its positions from the axis's start. A
      device takes the first extreme of the slots before its own, in tile
      order, one map per slot as a reduction's slots combine, and combines
      it with each of its results so, with the bits of the scan on one
      device.
    - A device whose tile is empty holds empty results.

    {1:traffic Traffic}

    A device of a group of [N], [t] of whose tiles hold terms, receives,
    where an axis took the leading axes' cut, its part of each slot but its
    own, [1 / N] of its window of the results each, then the [N - 1] other
    parts of its window: [2 (N - 1) / N] of its window where [t = N]. Where
    none did, it receives each slot but its own: [t - 1] windows of the
    results, [t] where its tile is empty. An [Arg] moves an extreme and a
    position per output for each, and [Moments] three such closings. For a
    [Scan], a device receives the totals of the other tiles of its group
    holding terms, one row of its window off the axis each, two for an
    [Arg]. No part of an operand moves.

    A [Reduce] of [k] reductions runs [k] loops over each device's loads,
    [Moments] two over its terms, laid out once where they are not a load.
    A [Scan] runs one loop over each device's loads, then one map over its
    results on each device but the first, and on the first where the
    carrier is not the result's dtype. A tile cut into pieces reads each of
    its scales once per piece, and a piece of [Rows]' groups runs one
    [Scan] over its device's offsets. *)

val run : by:string -> unit Route.t -> 'r Value.prim -> 'r
(** [run ~by route op] is [op], its operands lying where [route], its route,
    reads them, computed as stated above.

    Raises as {!Exec} states for the kernels, naming [by], and
    [Invalid_argument] as an internal fault for an operation other than a
    [Reduce], a [Scan] of no carry or a [Contract] of no segments or of
    [Rows], since no route computes another by partials, and where the
    result's grid and the folded operand's are not of one mesh
    ({!Grid.cut_like}). *)
