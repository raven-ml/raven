(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Folds over a cut axis: operations whose route computes by partials
    ({!Route.Partials}).

    A {e fold} is a reduction, a scan without carry, a contraction, a
    gather or a scatter's [Add], and its {e folded} axes are those it
    reduces, scans or contracts, [b]'s axis of [Rows]' groups
    ({!Nx_kernel.type-segments}), the axis a gather reads at positions and
    the axis of an [Add]'s positions and updates. Its route
    reads an operand cut along a folded axis, so no device holds every term
    of its window of the results. Each device computes a {e partial}, the
    fold of the terms its operands' windows hold, and the partials
    {e close}: they move between devices and combine into the results at
    the route's result, but a contraction's asked for its sum
    ({!section-sums}). Every step is an operation on one device, one
    computed by windows ({!Partition.windows}) or one move
    ({!Partition.place}), each on the engine the caller gives, so a fold is
    a partition. A partial, a part or a window of the results that devices
    of several architectures hold is computed by those of one and copied to
    the others ({!Partition.section-replicas}). A partial computes on its device alone, over a route this
    module builds that reads every operand on that device and puts every
    result there. This module lies above {!Partition}; {!Check} and {!Eval}
    route operations and give it those whose route computes by partials. A
    scatter into a target cut along its axis computes by partials too, and
    closes nothing ({!section-scatters}).

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
    ({!Nx_kernel.carrier}), a contraction's accumulator, an [Arg]'s term's
    dtype or a gather's bits ({!section-gathers}), and the closing rounds
    once into the result's dtype, a gather's exactly. So
    a result depends on the terms, the placement and the devices' kernels
    alone, and is the same on every run. Barring overflow and underflow,
    the bounds below are on a result before that rounding, which adds its
    own, where each device's kernels keep the bounds of
    {!Nx_kernel.section-bounds}. Let [n] be an output's terms, [t] the
    tiles holding terms and [u] the carrier's unit roundoff. A NaN result
    but a gather's is a NaN whose sign and payload this module does not
    state. *)

(** {1:reductions Reductions and contractions}

    A reduction or a contraction closes so. Each device of a group computes
    its {e part} of the group's window of the results: its window along the
    result's first axis that the result keeps whole and that holds at least
    [N] elements, that axis cut over the group as the grid axes cut the
    leading axes, or the whole window where the result has no such axis. It
    computes its part in {e pieces}, each of whose slots hold at most
    [C = 256] KiB, a buffer of fixed size whatever the result's
    (Rajbhandari et al., {e ZeRO}, SC 2020, §6.2): boxes in C order that
    tile the part, for the first axis on which the slots of one index fit
    in [C], single indices of every axis before it and runs along it of as
    many indices as fit, every axis after it whole. A part is one piece
    where its slots fit in [C], or where its dtype is narrower than a byte
    and the host does not address the device's memory, since a piece may
    start inside a byte. For each
    piece in turn, the device takes the slots' elements there one slot at a
    time, in C order of their indices ({!Partition.take}), and combines each
    with what the slots before it gave, by one map of the fold's monoid: the
    piece's results are the left fold of the slots in that order, finished
    by the maps below. A device so holds at most two slots of a piece and
    their combination at a time, and every device folds a window's slots in
    one order, so each result's bits depend on the slots alone, whatever the
    pieces and whichever device computes it. A [Logsumexp], which has no
    monoid, takes every slot's elements of a piece at once, and one [Reduce]
    over the leading axes gives the piece's results, by the kernels'
    association of each output's terms: where that association depends on
    their count alone, as nx.cpu's does ({!Nx_cpu.section-lane_order}), each
    result has the bits of one [Reduce] of every slot of the group's window,
    and a device holds every slot of a piece at once. Each device of the
    route's
    result then gathers its window from the pieces, its own where it holds
    them. A [Reduce] of several reductions computes each one alone.

    A [Sum], [Prod], [Max] or [Min] partial reduces its tile into the
    carrier, and the closing folds the slots by the same monoid. Over
    integers and booleans this is exact: the result has the bits of the
    reduction on one device. A float or complex [Max] or [Min] has them too
    where it is not a NaN, since {!Nx_kernel.Prog}'s [Maximum] and
    [Minimum] round nothing and put [-0] below [+0]. Any other float or
    complex result has the kernels' association over its tile and the left
    fold over the slots, and does not keep the bits of the reduction on one
    device. A
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
    taken piece by piece as above, and each device combines a piece's
    slots, in order, one map per slot: of two, the first extreme, a NaN
    before any number and the smaller position of two NaNs, else the
    extreme in {!Nx_kernel.Prog}'s order, [-0] below [+0] and complex
    numbers by their parts, and the smaller position of two of one bits. So
    the result is the whole's first extreme and its position, whatever the
    tiles' order.

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

    A piece of groups [lo] to [hi - 1] of [Rows]' groups takes the ids of
    its rows less [lo], so that group [e] of the piece is group [lo + e] of
    the whole and an id outside the piece's groups falls in none of them
    ({!Nx_kernel.type-segments}). A device multiplies the rows of its
    groups. Every other row has no products there and holds [+0], or the
    sum before it in a later piece. An output's slots so hold its sum on
    its group's device and [+0] on the others, and a sum from [+0] is never
    [-0]. Without [init] the closing adds exact zeros, and each output has
    the bits it has with [b]'s groups whole on each device, NaNs and
    infinities included, where the kernels' association does not depend on
    the count of [b]'s groups, as nx.cpu's ({!Nx_cpu}). With [init], an
    output whose slots sum to zero takes [init] alone, so that one without
    products keeps [init]'s bits.

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

    A contraction asked for its sum ([sum], {!Value.Contract}) does not
    close: each device's partial, from [init] on the device whose term is
    its group's first ({!Grid.term}) and from {!Partition.identity} on the
    others where there is an [init], so that [init] joins the sum once, is
    its term, rounded once into the result's dtype ({!section-sums}). Its
    terms are numbered as the tiles of its folded axes, in C order of their
    first indices ({!Grid.sum}). Rounded into a dtype of unit roundoff [v],
    each term errs by [v] more, and a closing of [t] terms adds them there,
    [t - 1] additions: [γ(t, v) Σ|term|] more (Higham, §4.2 and Lemma 3.3).

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

    {1:gathers Gathers}

    A [Gather] closes as a [Reduce] does, its tile its window of [x] along
    the gathered axis. A device gathers from its tile at its window of the
    positions, each moved by the tile's start, so that one outside the tile
    reads the element of zero bits, and reads the result's bits as the
    unsigned integer of the dtype's width, a [complex128]'s as two
    [uint64]s along a new last axis, a [bit]'s as themselves. The closing is
    the slots' [Max]. A position within the axis lies in one tile, whose
    slot holds its element's bits there, every other slot zero bits; one
    outside the axis reads zero bits from every tile. The parts, read back
    as the dtype, so have the bits of the gather on one device, a NaN's
    included.

    {1:scatters Scatters}

    A [Scatter] into an [into] cut along its axis computes on each device its
    window of the results, which lie where [into] does. A device writes into
    its window of [into] the updates of its windows of [idx] and [updates],
    whole along the axis, at the positions moved by its window's start, so
    that one outside its window writes nothing. Each element so takes the
    updates whose positions name it, in their order on one device. A [Set]
    so has the bits of the scatter on one device, and so does an [Add] where
    the kernels add an element's updates in their order, as nx.cpu's do
    ({!Nx_cpu}); a [Max] or [Min] has them but for a NaN's sign and
    payload.

    An [Add] whose positions and updates are cut along its axis is a fold
    of the updates, its tile its windows of them along the axis. A device
    adds the updates of its tile, at its window of the positions moved by
    its window's start, into its window of the results in the carrier:
    into [into]'s window for the first tile, cast to the carrier, and into
    [-0] for every other, which adds nothing. The stack closes as a
    reduction's does, each piece the slots added in tile order by one map,
    with no [+0] added, then rounded once into the dtype. Each element so
    is [into]'s plus its updates. Over integers this is exact, with the
    bits of the scatter on one device. For finite operands, barring
    overflow, a float or complex element of [k] updates over [t] tiles
    holding updates is within [γ(k + t) (|x| + Σ|u|)] of the exact sum
    before a narrow float's rounding, a complex one part by part: its
    tile's scatter is within the kernels' [γ(k{_j} + 1)] for the [k{_j}]
    updates there ({!Nx.scatter}), and the slots add [t - 1] roundings.
    Where no update lands, an element keeps [into]'s value, a [-0]'s sign
    included; a NaN stays a NaN whose sign and payload this module does
    not state.

    {1:sums Sums}

    A value at a partial placement ({!Grid.sums}) is a {e sum}: each
    device holds a term, its window of a value of the whole's shape, and
    the value is the sum of its group's terms. Its terms are values of its
    dtype, a float, a complex or an integer that is its own carrier, and add
    as its own additions do ({!Prim.summable}). {!place} closes a sum
    at a placement that is none as a reduction closes: its terms are the
    slots, stacked in term order along one leading axis cut over the summed
    grid axes, and each device of a group computes its part piece by piece,
    the left fold of the slots in term order by one map of [Add] of the
    dtype, from the first term and from no [+0]. Each element so has the same
    bits whichever device adds it, but a NaN's sign and payload. Where the
    target cuts one axis over the summed grid axes alone, a
    {e reduce-scatter}, it is the parts: each device computes its own
    window. Otherwise the parts cut, over the summed grid axes, the first
    axis that the placement closed ({!Grid.closed}) keeps whole and that
    holds an element per term of a group, or are that placement where no
    axis does; each device of it gathers its window from the parts, an
    {e all-reduce}, and the result moves to the target ({!Partition.place}).

    A value that is no sum placed at a sum holds, on each device whose term
    is the first of its group, its window placed at the sum's placement
    closed, and on every other device a map of no loads of the additive
    identity, [-0] for a float and a complex number's parts and [0] for an
    integer, so that closing it gives its bits, NaNs as a class. A sum placed
    at another sum closes first.

    {1:memory Memory}

    A device's {e partials} are its slots: one in the carrier, for an [Arg]
    its extreme and its position, for [Moments] the mean, whole, and the
    two sums of deviations. Where a group holds [s] slots, a device holds,
    beside the operands, the larger of its partials and the results, its
    part of the results and, at a time, at most [C] bytes of one piece's
    slots and [m] of the piece's results, of at most [C / s] bytes each:
    [m = 1] for a [Sum], [Prod], [Max], [Min] or [Logsumexp], a gather, a
    contraction without [init] and a sum's closing, whose slots are the
    sum's terms, [m = 2] for an [Arg], [Moments], a contraction with [init]
    and a scatter's [Add], which finish a piece by a second map. A
    contraction asked for its sum holds its term, beside its operands, and
    one piece's sum at a time. A piece holds more than [C] only where the
    slots of one element do, or where a part of a dtype narrower than a byte on a device
    whose memory the host does not address is one piece. Its partials go
    before the result's windows are gathered. So a float32 sum whose result
    of [r] bytes is whole on [N] devices, each holding a slot of [r] bytes,
    holds at most [r + r / N + C (1 + 1 / N)] beside the operands.

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
    [Arg]. A [Gather] moves its slots as a [Reduce] of one reduction does,
    and no part of its positions. A [Scatter] into a cut target moves
    nothing; an [Add] of cut updates moves its slots as a [Reduce] of one
    reduction does, and no part of its positions or updates. No part of an
    operand moves. A contraction asked for its sum moves nothing. A sum's
    closing moves its terms as a [Reduce] of one reduction moves its slots,
    every device's term holding terms; a reduce-scatter gathers no part, so
    each device receives [(N - 1) / N] of its window of the sum.

    A [Reduce] of [k] reductions runs [k] loops over each device's loads,
    [Moments] two over its terms, laid out once where they are not a load.
    A [Scan] runs one loop over each device's loads, then one map over its
    results on each device but the first, and on the first where the
    carrier is not the result's dtype. A tile cut into pieces reads each of
    its scales once per piece, and a piece of [Rows]' groups runs one
    [Scan] over its device's offsets. A [Scatter] runs on each device one
    map shifting its positions where its window starts past [0] and one
    scatter, and an [Add] of cut updates also one map for its target and,
    where the carrier is not the dtype, one casting its updates, then one
    map adding the slots and, where the carrier is not the dtype, one
    casting the result. A [Gather] runs on each
    device one map shifting its positions where its tile starts past [0],
    one gather and one bitcast, and one bitcast over the results. *)

val place :
  Partition.engine ->
  by:string ->
  'e Devices.placement ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'e) Value.t
(** [place e ~by p x] is {!Partition.place}[ e ~by p x] where neither [x]'s
    placement nor [p] is a sum, as {!section-sums} states otherwise.

    Raises what {!Partition.place} raises, and [Invalid_argument] naming
    [by] for a sum [p] of a dtype no sum has. *)

val run : Partition.engine -> by:string -> unit Route.t -> 'r Value.prim -> 'r
(** [run e ~by route op] is [op], its operands lying where [route], its
    route, reads them, computed as stated above on [e].

    Raises what [e]'s steps raise, naming [by], and
    [Invalid_argument] as an internal fault for an operation other than a
    [Reduce], a [Scan] of no carry, a [Gather], a [Scatter] or a [Contract]
    of no segments or of [Rows], since no route computes another by partials,
    and where the result's grid and the folded operand's are not of one
    mesh ({!Grid.cut_like}). *)
