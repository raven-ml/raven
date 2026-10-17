(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Operations whose route computes by partials ({!Route.Partials}), as core
    operations applied through [apply]: on one device each, or a [Place].

    A [Reduce] computes per group of its results' devices that hold one window
    of its results. A device's window along the reduced axes is its
    {e tile}; a device whose tile is empty reduces nothing. Each device of a
    group reduces terms its tile holds into partial results, takes the
    others' and combines them, rounding once into the result's dtype. The
    results lie where the route puts them, with the bits of the reduction on
    one device:

    - An integer or boolean [Sum], [Prod], [Max] or [Min] reduces into the
      carrier of its output's dtype, [int8] and [uint8] for [int4] and
      [uint4] and [bool] for [bit], one partial per output and device. Each
      device takes the partials of each tile it does not hold from the first
      device of its group that holds it, and combines one per tile in the
      tiles' order by [Add], [Mul], [Maximum] or [Minimum], exact in the
      carrier.
    - An [Arg], and a float or complex [Max] or [Min], reduces as an [Arg]
      into its output's dtype, each position renumbered from C order of the
      window's reduced indices into C order of the whole's, and its partials
      move and combine as an integer's. Two partials
      combine into the first extreme: a NaN before any number and the smaller
      position of two NaNs, else the extreme in {!Nx_kernel.Prog}'s order,
      [-0] below [+0] and complex numbers by their parts, and the smaller
      position of two of one bits. So the result is the whole's first
      extreme with its bits, a NaN quieted, whatever the tiles' order. [Max]
      and [Min] drop the positions.
    - A float or complex [Sum] or [Prod] reduces by lane order's tree
      ({!Nx_kernel.section-orders}), a function of the term count alone, over
      the group's terms numbered in C order of the whole's reduced indices. A
      node whose terms one tile holds is a {e unit} there, as is a block that
      window edges split, on the device that holds most of its terms, the
      others moved there first. Each unit is one reduction of exactly its
      terms, into [float32] for a narrower float and part by part for a
      complex sum, with whether a term is a NaN in the same loop, and into the
      result's dtype where that differs; its terms are first laid out as
      [outputs; terms] where they are not all of its device's window. Each
      device combines the units up the tree by [Add] or [Mul] and rounds once.
      Where a unit has a NaN term, the result is the first such unit's
      reduction into the result's dtype, the first NaN term quieted; a complex
      product with a NaN part and no NaN term is the quiet NaN in each part.
    - A [Logsumexp] and [Moments] compute by their expansion ({!Expand.run}),
      whose reductions compute as above.

    For an exact or ranked reduction, each device moves one partial per output
    for each tile of its group it does not hold: of [t] tiles, [t - 1] rows of
    its window of the results, where reading the loads whole moves
    [(t - 1) / t] of each load's part on its group. For lane order, each
    device takes three rows per unit it does not compute, two where the
    carrier is the result's dtype: a tile whose terms form one run holds at
    most [2 ⌈log₂ b⌉] units of [b] blocks; a block that window edges split
    moves, once, the terms its device lacks, at most the block. Windows that
    interleave finely, a cut along a reduced axis after another, make a
    unit of each block: per block, a [Place] per other tile, an assembly, a
    reduction and a combination on each device, some 8000 of each for
    [[8192; 1000]] cut along axis 1. Laying a window's terms out is one pass,
    after a map where the program is not its operand.

    A [Contract] computes per group of its result's devices that hold one
    window of it, in its order of summation, the whole contraction's. A
    device's windows of [a] and [b] along the contracting pairs, alike, are
    its {e tile}; its terms are numbered as {!Nx_kernel.section-orders}
    numbers the whole's, and it lays its windows out as
    [[batch; rows; terms]] and [[batch; terms; columns]], the terms in
    their number's order. It reads [init] over its window of the result,
    from wherever [init] lies.

    - In lane order, a node of the tree whose terms one tile holds is a
      {e unit} there, as is a block that window edges split, on the device
      that holds most of its terms, the others' parts of [a] and [b] moved
      there first. Each unit is one contraction of exactly its terms into
      the accumulator from no [init]. Each device of the group takes every
      unit, combines them up the tree by [Add], adds [init] as the
      accumulator holds it, and rounds once into the result's dtype.
    - In chain order, the runs of the group's terms, in increasing number,
      each on the first device holding it, contract in turn into the
      accumulator: the first from [init] as the accumulator holds it, or
      none, each other from the sum before it, moved to it. The last sum
      rounds once into the result's dtype and moves to every device of the
      group.

    In lane order, each device takes its window of the results in the
    accumulator for each unit it does not compute, and a block that window
    edges split moves the parts of [a] and [b] its device lacks, once. In
    chain order, each run after the first takes the sum before it, and the
    devices holding the runs compute one after the other. Laying a window
    out is one copy where its axes are not in that order already.

    A [Scan] computes per group of its results' devices that hold one window
    of the axes off its axis; a device's window along its axis is its
    {e tile}, and the group's tiles are in their order along it. Each
    device's results have the bits of the scan on one device:

    - An integer or boolean [Sum], [Prod], [Max] or [Min] scans each tile's
      terms into the carrier, a float or complex [Max] or [Min] into the
      result's dtype, and an [Arg] into an extreme and its positions, which
      count from the axis's start. The carry into a tile is the last results
      of the tiles before it combined in order on its device; one map
      combines it with each of the tile's results: by [Add], [Mul],
      [Maximum] or [Minimum] for an integer or a boolean; for a float
      extreme, a NaN carry, else a NaN result, else their extreme; for an
      [Arg], the first extreme, as a reduction's partials combine.
    - A float or complex [Sum] or [Prod], a complex sum part by part as
      floats, continues the scan's chunks ({!Nx_kernel.type-reduction}).
      The total of each chunk but the last is a reduction of exactly its
      terms, with whether one is a NaN: one reduction per tile over the
      chunks it holds whole, and lane order's units over a chunk across
      tiles, as a [Reduce]'s, combined on the tile holding the next chunk's
      start. The carry into each chunk combines the totals before it in
      order, on the tile holding its start. A tile scans from the carry
      into its first chunk ({!Nx_kernel.S.scan}'s carry), and its terms
      before that from the running result before them in the carrier: the
      tile before it scans its last piece again into the carrier where the
      result's dtype is not, so tiles that share a chunk scan in turn. Where
      a term before a tile's first chunk, or before the tile, is a NaN,
      each of those results is the first such term, from a reduction of
      each piece of a tile within one chunk into the result's dtype.
    - A device whose tile is empty scans its window alone; one whose tile
      another holds takes that one's results.

    An exact or ranked scan moves one row of each tile's window of the
    results off the axis to the next tile, plus a map over the tile's
    results. A float sum or product moves its chunk totals as rows, three
    per chunk where the carrier is not the result's dtype, two elsewhere,
    and a chunk across tiles moves as a reduction's units do; it adds a
    reduction over the tile's terms, a map over its results but the first
    tile's, an assembly where the tile starts inside a
    chunk, and, for a narrower float, a scan of up to a chunk of 4096 terms
    per slice into the carrier. Tiles sharing a chunk scan one after the
    other. *)

val reduce :
  ('q. by:string -> 'q Value.prim -> 'q) ->
  by:string ->
  'd Route.t ->
  Nx_array.Layout.t ->
  int array ->
  Nx_kernel.Prog.t ->
  ('d, 'r) Value.reductions ->
  'd Value.load array ->
  'r
(** [reduce apply ~by route layout axes prog rs loads] is the [Reduce] of
    those fields, its loads lying where [route] reads them and [route]
    computing by {!Route.Partials}, computed through [apply] as stated
    above. *)

val scan :
  ('q. by:string -> 'q Value.prim -> 'q) ->
  by:string ->
  'd Route.t ->
  Nx_array.Layout.t ->
  int ->
  Nx_kernel.Prog.t ->
  ('d, 'r) Value.reduction ->
  'd Value.load array ->
  'r
(** [scan apply ~by route layout axis prog r loads] is the [Scan] of those
    fields and no carry, as {!reduce}. *)

val contract :
  ('q. by:string -> 'q Value.prim -> 'q) ->
  by:string ->
  'd Route.t ->
  order:Nx_kernel.order ->
  batch:(int * int) array ->
  contracting:(int * int) array ->
  acc:Nx_array.Dtype.any ->
  ('v, 's) Nx_array.Dtype.t ->
  ('a, 'b, 'd) Value.t ->
  ('c, 'e, 'd) Value.t ->
  ('v, 's, 'd) Value.t option ->
  ('v, 's, 'd) Value.t
(** [contract apply ~by route ~order ~batch ~contracting ~acc out a b init]
    is the [Contract] of those fields, as {!reduce}. *)
