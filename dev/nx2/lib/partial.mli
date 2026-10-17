(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Operations whose route reads an operand cut along an axis they reduce
    ({!Route.Reduced}), as core operations applied through [apply]: on one
    device each, or a [Place].

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
      extreme with its bits, whatever the tiles' order. [Max] and [Min] drop
      the positions.
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
      reduction into the result's dtype, the first NaN term's bits; a complex
      product with a NaN part and no NaN term is the quiet NaN in each part.
    - A [Logsumexp] and [Moments] compute by their expansion ({!Expand.run}),
      whose reductions compute as above.
    - Every reduction where a reduced axis has no element reduces once over
      the loads placed whole along the axes it reduces, as the route reads
      loads that lie uncut there.

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
    after a map where the program is not its operand. Any other operation, a
    [Contract], reads its operands whole along the axes it reduces, placed
    as above. *)

val applies : 'r Value.prim -> bool
(** [applies op] is whether [op], whose operands lie where its route reads
    them, reads one cut along an axis it reduces. It allocates nothing where
    no operand of a [Reduce] or a [Contract] is cut. *)

val run :
  ('q. by:string -> 'q Value.prim -> 'q) -> by:string -> 'r Value.prim -> 'r
(** [run apply ~by op] is [op], for which {!applies} holds, computed through
    [apply] as stated above. *)
