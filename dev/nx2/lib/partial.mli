(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Operations whose route reads an operand cut along an axis they reduce
    ({!Route.Reduced}), as core operations applied through [apply]: on one
    device each, or a [Place].

    A [Reduce] computes per group of its results' devices that hold one window
    of its results. Each device of a group reduces the terms its loads'
    windows hold, one partial result per output; a device whose window along a
    reduced axis is empty has none. A device's window along the reduced axes
    is its {e tile}. Each device takes the partials of each tile it does not
    hold from the first device of its group that holds it, combines one per
    tile in the tiles' order, and rounds once into the result's dtype. The
    results lie where the route puts them, with the bits of the reduction on
    one device:

    - An integer or boolean [Sum], [Prod], [Max] or [Min] reduces into the
      carrier of its output's dtype, [int8] and [uint8] for [int4] and
      [uint4] and [bool] for [bit], and its partials combine by [Add], [Mul],
      [Maximum] or [Minimum], exact in the carrier.
    - An [Arg], and a float or complex [Max] or [Min], reduces as an [Arg]
      into its output's dtype, each position renumbered from C order of the
      window's reduced indices into C order of the whole's. Two partials
      combine into the first extreme: a NaN before any number and the smaller
      position of two NaNs, else the extreme in {!Nx_kernel.Prog}'s order,
      [-0] below [+0] and complex numbers by their parts, and the smaller
      position of two of one bits. So the result is the whole's first
      extreme with its bits, whatever the tiles' order. [Max] and [Min] drop
      the positions.
    - A float or complex [Sum] or [Prod], a [Logsumexp] and [Moments] reduce
      once over the loads placed whole along the axes they reduce, as the
      route reads loads that lie uncut there; so does every reduction where a
      reduced axis has no element.

    Each device moves one partial per output for each tile of its group it
    does not hold: of [t] tiles, [t - 1] rows of its window of the results,
    where reading the loads whole moves [(t - 1) / t] of each load's part on
    its group. Any other operation, a [Contract], reads its operands whole
    along the axes it reduces, placed as above. *)

val applies : 'r Value.prim -> bool
(** [applies op] is whether [op], whose operands lie where its route reads
    them, reads one cut along an axis it reduces. It allocates nothing where
    no operand of a [Reduce] or a [Contract] is cut. *)

val run :
  ('q. by:string -> 'q Value.prim -> 'q) -> by:string -> 'r Value.prim -> 'r
(** [run apply ~by op] is [op], for which {!applies} holds, computed through
    [apply] as stated above. *)
