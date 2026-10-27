(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Gathers, scatters, folds, assemblies and sorts as tolk graphs.

    Each function takes nodes as {!Lower_arith} holds elements and computes
    what {!Nx_kernel.S} states of its operation, of static shapes.

    {b Numerics.} A gather, a scatter by [Set], an assembly and a sort move
    elements: they give eager's bits, a NaN's sign and payload included, and
    so does a scatter where no update lands. A scatter by [Max] or [Min] gives
    eager's bits but for a NaN, of no promised sign or payload, a complex one
    with two NaN parts. A scatter by [Add] wraps integers exactly. Its float
    sum, of [x] and the [n] updates [u1], …, [un] that land on it, adds in the
    carrier ({!Nx_kernel.carrier}) in an association tolk chooses and rounds
    once: it is within [γ(n) (|x| + Σ |ui|)] of the exact sum before that
    rounding, as {!Lower_reduce} states its sums, each part of a complex sum
    such a sum. It is [-0] where [x] and every [ui] are, and [+0] for other
    terms of zero sum. A fold of [m] terms [ti] sums them as a scatter's [Add]
    does, within [γ(m - 1) Σ |ti|], [-0] where every [ti] is and [+0] where it
    has none: it rounds a narrow float once from [float32], where eager rounds
    each addition.

    {b Work.} A scatter and an assembly do more work than eager does, growing
    with their updates or pieces, and raise [Invalid_argument] where it would
    be more than 16 times eager's, as each states. *)

open Tolk

val gather : Nx_array.Dtype.any -> axis:int -> Ops.t -> Ops.t -> Ops.t
(** [gather d ~axis idx x] is the node [x] of a value of [d] read at the
    [int64] positions [idx] along [axis], as {!Nx_kernel.S.gather} states: of
    [idx]'s shape, the element of zero bits where a position lies outside the
    axis. Each element is one indexed load. *)

val scatter :
  Nx_array.Dtype.any ->
  Nx_kernel.combine ->
  axis:int ->
  idx:Ops.t ->
  updates:Ops.t ->
  Ops.t ->
  Ops.t
(** [scatter d c ~axis ~idx ~updates into] is [into], a value of [d], with
    [updates] combined by [c] at the positions [idx] holds along [axis], as
    {!Nx_kernel.S.scatter} states. Each element of [into] meets each update
    along [axis]: [n k] comparisons for [n] positions and [k] updates along
    it, where eager stores [n + k] elements. Under [Set], the last update in C
    order of [updates] wins; a promise of distinct targets changes nothing.

    Raises [Invalid_argument] for a boolean [Add], and where
    [n k > 16 (n + k)]. *)

val fold :
  Nx_array.Dtype.any -> Nx_kernel.pad -> int iarray -> Ops.t -> Ops.t
(** [fold d pad shape x] is the adjoint of a [Padded] load by [pad] of a value
    of [d] and of [shape], [x] of the shape the load gives, as
    {!Nx_kernel.S.fold} states. Each window's terms sum as copies of its axis
    laid end to end: [O(size·(l·step + n))] work at each index of the other
    axes, for [l] windows of [size] elements [step] apart over an axis of [n].

    Raises [Invalid_argument] for a boolean [d]. *)

val assemble :
  ('v, 's) Nx_array.Dtype.t ->
  'v ->
  int iarray ->
  (Nx_array.Move.range iarray * Ops.t) list ->
  Ops.t
(** [assemble dt fill shape pieces] is the value of [dt] and [shape] whose
    element at an index is the last piece's whose region holds it and [fill]
    where none does, as {!Nx_kernel.S.assemble} states. Each piece selects its
    elements over the earlier ones: [p N] selects for [p] pieces of [m]
    elements in all into [N] elements, where eager stores [N + m].

    Raises [Invalid_argument] where [p N > 16 (N + m)], pieces of no element
    not counted. *)

val sort :
  Nx_array.Dtype.any ->
  axis:int ->
  descending:bool ->
  k:int option ->
  Ops.t ->
  Ops.t * Ops.t
(** [sort d ~axis ~descending ~k x] is the elements of [x], a value of [d],
    sorted stably along [axis], then their [int64] positions, as
    {!Nx_kernel.S.sort} states; with [Some k], the first [k] of each slice.
    Each slice sorts by a bitonic network over its keys packed above their
    positions, one pass per 32 bits of key: [O(n log² n)] work and about
    [log² n / 2] kernels a pass for [n] elements, two passes for a 64-bit
    dtype or [complex64], four for [complex128].

    Raises [Invalid_argument] for an axis of more than [2{^31}] elements. *)
