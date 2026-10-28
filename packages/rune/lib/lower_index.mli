(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Gathers, scatters, folds, assemblies and sorts as tolk graphs.

    Each function takes nodes as {!Lower_arith} holds elements and computes what
    {!Nx_kernel.S} states of its operation. Every shape must be static.

    {b Numerics.} These bounds implement the indexing rows of
    {!Rune.section-numerics}. A gather, a scatter by [Set], an assembly and a
    sort move elements, so they give the bits of nx's kernels, a NaN's sign and
    payload included. So does a scatter where no update lands. A scatter by
    [Max] or [Min] gives nx's bits except at a NaN result, whose sign and
    payload are unspecified, and a complex one has two NaN parts. A scatter by
    [Add] wraps integers exactly. Where [n] updates [u1], …, [un] land on an
    element [x], the float sum of [x] and those updates adds in the carrier
    ({!Nx_kernel.carrier}) in an association tolk chooses, and rounds once.
    Before that rounding it is within [γ(n) (|x| + Σ |ui|)] of the exact sum,
    where [γ] is as {!Nx_kernel.section-bounds} defines it for the carrier. Each
    part of a complex sum is such a sum. The sum is [-0] where [x] and every
    [ui] are [-0], and [+0] for other terms whose sum is zero. A fold of [m]
    terms [ti] sums them as a scatter's [Add] does, within [γ(m - 1) Σ |ti|]. It
    is [-0] where every [ti] is [-0], and [+0] where it has no term. It rounds a
    narrow float once, from [float32], while nx's kernel rounds each of its
    additions.

    {b Work.} The work of a scatter and of an assembly grows with their updates
    or pieces and exceeds that of nx's kernels. Where it would be more than 16
    times the kernel's, a scatter combines its updates in a loop of its own
    instead, within that bound ({!scatter}), and an assembly raises
    [Invalid_argument] ({!assemble}). *)

open Tolk

val gather : Nx_array.Dtype.any -> axis:int -> Ops.t -> Ops.t -> Ops.t
(** [gather d ~axis idx x] is [x], a value of [d], read at the [int64] positions
    [idx] along [axis], as {!Nx_kernel.S.gather} states. It has [idx]'s shape,
    and zero bits where a position lies outside the axis. Each element is one
    indexed load. *)

val scatter :
  ?dst:(Ops.t -> Ops.t) ->
  host:bool ->
  Ops.device ->
  Nx_array.Dtype.any ->
  Nx_kernel.combine ->
  axis:int ->
  idx:Ops.t ->
  updates:Ops.t ->
  Ops.t ->
  Ops.t
(** [scatter ~dst ~host device d c ~axis ~idx ~updates into] is [into], a value
    of [d], with [updates] combined by [c] at the positions [idx] holds along
    [axis], as {!Nx_kernel.S.scatter} states, for a program on [device], whose
    kernels compile for the host's processors where [host] holds
    ({!Tolk_engine.target}). Under [Set], the last update in C order of
    [updates] wins, whether or not the targets are promised distinct.

    For [n] positions and [k] updates along [axis], nx's kernel stores [n + k]
    elements. Where [n k <= 16 (n + k)], the scatter compares each element of
    [into] with each update along [axis], [n k] comparisons, in kernels it
    shares with its neighbours. Otherwise it combines the updates in a kernel of
    its own, into storage that holds [into], each element taking its updates in
    C order: [n + k] work at each index of the other axes where [host] holds,
    and at most [n + 16 k] elsewhere. There, a narrow float's [Add] sums in a
    [float32] copy of [into] and marks where updates land in a byte per element
    of [into].

    Every other combine of that form combines into [dst v], where [v] is
    [into]'s parts flat in C order: storage of [v]'s dtype and element count
    that holds [v], such as a buffer once [v] is stored into it, an
    {!Ops.after} over the store. The result is that storage once combined. The
    scatter writes over it, so no value but the result may lie in it once the
    scatter's kernel runs: it is fresh storage, or a donated argument's that
    the result takes, which each value that reads the argument as it was reads
    before. [dst] defaults to a fresh buffer that [v] is stored into.

    Raises [Invalid_argument] for a boolean [Add]. *)

val fold :
  Nx_array.Dtype.any -> Nx_kernel.pad -> int iarray -> Ops.t -> Ops.t
(** [fold d pad shape x] is the adjoint, applied to [x], of a [Padded] load by
    [pad] from a value of [d] and [shape], as {!Nx_kernel.S.fold} states. [x]
    has the load's result shape. Its work is [O(size·(l·step + n))] at each
    index of the other axes, for [l] windows of [size] elements [step] apart
    over an axis of [n].

    Raises [Invalid_argument] for a boolean [d]. *)

val assemble :
  ('v, 's) Nx_array.Dtype.t ->
  'v ->
  int iarray ->
  (Nx_array.Move.range iarray * Ops.t) list ->
  Ops.t
(** [assemble dt fill shape pieces] is the value of [dt] and [shape] made of
    [pieces], as {!Nx_kernel.S.assemble} states. Its element at an index is that
    of the last piece whose region holds the index, or [fill] where none does.
    Each piece selects its elements over the earlier ones, making [p N] selects
    for [p] pieces of [m] elements in all into [N] elements. nx's kernel stores
    [N + m].

    Raises [Invalid_argument] where [p N > 16 (N + m)], counting only the pieces
    that hold an element. *)

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
