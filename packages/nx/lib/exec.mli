(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Computing operations by their set's kernels on one device, and the eager
    engine of partitions.

    This module computes an operation whose operands have bytes and lie on
    one device, where its results lie too. Its callers choose the route
    ({!Prim.val-route}) and give it the memory a donation frees. It settles
    no check. An operation over several devices computes as its partition
    ({!Partition}) on {!eager}, whose steps compute here, view arrays, and
    move them through nx.array and rig. This module lies above {!Prim},
    {!Route} and {!Partition}; {!Check} and {!Eval} call it.

    Where a layout expresses a movement or a bitcast, its result is a view
    of the operand's memory and no kernel runs. Otherwise the operand is
    copied and the result is a view of the copy. A result shares an
    operand's memory only as its view, or where a donation gives the
    operand's memory to it. A program that reads coordinates reads them
    from its loop's start, which a partition's window offsets
    ({!Prim.local}).

    A {e refusal} of the kernels raises through {!Nx_array.refused} naming
    the [by] the caller passes. A {e decline} raises [Invalid_argument]
    naming [by], the kernels, the operation (a one-node map's kind, a
    routine's name, what a reduction or a scan computes, as
    ["Reduce Sum and Moments"]), the dtypes, the device and the move the
    program makes, [Nx.place] onto a set whose kernels compute it: nx
    computes no case on the kernels' behalf and moves no data between
    devices on its own. A set without kernels raises [Invalid_argument]
    naming [by] before any kernel runs, but for a map of no loads, a
    creation, which the host's kernels compute and {!place} puts on the
    device.

    Over devices of one architecture, a result's bits are those of the same
    operation on one device, by the window law ({!Route.window}), wherever
    the operands lie, but for a float or complex contraction, whose library
    chooses its association from each window's extents. That contraction's
    bits are within its bound, and the same on every run for a given
    placement.

    Unless a value's doc says otherwise, it raises as stated above, and
    allocates the results' memory and the values over it. *)

(** {1:computing One device} *)

val windows :
  by:string ->
  ?into:Nx_array.any ->
  unit Devices.placement ->
  Nx_array.Layout.t array ->
  'r Value.prim ->
  'r
(** [windows ~by ?into p layouts op] is [op] computed at the one-device
    placement [p], where its operands must lie, its results of the shapes
    of [layouts] ({!Prim.val-layouts}), fresh and C-contiguous but the
    first, which is [into] where it is given: memory of a donated operand
    ({!Donate.reuse}) that holds the first result.

    Raises [Invalid_argument] as an internal fault for a [Place] or a
    [Check]. *)

val run :
  by:string ->
  ?into:'d Value.any ->
  unit Devices.placement ->
  Nx_array.Layout.t array ->
  'r Value.prim ->
  'r
(** [run ~by ?into p layouts op] is {!windows}[ ~by p layouts op] over the
    memory of [into]'s array. Without [into], a map whose loads lie beside
    each other ({!is_local}), and whose one result, if alone, is laid out
    C-contiguous from offset 0, computes by {!local}, which gives it the
    same results. *)

val eager : Partition.engine
(** [eager] is the engine of eager execution. {!run} computes its
    operations. Its pieces are a value's arrays, and its views are views of
    them that leave their memory unshared. A move takes an array on the
    device itself, else one whose memory the host and the device both
    address ({!Rig.reaches}), as a borrow, else a copy of the first
    ({!Nx_array.to_device}). A gather fills a fresh C-contiguous array from
    the boxes by {!Nx_array.transfer}, on the host for an element narrower
    than a byte where the host does not address the device's memory, then
    brought to the device. Moves and gathers return as nx.array's placement
    does ({!Nx_array.section-placement}): where a device copies, as
    {!Rig.Buffer.submit_copy} returns. An empty window is an array of no
    bytes of its own. *)

(** {1:local Maps beside each other} *)

val is_local : 'r Value.prim -> bool
(** [is_local op] is whether [op] is a map whose loads are live arrays read
    plainly at physically one placement. It allocates nothing. *)

val local : by:string -> 'r Value.prim -> 'r
(** [local ~by op] is the map [op], for which {!is_local} holds, computed on
    its loads' one device without a route: its rule checked
    ({!Prim.val-layouts}), its results fresh there, the set's kernels called
    once.

    Raises [Invalid_argument] naming [by] where [op]'s operands break its
    rule, and as an internal fault for an operation {!is_local} refuses. *)

(** {1:placing Placing}

    Placing reads and writes memory through nx.array and rig alone and runs
    no kernel, so a set without kernels places. Each device of the
    destination holds its window of the value and allocates at most the
    window's bytes. Nothing holds the whole value unless the destination
    does. *)

val value :
  by:string ->
  'e Devices.placement ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'e) Value.t
(** [value ~by p x] is {!Partition.place} of [x] at [p] on {!eager}: over
    [x]'s own arrays when [x]'s placement equals [p].

    Raises [Invalid_argument] naming [by] if [p] cuts an axis [x] lacks, and
    if [x] is traced or a handle, {!Rig.Lost} for a lost device, and what
    {!Rig.Buffer.submit_copy} raises. *)

val place :
  by:string ->
  'e Devices.placement ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'e) Value.t
(** [place ~by p x] is {!value}[ ~by p x]. It settles no check on [x]'s
    set. An array of the result over memory of [x]'s, kept or borrowed,
    marks that memory shared ({!Rig.Claim.share}), so that no donation of
    either value writes what the other reads. *)

(** {1:fast Fast paths}

    An operation of live arrays on one device whose set has kernels may be
    computed without building it. These values let the dispatch ({!Eval})
    test for that case and compute it; each answers what the operation
    would, by its kernels, and raises as it would. *)

val live_kernels : ('v, 's, 'd) Value.t -> (module Nx_kernel.S) option
(** [live_kernels x] is the kernels of [x]'s set where [x] is a live
    [Array] and the set has kernels. It allocates nothing. *)

val beside : ('v, 's, 'd) Value.t -> ('w, 'r, 'd) Value.t -> bool
(** [beside x y] is whether [x] and [y] are [Array]s at physically one
    placement and [y] is live. It allocates nothing. *)

val view_of : ('v, 's) Nx_array.t -> unit
(** [view_of a] marks [a]'s memory shared ({!Rig.Claim.share}), as a view
    over it made outside any handle chain needs: no donation of [a]'s value
    may then write what the view reads. *)

val map :
  by:string ->
  (module Nx_kernel.S) ->
  'd Devices.placement ->
  Nx_kernel.Prog.t ->
  ('w, 'r) Value.dtype ->
  ('v, 's) Nx_array.t ->
  Nx_kernel.load array ->
  ('w, 'r, 'd) Value.t
(** [map ~by kernels at p dt x loads] is the live value at [at] of the
    program [p], of one output of [dt], over [loads], computed by [kernels]
    on [x]'s device, which is [at]'s one device: one call of their [map].
    The result shares [x]'s layout where that is C-contiguous from offset 0
    and of [dt], and [dt] is at least a byte wide; it is fresh and
    C-contiguous otherwise. *)

val contract :
  by:string ->
  batch:(int * int) iarray ->
  contracting:(int * int) iarray ->
  acc:Nx_array.Dtype.any ->
  ('v, 's) Value.dtype ->
  ('a, 'b, 'd) Value.t ->
  ('c, 'e, 'd) Value.t ->
  ('v, 's, 'd) Value.t option ->
  ('v, 's, 'd) Value.t option
(** [contract ~by ~batch ~contracting ~acc dt a b init] is [Some] the
    contraction [Value.Contract] of [a] and [b] from [init] into [dt], where
    its operands are live arrays on one device at physically one placement
    whose set has kernels and its pairs fit them: it builds no operation and
    allocates the C-contiguous result of the shape
    {!Nx_kernel.Shape.contract} gives. [None] otherwise, for the operation
    to decide. *)
