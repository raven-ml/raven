(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Computing operations by their set's kernels, each device its window.

    This module computes an operation whose operands have bytes and lie
    where its route reads them ({!Prim.route}). Its callers choose the
    route, read the operands there ({!placer}) and give it the memory a
    donation frees. It settles no check. Below it lie {!Prim}, {!Route} and
    {!Place}; above it {!Partial}, {!Check} and {!Eval} call it, and it calls
    none of them.

    An operation's results lie at its route's result. Each device of that
    placement computes its {e window} of the results ({!Devices.window}) by
    one call of the set's kernels, over each operand's window there
    ({!Route.window}); a program that reads coordinates adds the window's
    first index to them. Where a layout expresses a movement or a bitcast,
    its result is a view of the operand's memory and no kernel runs.
    Otherwise the operand is copied and the result is a view of the copy. A
    result shares an operand's memory only as its view, or where a donation
    gives the operand's memory to it.

    A {e refusal} of the kernels raises through {!Nx_array.refused} naming
    the [by] the caller passes. A {e decline} raises [Invalid_argument]
    naming [by], the kernels, the operation (a one-node map's kind, a
    routine's name, what a reduction or a scan computes, as
    ["Reduce Sum and Moments"]), the dtypes, the device and the move the
    program makes, [Nx.place] onto a set whose kernels compute it: nx
    computes no case on the kernels' behalf and moves no data between
    devices on its own. A set without kernels raises [Invalid_argument]
    naming [by] before any kernel runs.

    Over devices of one architecture, a result's bits are those of the same
    operation on one device, by the window law ({!Route.window}), wherever
    the operands lie, but for a float or complex contraction, whose library
    chooses its association from each window's extents. That contraction's
    bits are within its bound, and the same on every run for a given
    placement.

    Unless a value's doc says otherwise, it raises as stated above, and
    allocates the results' memory and the values over it. *)

(** {1:windows Windows} *)

val windows :
  by:string ->
  ?into:Nx_array.any ->
  unit Route.t ->
  Nx_array.Layout.t array ->
  'r Value.prim ->
  'r
(** [windows ~by ?into route layouts op] is [op] computed by windows, its
    operands lying where [route] reads them, its results at [route]'s result
    laid out as [layouts] ({!Prim.layouts}), fresh but the first, which is
    [into] where it is given: memory of a donated operand ({!Donate.reuse})
    that holds the first result's window on the one device of the route's
    result.

    Raises [Invalid_argument] as an internal fault for a route that
    computes by partials ({!Route.Partials}), a [Place] or a [Check]. *)

(** {1:local One device} *)

val is_local : 'r Value.prim -> bool
(** [is_local op] is whether [op] is a map whose loads are live arrays read
    plainly at physically one placement. It allocates nothing. *)

val local : by:string -> 'r Value.prim -> 'r
(** [local ~by op] is the map [op], for which {!is_local} holds, computed on
    its loads' one device without a route: its rule checked
    ({!Prim.layouts}), its results fresh there, the set's kernels called
    once.

    Raises [Invalid_argument] naming [by] where [op]'s operands break its
    rule, and as an internal fault for an operation {!is_local} refuses. *)

(** {1:placing Placing} *)

val place :
  by:string ->
  'e Devices.placement ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'e) Value.t
(** [place ~by p x] is [x]'s elements at [p] ({!Place.value}). It settles
    no check on [x]'s set. An array of the result over memory of [x]'s,
    kept or borrowed, marks that memory shared ({!Rig.Claim.share}), so
    that no donation of either value writes what the other reads.

    Raises what {!Place.value} raises. *)

val placer :
  by:string ->
  'd Devices.placement ->
  ('v, 's, 'd) Value.t ->
  ('v, 's, 'd) Value.t
(** [placer ~by p x] is [x] read at [p], an operand where its route reads it
    ({!Route.t}): [x] where its placement equals [p], and otherwise the value
    at [p] whose array on each device is a view of [x]'s array there at that
    device's window, or an array of no bytes where the window is empty. It
    moves no bytes.

    Raises [Invalid_argument] as an internal fault where [x] has no array
    holding a nonempty window of [p] on its device, as {!Place.view}
    does. *)

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
