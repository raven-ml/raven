(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Multi-device collective reduction.

    Builds allreduce computation graphs using naive, hierarchical, ring, or all-to-all
    strategies depending on device count, element count, and the [RING],
    [ALL2ALL], and [RING_ALLREDUCE_THRESHOLD] context variables. *)

val handle_allreduce :
  Tolk_uop.Uop.t ->
  op:Tolk_uop.Ops.t ->
  device:Tolk_uop.Uop.device ->
  Tolk_uop.Uop.t option
(** [handle_allreduce buf ~op ~device] builds a reduction graph that
    combines every shard of [buf] with [op] and places the result
    on [device].

    Returns [None] if [buf] is not on multiple devices. Symbolic shapes use
    padded maximum-size transfers and retain their logical extents. Ring and
    all-to-all strategies require concrete shapes. When [ALLREDUCE_NODE_NDEVS]
    divides the device count, hierarchical reduction first combines chunks
    within each node, then across corresponding ranks, and gathers locally.

    The strategy is selected automatically:
    {ul
    {- {e Naive} when the device count is [<= 2] or the element
       count is below [RING_ALLREDUCE_THRESHOLD] (default 256k).}
    {- {e All-to-all} when [ALL2ALL >= 2], or [ALL2ALL >= 1] and
       the size exceeds the threshold with [> 2] devices.}
    {- {e Ring} when [RING >= 2], or [RING >= 1] and the size
       exceeds the threshold with [> 2] devices.}} *)

val collective :
  name:string ->
  device:Tolk_uop.Uop.device ->
  like:Tolk_uop.Uop.t ->
  Tolk_uop.Uop.t ->
  (dst:Tolk_uop.Uop.t -> src:Tolk_uop.Uop.t -> Tolk_uop.Uop.t list) ->
  Tolk_uop.Uop.t
(** [collective ~name ~device ~like src body] is the value, of [like]'s shape
    and dtype on [device], that a precompiled [CALL] named [name] computes
    from [src]. The call's two arguments are storage: a fresh allocation for
    the result, and the storage [src] views, or [src] made contiguous when it
    is not a view of storage. [body ~dst ~src] gives the stores into [dst], a
    view of the allocation at [like]'s shape, from [src], the same view of the
    call's input parameter. Every collective has this (dst, src) contract, so
    a backend can replace a body with a library call. *)

val create_allreduce_function :
  Tolk_uop.Uop.t ->
  op:Tolk_uop.Ops.t ->
  device:Tolk_uop.Uop.device ->
  Tolk_uop.Uop.t option
(** [create_allreduce_function buf ~op ~device] is the {!collective} named
    ["allreduce"] whose body is {!handle_allreduce} over [buf].

    Returns [None] if [buf] is not on a multi-device. *)
