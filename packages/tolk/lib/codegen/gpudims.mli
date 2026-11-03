(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Launch dimensions.

    A kernel's global and local loops ({!Ops.Axis_type.Global},
    {!Ops.Axis_type.Warp} and {!Ops.Axis_type.Local} ranges) become the hardware
    indices of a launch: workgroup indices and thread indices, numbered from [0]
    on each of their axes. A target has at most three of each and bounds each
    ([global_max], [local_max] and [global_prod_max] of {!Renderer.t}), so loops
    are merged or split to fit, and each loop's index is recovered from the
    hardware indices by division and remainder. The hardware indices stay
    ranges, {!Ops.Axis_type.Global} and {!Ops.Axis_type.Local} ones whose
    identity is their axis, so that their index arithmetic is lowered as any
    other's; once lowered, they become {!Op.Special} nodes named [gidx0],
    [gidx1], ... and [lidx0], [lidx1], .... *)

val grouped_dims :
  ?reverse:bool -> string -> Ops.sint list -> int list option -> Ops.t list
(** [grouped_dims ~reverse prefix dims max_sizes] is the index of each loop of
    sizes [dims], in terms of hardware indices, one per axis, whose sizes fit
    [max_sizes] if given. A hardware index is a range identified by its axis,
    [0] first, {!Ops.Axis_type.Global} if [prefix] is ["gidx"] and
    {!Ops.Axis_type.Local} otherwise. The sizes are fit as follows:
    - if [dims] has more sizes than [max_sizes] or one exceeds its bound
      (counting a symbolic size at its upper bound), adjacent sizes are merged,
      the leftmost pair that fits first, until they fit;
    - if merging fails, the sizes are laid on three axes, and each axis in turn
      whose size exceeds its bound is split by the size's least divisor until it
      fits, the quotient kept and the divisor moved to the next axis. The last
      axis's divisors move to the first, which is not checked again, so the
      first size can end above its bound. A third axis of size [1] is dropped.

    If [reverse] (default [false]), [dims] are laid out on the axes in reverse
    order. A loop that keeps its own axis is that hardware index itself.

    Raises [Invalid_argument] if the sizes cannot be made to fit: merging fails
    and a size to split is symbolic or has no divisor up to the ceiling of its
    square root, or [max_sizes] has fewer than three bounds. *)

val group_gpudims : Renderer.t -> Ops.t -> Ops.t option
(** [group_gpudims r sink] is the kernel [sink] with its global loops replaced
    by workgroup indices ({!grouped_dims} [~reverse:true] with prefix ["gidx"])
    and its warp and local loops by thread indices (prefix ["lidx"]), within
    [r]'s bounds. Global loops take the order of their identities; a warp
    takes the first thread axis, then local loops follow in the order of their
    identities. A warp keeps its own thread axis. A store to global memory
    whose index, its validity included, does not use every workgroup and
    thread index is masked to the threads whose unused indices are [0], so it
    runs once for each point of the indices it uses. Every loop is replaced at
    once, so a hardware index that is the same node as a loop, of one
    identity, type and size, stays. It is [None] if [sink] carries no kernel
    information, has hardware indices already, or has no global or local
    loop.

    A tensor core ({!Op.Wmma}) computes on its lanes as the hardware numbers
    a warp's threads, so a kernel holding one of more than one thread must
    have a warp of exactly its threads, which takes the first thread axis
    whole. A kernel without global or local loops is not checked, and is
    [None] as above.

    Raises [Invalid_argument] if {!grouped_dims} does, if the index of a store
    to mask has more than one index, or if [sink] has a global or local loop,
    holds a tensor core of more than one thread, and its first thread loop is
    not a warp of as many threads. *)

val pm_group_gpudims : (Renderer.t, Ops.t) Ops.Pattern_matcher.t
(** [pm_group_gpudims] applies {!group_gpudims} to sinks. It is to run once
    ({!Ops.type-pass} [Once]): the indices it makes are global and local
    ranges, which it would group again. *)

val pm_range_to_special : (unit, Ops.t) Ops.Pattern_matcher.t
(** [pm_range_to_special] replaces each {!Ops.Axis_type.Global} and
    {!Ops.Axis_type.Local} range by the hardware index ({!Op.Special}) of its
    axis, its identity's last part: [gidx0], [lidx1], ..., keeping its bound,
    and splits ends ({!Linearizer.pm_split_ends}), which drops the hardware
    indices from the ranges they close. *)
