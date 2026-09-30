(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Launch dimensions.

    A kernel's global and local loops ({!Ops.Axis_type.Global},
    {!Ops.Axis_type.Warp} and {!Ops.Axis_type.Local} ranges) become the hardware
    indices of a launch ({!Op.Special}): workgroup indices named [gidx0],
    [gidx1], ... and thread indices named [lidx0], [lidx1], .... A target has at
    most three of each and bounds each ([global_max], [local_max] and
    [global_prod_max] of {!Renderer.t}), so loops are merged or split to fit,
    and each loop's index is recovered from the hardware indices by division and
    remainder. *)

val grouped_dims :
  ?reverse:bool -> string -> Ops.sint list -> int list option -> Ops.t list
(** [grouped_dims ~reverse prefix dims max_sizes] is the index of each loop of
    sizes [dims], in terms of hardware indices named [prefix] followed by their
    axis, [0] first, whose sizes fit [max_sizes] if given:
    - if [dims] has more sizes than [max_sizes] or one exceeds its bound
      (counting a symbolic size at its upper bound), adjacent sizes are merged,
      the leftmost pair that fits first, until they fit;
    - if merging fails, a size that exceeds its bound is split by its least
      divisor, the quotient kept and the divisor moved to the next axis, until
      each fits.

    If [reverse] (default [false]), [dims] are laid out on the axes in reverse
    order. A loop that keeps its own axis is that hardware index itself.

    Raises [Invalid_argument] if the sizes cannot be made to fit: merging fails
    and a size to split is symbolic or has no divisor up to the ceiling of its
    square root, or [max_sizes] has fewer than three bounds. *)

val add_gpudims : Renderer.t -> Ops.t -> Ops.t option
(** [add_gpudims r sink] is the kernel [sink] with its global loops replaced by
    workgroup indices ({!grouped_dims} [~reverse:true] with prefix ["gidx"]) and
    its warp and local loops by thread indices (prefix ["lidx"]), in order of
    their axis identifiers, within [r]'s bounds. A warp keeps its own thread
    axis. A store to global memory whose index does not use every thread index
    is masked to the threads whose unused indices are [0]. It is [None] if
    [sink] carries no kernel information, has hardware indices already, or has
    no global or local loop.

    Raises [Invalid_argument] if {!grouped_dims} does, or if the index of a
    store to mask has more than one index. *)

val pm_add_gpudims : (Renderer.t, Ops.t) Ops.Pattern_matcher.t
(** [pm_add_gpudims] applies {!add_gpudims} to sinks, and replaces each
    {!Ops.Axis_type.Device} range by the variable [_device_num], bound per
    device at launch, which no end closes. *)
