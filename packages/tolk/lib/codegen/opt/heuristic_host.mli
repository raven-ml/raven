(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** The host's rules of the hand-coded optimisations.

    On the host a kernel's lanes are counted in its target's vector registers:
    the accumulators of its reduces take at most half of them, and each
    conversion and each iteration a bounded number of lanes. These rules
    decide what a GPU, whose outputs are its threads, does not count.
    {!Heuristic} states what a kernel gets from them. *)

val on_host : Postrange.Scheduler.t -> bool
(** [on_host k] is [true] iff [k] renders for the [CPU] device. *)

val budgeted : Postrange.Scheduler.t -> bool
(** [budgeted k] is [true] iff [k] renders for the [CPU] device of a renderer
    that knows its vector registers. *)

val fits : ?upcast:int * int -> ?unroll:int * int -> Postrange.Scheduler.t -> bool
(** [fits ?upcast ?unroll k] is whether [k] stays within the host's budget once
    [upcast] splits [n] lanes off an axis, given as [(axis, n)], and [unroll]
    unrolls an axis by [n] lanes, [0] for whole. It is [true] off the host. *)

val host_elementwise : Postrange.Scheduler.t -> bool
(** [host_elementwise k] is [true] iff [k] renders for the host and holds no
    reduce. *)

val upcast_vectors : Postrange.Scheduler.t -> unit
(** [upcast_vectors k] upcasts the last output axes of [k], a kernel without a
    reduce, while its lanes have room in a vector. *)

val upcast_contiguous : Postrange.Scheduler.t -> unit
(** [upcast_contiguous k] upcasts the axis [k]'s stores write at unit stride,
    when every access reading it does so at unit stride, by the largest power
    of two that fills a vector and fits the budget. In a kernel that sums
    products an access may instead read the axis apart and the last reduce
    axis at unit stride, its lanes gathered by a transpose: the axis then
    takes the largest power of two that fills one register and fits, and the
    output axis the gathering accesses do not read the largest that fits
    beside it. In a kernel that sums products, an axis whose largest
    power-of-two divisor holds fewer than 16 bytes of lanes is first padded to
    a multiple of 16 bytes, where that adds at most a third of its extent. *)

val unroll_vector : Postrange.Scheduler.t -> fits:(int -> bool) -> int -> int -> unit
(** [unroll_vector k ~fits unrolled axis] unrolls [axis] by the largest power
    of two that fills a vector beside the [unrolled] lanes and that [fits]
    admits. *)

val unroll_room : Postrange.Scheduler.t -> bool
(** [unroll_room k] is whether the lanes [k] has upcast leave an unroll room:
    fewer registers of them than the target has, where the target knows its
    registers. *)

val share_decodes : Postrange.Scheduler.t -> Ops.t -> unit
(** [share_decodes k m] upcasts the output axes along which the decoded
    operands of [k]'s summed product [m] read one value, so a tile's rows
    share a decode of it. *)

val tile_rows : Renderer.t -> int option
(** [tile_rows ren] is the most rows {!share_decodes} upcasts along an axis
    whose stores lie apart, where [ren] renders for the [CPU] device, and
    [None] elsewhere. *)

val tile_outputs : Renderer.t -> Dtype.t -> int option
(** [tile_outputs ren dt] is the most outputs of a reduce of [dt] that the
    accumulators' share of [ren]'s registers holds, where [ren] renders for
    the [CPU] device and knows its registers, and [None] elsewhere. *)

val copy_entries : Dtype.t -> int
(** [copy_entries dt] is the most entries of a value into a host tile's lanes
    that a copy of the value stored at [dt] costs. *)

val launch_entries : int
(** [launch_entries] is the entries into host tiles' lanes that the launch of
    a kernel of its own costs. *)

val held_registers : Postrange.Scheduler.t -> Renderer.registers -> int
(** [held_registers k file] is the registers of [file] that [k]'s reduces'
    accumulators take. *)
