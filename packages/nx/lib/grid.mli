(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Device grids: where each device's window of a value lies.

    A grid is one device, or distinct devices in row-major order over the grid's
    extents, with, for each cut axis of a value, the grid axes that cut it,
    major first, and the grid axes over which the value is a {e sum}. A grid
    axis that neither names holds copies. Along summed grid axes, each device
    holds a {e term}: a value of the whole's shape, cut as the grid cuts it,
    and the value is the sum of the terms of each {e group}, the devices that
    differ only along the summed axes. The terms of a group are numbered in
    row-major order over the summed axes in the order the grid lists them,
    the first major.
    Devices are positions in a set. A grid is kept in normal form, in which
    extents are at least 2, adjacent grid axes merge when neither is named or
    one cut, or the sums, names both in order, and a grid of one device is
    that device. So two grids that give each device the same window and the
    same term are equal. A grid is immutable, and any domain reads it.

    Nothing of the engine lies below this module; {!Devices} lays a grid
    over a set's devices. *)

type t

val device : int -> t
(** [device k] is the device [k] alone. *)

val v :
  devices:int array ->
  extents:int array ->
  cuts:(int * int array) array ->
  sums:int array ->
  (t, string) result
(** [v ~devices ~extents ~cuts ~sums] is the grid over [devices], in row-major
    order over [extents], each [(axis, over)] of [cuts] cutting the value's
    [axis] over the grid axes [over], major first, the value a sum over the
    grid axes [sums], listed major first. It is [Error] unless the extents are positive and
    multiply to the number of devices, the devices are distinct and not
    negative, and the cut axes and the grid axes the cuts and [sums] name are
    distinct and in range. *)

val devices : t -> int array
(** [devices g] is a fresh array of [g]'s devices in row-major order. *)

val count : t -> int
(** [count g] is the number of [g]'s devices. *)

val one : t -> int option
(** [one g] is [Some k] iff [g] is the device [k] alone. *)

val position : t -> int -> int
(** [position g k] is the position of the device [k] in {!devices}[ g], or
    [-1] where [g] lacks it. It allocates nothing. *)

val cuts : t -> (int * int) array
(** [cuts g] is each cut axis with its number of tiles, by increasing axis. *)

val is_cut : t -> bool
(** [is_cut g] is [cuts g <> [||]]. It allocates nothing. *)

val is_partial : t -> bool
(** [is_partial g] is [sums g <> [||]]. *)

val sums : t -> int array
(** [sums g] is a fresh array of the grid axes over which [g]'s value is a
    sum, major first, numbered as {!extents} orders them: [[||]] where it is
    not partial. *)

val term : t -> int -> int
(** [term g i] is the number of the term the [i]th device of [devices g]
    holds within its group: [0] where [g] is not partial.

    Raises [Invalid_argument] if [i] is not a position of [devices g]. *)

val window : t -> int array -> int -> (Nx_array.Move.range array, string) result
(** [window g shape i] is the window of a value of [shape] that the [i]th device
    of [devices g] holds: the whole shape on an axis no cut names. Along a cut
    axis of extent [n] in [t] tiles, numbered major first over the cut's grid
    axes, tile [j] holds the [n / t] elements after those of the tiles before
    it, one more where [j < n mod t]. The windows of one cut are disjoint,
    cover the axis in tile order and differ by one element at most; one is
    empty only where [n < t]. It is [Error] if a cut axis is not an axis of
    [shape].

    Raises [Invalid_argument] if [i] is not a position of [devices g]. *)

val map_axes : (int -> int) -> t -> t
(** [map_axes f g] cuts the value's axis [f a] where [g] cuts [a], over the same
    grid axes: [g] for a value whose axes [f] renumbers. [f] is injective on
    [g]'s cut axes; nothing checks it. *)

val uncut : t -> axis:int -> t
(** [uncut g ~axis] is [g] with the value's [axis] whole on every device: the
    grid axes that cut it hold copies. If [g] does not cut [axis], it is a
    grid {!equal} to [g]. *)

val closed : t -> t
(** [closed g] is [g] where the value is no sum: the grid axes it summed over
    hold copies. If [g] is not partial, it is a grid {!equal} to [g]. *)

val sum : t -> axis:int -> t
(** [sum g ~axis] is [g] with the value's [axis] whole on every device and
    the grid axes that cut it summed after any [g] sums already, its terms
    numbered as the tiles of [axis] were: [g] for a value that a contraction
    over [axis] made. If [g] does not cut [axis], it is a grid {!equal} to
    [g]. *)

val unsum : t -> axis:int -> (t, string) result
(** [unsum g ~axis] is [g] with the value's [axis] cut over the grid axes
    [g] sums over, in their order, and no sum: each device holds tile
    {!term}[ g i] of [axis]. It is a grid {!equal} to [g] where [g] is not
    partial, and [Error] where [g] cuts [axis]. *)

val cut_like :
  t -> like:t -> axes:int list -> axis:int -> (t, string) result
(** [cut_like g ~like ~axes ~axis] is [g] with the value's [axis] also cut
    over the grid axes over which [like] cuts [axes], those of each axis of
    [axes] in turn, major first. Each device holds its window under [g],
    and along [axis] the tile whose number is that of its tiles under [like]
    along [axes], in C order of [axes]. An axis of [axes] that [like] does
    not cut adds no grid axis.

    It is [Error] unless [g] and [like] lay the same devices in one order
    over one mesh, each grid axis of either a run of the mesh's adjacent
    axes, [g] leaves [axis] uncut, and [g] cuts no axis over the grid axes
    it adds. *)

val extents : t -> int array
(** [extents g] is a fresh array of [g]'s extents in normal form, major
    first: [[||]] for one device. *)

val over : t -> int -> int array
(** [over g a] is a fresh array of the grid axes that cut the value's axis
    [a], major first, numbered as {!extents} orders them: [[||]] where [g]
    does not cut [a]. *)

val union : t -> t -> t option
(** [union g g'] is the grid over [g]'s devices that cuts each axis [g] or
    [g'] cuts, giving each device its window of [g] along [g]'s cut axes
    and of [g'] along [g']'s. It is [None] where no grid does: where [g] and
    [g'] list other devices or one order of them, where they cut one axis
    unlike, and where the tiles of their cuts do not vary independently
    over the devices, as two axes cut over one grid axis. *)

val equal : t -> t -> bool
(** [equal g g'] is [true] iff every device holds the same window under [g] and
    [g'], whatever the shape. *)

val pp : (Format.formatter -> int -> unit) -> Format.formatter -> t -> unit
(** [pp pp_device] formats one device as [pp_device] does, a grid of one axis as
    [on [d0; d1]], [split ~axis:a [d0; d1]] or [partial [d0; d1]], and any
    other as [mesh E1xE2 [d0; …]] followed by [~partial:g1,g2] for the grid
    axes it sums over and [~axis:a/g1,g2] for each cut axis and the grid axes
    that cut it. *)
