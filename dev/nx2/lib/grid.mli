(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Device grids: where each device's window of a value lies.

    A grid is one device, or distinct devices in row-major order over the grid's
    extents, with, for each cut axis of a value, the grid axes that cut it,
    major first. A grid axis that no cut names holds copies. Devices are
    positions in a set. A grid is kept in normal form: extents are at least 2,
    adjacent grid axes merge when no cut names either or one cut names both in
    order, and a grid of one device is that device. So two grids that give each
    device the same window are equal. *)

type t

val device : int -> t
(** [device k] is the device [k] alone. *)

val v :
  devices:int array ->
  extents:int array ->
  cuts:(int * int array) array ->
  (t, string) result
(** [v ~devices ~extents ~cuts] is the grid over [devices], in row-major order
    over [extents], each [(axis, over)] of [cuts] cutting the value's [axis]
    over the grid axes [over], major first. It is [Error] unless the extents are
    positive and multiply to the number of devices, the devices are distinct and
    not negative, and the cut axes and the grid axes they name are distinct and
    in range. *)

val devices : t -> int array
(** [devices g] is a fresh array of [g]'s devices in row-major order. *)

val count : t -> int
(** [count g] is the number of [g]'s devices. *)

val one : t -> int option
(** [one g] is [Some k] iff [g] is the device [k] alone. *)

val cuts : t -> (int * int) array
(** [cuts g] is each cut axis with its number of tiles, by increasing axis. *)

val is_cut : t -> bool
(** [is_cut g] is [cuts g <> [||]]. It allocates nothing. *)

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
    grid axes that cut it hold copies. [g] itself if [axis] is not cut. *)

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

val equal : t -> t -> bool
(** [equal g g'] is [true] iff every device holds the same window under [g] and
    [g'], whatever the shape. *)

val pp : (Format.formatter -> int -> unit) -> Format.formatter -> t -> unit
(** [pp pp_device] formats one device as [pp_device] does, a grid of one axis as
    [on [d0; d1]] or [split ~axis:a [d0; d1]], and any other as
    [mesh E1xE2 [d0; …]] followed by [~axis:a/g1,g2] for each cut axis and the
    grid axes that cut it. *)
