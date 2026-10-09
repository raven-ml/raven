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

val window : t -> int array -> int -> (Nx_array.Move.range array, string) result
(** [window g shape i] is the window of a value of [shape] that the [i]th device
    of [devices g] holds: the whole shape on an axis no cut names. It is [Error]
    if a cut axis is not an axis of [shape] or its tiles do not divide it
    evenly.

    Raises [Invalid_argument] if [i] is not a position of [devices g]. *)

val map_axes : (int -> int) -> t -> t
(** [map_axes f g] cuts the value's axis [f a] where [g] cuts [a], over the same
    grid axes: [g] for a value whose axes [f] renumbers. [f] is injective on
    [g]'s cut axes; nothing checks it. *)

val select : t -> axis:int -> int -> t
(** [select g ~axis j] is the grid of [g]'s devices that hold tile [j] of the
    cut [axis]: the grid axes of that cut go. [g] itself if [axis] is not cut.

    Raises [Invalid_argument] if [j] is not a tile of [axis]. *)

val uncut : t -> axis:int -> t
(** [uncut g ~axis] is [g] with the value's [axis] whole on every device: the
    grid axes that cut it hold copies. [g] itself if [axis] is not cut. *)

val equal : t -> t -> bool
(** [equal g g'] is [true] iff every device holds the same window under [g] and
    [g'], whatever the shape. *)

val pp : (Format.formatter -> int -> unit) -> Format.formatter -> t -> unit
(** [pp pp_device] formats one device as [pp_device] does, a grid of one axis as
    [on [d0; d1]] or [split ~axis:a [d0; d1]], and any other as
    [mesh E1xE2 [d0; …]] followed by [~axis:a/g1,g2] for each cut axis and the
    grid axes that cut it. *)
