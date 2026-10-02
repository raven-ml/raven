(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Device grids: where each device's window of a value lies.

    A grid is one device, or distinct devices in row-major order over the grid's
    extents, and for each cut tensor axis the grid axes that cut it, major
    first. A grid axis that no cut names holds copies. A grid is kept in normal
    form: extents are at least 2, adjacent grid axes merge when no cut names
    either or one cut names both in order, and a grid of one device is that
    device. It is abstract over the device type and taken apart by the window
    each device holds. *)

type 'd t
(** The type for grids over devices of type ['d]. *)

val device : 'd -> 'd t
(** [device d] is [d] alone. *)

val v : 'd list -> int list -> (int * int list) list -> 'd t
(** [v devices extents cuts] is the grid over [devices], in row-major order over
    [extents], each [(axis, over)] of [cuts] cutting tensor [axis] over the grid
    axes [over], major first, in normal form.

    Raises [Invalid_argument] unless the extents multiply to the number of
    devices, and the cut axes and the grid axes they name are distinct and in
    range. *)

val devices : 'd t -> 'd list
(** [devices p] is [p]'s devices, in row-major order. *)

val cuts : 'd t -> (int * int) list
(** [cuts p] is each cut tensor axis of [p] with the number of tiles it is cut
    into, by increasing axis. *)

val tile_index : 'd t -> int -> (int * int) list
(** [tile_index p k] is, for each cut tensor axis of [p], the index of the tile
    that the device at position [k] holds. *)

val map_axes : (int -> int) -> 'd t -> 'd t
(** [map_axes f p] cuts tensor axis [f a] where [p] cuts [a]. *)

val select : 'd t -> axis:int -> int -> 'd t
(** [select p ~axis j] is the grid of the devices of [p] that hold tile [j] of
    the cut tensor [axis]: the grid axes of that cut go. *)

val uncut : 'd t -> axis:int -> 'd t
(** [uncut p ~axis] is [p] with tensor [axis] whole on every device: the grid
    axes that cut it hold copies. *)

val equal : ('d -> 'd -> bool) -> 'd t -> 'd t -> bool
(** [equal eq p q] is [true] iff every device, compared with [eq], holds the
    same window of any value under [p] and [q]. *)

val pp : (Format.formatter -> 'd -> unit) -> Format.formatter -> 'd t -> unit
(** [pp pp_device] formats a grid of one axis as the placement constructor that
    builds it. A grid of several axes, which no placement constructor builds,
    prints as [grid E1xE2 [devices]] followed by [~axis:a/g1,g2] for each cut
    tensor axis [a] and the grid axes that cut it. *)
