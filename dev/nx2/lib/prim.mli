(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A value's facts: its form, dtype, placement and shape, for every kind of
    value. A sharded value's shape is the whole's: each shard's, times the
    number of tiles along each cut axis. *)

open Value

val form : ('v, 's, 'd) t -> ('v, 's, 'd) form
val dtype : ('v, 's, 'd) t -> ('v, 's) dtype
val placement : ('v, 's, 'd) t -> 'd Devices.placement
val rank : ('v, 's, 'd) t -> int

val dim : ('v, 's, 'd) t -> int -> int
(** Raises [Invalid_argument] if the axis is not below {!rank}. *)

val shape : ('v, 's, 'd) t -> int array
(** [shape x] is a fresh array. [dtype], [placement], [rank] and [dim] of a
    value on one device allocate nothing. *)
