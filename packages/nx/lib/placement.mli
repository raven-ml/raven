(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Placements: the implementation of {!Nx.Placement}.

    A placement is a grid of devices ({!Grid}). Besides {!Nx.Placement}'s
    functions, siblings read its cuts, rewrite its axes and compare its
    devices and memories. Every function below keeps a grid in normal form. *)

type t
(** {!Nx.Placement.t}. *)

val host : t
val on : Device.t -> t
val replicated : Device.t list -> t
val sharded : axis:int -> Device.t list -> t
val devices : t -> Device.t list
val window : t -> int array -> Device.t -> (int * int) array
val with_leading_axis : t -> t
val without_leading_axis : t -> t
val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit

(** {1:grid Grid} *)

val cuts : t -> (int * int) list
(** [cuts p] is {!Grid.cuts}[ p]. *)

val uncut : t -> axis:int -> t
(** [uncut p ~axis] is {!Grid.uncut}[ p ~axis]. *)

val map_axes : (int -> int) -> t -> t
(** [map_axes f p] is {!Grid.map_axes}[ f p]. *)

val select : t -> axis:int -> int -> t
(** [select p ~axis j] is {!Grid.select}[ p ~axis j]. *)

val check_shape : string -> t -> int array -> unit
(** [check_shape what p shape] raises [Invalid_argument], prefixed with
    [what], if [p] cuts an axis [shape] does not have or does not divide
    evenly over its tiles. *)

(** {1:queries Queries} *)

val is_host : t -> bool
(** [is_host p] is [true] iff [p] is the one device {!Device.host}. A test
    device or the host's memory with another backend is not. *)

val on_disk : t -> bool
(** [on_disk p] is [true] iff [p] is one device whose memory computes nothing
    ({!Rig.computes}), such as the disk. *)

val same_devices : t -> t -> bool
(** [same_devices p q] is [true] iff [p] and [q] have the same devices, in
    any order and layout. *)

val same_memories : t -> t -> bool
(** [same_memories p q] is [true] iff every window of a value lies in the
    same memory at [p] and at [q], whichever backends compute on it. *)
