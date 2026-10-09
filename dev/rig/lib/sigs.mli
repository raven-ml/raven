(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The signatures of drivers and io devices. *)

(** {!Rig.Driver}. *)
module type Driver = sig
  type t
  type region
  type image
  type capability

  exception Fault of string

  val key : t Type.Id.t
  val arch : t -> string
  val budget : t -> int
  val queues : t -> string list
  val completion : t -> [ `Store | `Object of nativeint | `Host ]
  val waits_on : t -> [ `Store | `Object | `Host ] -> bool
  val max_waits : t -> int
  val blocks : t -> [ `Returns | `May_block ]
  val maps_host : t -> bool
  val capability : t -> capability
  val capability_key : capability Type.Id.t
  val alloc : t -> [ `Device | `Pinned | `Mapped ] -> int -> region option
  val free : t -> region -> unit
  val address : region -> int option
  val handle : region -> nativeint
  val host : region -> int option
  val peer : t -> t -> bool
  val map_peer : t -> t -> region -> region option
  val map_host : t -> int -> int -> region option

  val image :
    t ->
    string ->
    ( [ `Loaded of image | `Place of int * (region -> image * string) ],
      string )
    result

  val entry : image -> string -> int option
  val unload : t -> image -> unit
  val word : t -> region
  val signaled : t -> int
  val sleep : t -> seen:int -> still_ms:int -> unit
  val edge : t -> nativeint
  val stop : t -> unit
end

(** {!Rig.Io}. *)
module type Io = sig
  type t
  type region

  exception Fault of string

  val region_key : region Type.Id.t
  val budget : t -> int
  val alloc : t -> int -> region option
  val free : t -> region -> unit
  val read : t -> region -> at:int -> dst:int -> len:int -> unit
  val write : t -> region -> at:int -> src:int -> len:int -> unit

  val pages :
    t ->
    region ->
    (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
    option

  val prefetch : t -> region -> at:int -> len:int -> unit
  val stop : t -> unit
end
