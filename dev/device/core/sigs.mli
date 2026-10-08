(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The signatures of drivers and io devices, documented in device_core.mli. *)

module type Driver = sig
  type t
  type region
  type image
  type part
  type capability

  exception Fault of string

  val key : t Type.Id.t
  val arch : t -> string
  val budget : t -> int
  val queues : t -> string list
  val alloc : t -> [ `Device | `Pinned | `Mapped ] -> int -> region option
  val free : t -> region -> unit
  val address : region -> int option
  val handle : region -> nativeint
  val host : region -> nativeint option
  val peer : t -> t -> bool
  val map_peer : t -> t -> region -> region option
  val map_host : t -> nativeint -> int -> region option
  val unmap : t -> region -> unit

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
  val completion : t -> [ `Store | `Object of nativeint | `Host ]
  val waits_on : t -> [ `Store | `Object | `Host ] -> bool
  val blocks : t -> [ `Returns | `May_block ]

  val part :
    t ->
    queue:string ->
    ?after:int array ->
    [ `Words of int array
    | `Fill of nativeint * nativeint * int * int
    | `Copy of (region * int) * (region * int) * int ] ->
    part

  val room : t -> part array -> [ `Fits | `Later | `Never ]

  val submit :
    t ->
    v:int ->
    waits:([ `Word | `Equal | `Object ] * int * int) array ->
    handles:nativeint array ->
    part array ->
    [ `Ok | `Failed of string ]

  val room_entry : nativeint
  val submit_entry : nativeint
  val self : t -> nativeint
  val capability : t -> capability
  val capability_key : capability Type.Id.t
  val stop : t -> [ `Stopped | `Unknown ]
end

module type Io = sig
  type t
  type region

  exception Fault of string

  val budget : t -> int
  val alloc : t -> int -> region option
  val free : t -> region -> unit
  val read : t -> region -> at:int -> dst:int -> len:int -> unit
  val write : t -> region -> at:int -> src:int -> len:int -> unit
  val stop : t -> unit
end
