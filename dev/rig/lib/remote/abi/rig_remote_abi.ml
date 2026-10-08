(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type area =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

type transfer = { src : int; dst : int; length : int }

type end_ = {
  outbound : area;
  inbound : area;
  counts : area;
  ready : int -> unit;
  ready_fn : nativeint;
  ready_arg : nativeint;
}

type rail = { id : int; local : end_ option; release : unit -> unit }

type host = {
  machine : string;
  rail :
    host option ->
    send:transfer array ->
    receive:transfer array ->
    (rail, string) result;
}

type device = { id : int }
type t = Host of host | Device of device

let key : t Type.Id.t = Type.Id.make ()
