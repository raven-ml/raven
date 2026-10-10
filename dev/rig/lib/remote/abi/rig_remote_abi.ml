(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type area =
  (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

type transfer = { src : int; dst : int; length : int }

(* The largest end of a transfer. *)
let max_end = 1 lsl 60

(* Each end is compared by subtraction: a sum near [max_int] would wrap. *)
let check_transfers ~send ~receive =
  let bad t =
    t.length <= 0 || t.src < 0 || t.dst < 0
    || t.src > max_end - t.length
    || t.dst > max_end - t.length
  in
  if Array.length send = 0 && Array.length receive = 0 then
    Error "the rail carries no transfer"
  else
    match (Array.find_opt bad send, Array.find_opt bad receive) with
    | Some t, _ | None, Some t ->
        Error
          (Printf.sprintf "a transfer of %d bytes from %d to %d is invalid"
             t.length t.src t.dst)
    | None, None -> Ok ()

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
