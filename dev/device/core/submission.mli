(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Prepared submissions and submit, documented in device_core.mli. A
   submission's slots are set and submitted by one domain at a time. *)

open Def

type t

type work =
  | Words of buffer
  | Fill of {
      fill : nativeint;
      arg : buffer;
      ring_units : int;
      segment_bytes : int;
    }
  | Copy of { src : buffer; dst : buffer }

type part = { queue : string; after : int array; work : work }

val make :
  ?hold:hold ->
  reads:int ->
  writes:int ->
  waits:int ->
  device ->
  part array ->
  t

val read : t -> int -> buffer -> unit
val write : t -> int -> buffer -> unit
val wait_for : t -> int -> int -> unit
val submit : t -> int
