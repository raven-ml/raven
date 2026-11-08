(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The metadata of Arrow IPC ({!Types}) read from and written to FlatBuffers.
    A read raises {!Malformed}. *)

open Types

type bigbytes =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

exception Malformed of string

val fail : ('a, unit, string, 'b) format4 -> 'a
val word : bigbytes -> int -> int
val long : bigbytes -> int -> int
(** [word b p] and [long b p] are little-endian integers at [p]. [long] raises
    {!Malformed} past OCaml's [int], and both raise [Invalid_argument] past
    [b]. *)

val scalars : scalar list
val find_scalar : arrow -> scalar option

val field_text : field -> string

val footer :
  bigbytes -> first:int -> last:int -> schema * int list * int list

val message : bigbytes -> int -> int -> (message * int) option
(** [message b pos limit] is the message at [pos] and the position after it,
    or [None] at the end-of-stream marker or at [limit]. *)



val schema_message : schema -> string

val batch_message :
  dictionary:int option ->
  length:int ->
  nodes:(int * int) list ->
  buffers:(int * int) list ->
  compression:int option ->
  body:int ->
  string
(** [batch_message] is a framed record batch or, for [Some id], a non-delta
    dictionary batch. *)

val footer_bytes :
  schema ->
  dictionaries:(int * int * int) list ->
  batches:(int * int * int) list ->
  string
(** The blocks are [(offset, metadata length, body length)]. *)
