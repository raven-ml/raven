(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The metadata of Arrow IPC as OCaml values. A timestamp's zone is empty when
    absent. [compression] is [0] for LZ4_FRAME and [1] for ZSTD. [nodes] are
    lengths and null counts, [buffers] offsets and lengths. *)

open Talon

type arrow =
  | Bool
  | Int of { bits : int; signed : bool }
  | Float of int
  | Utf8
  | Large_utf8
  | Utf8_view
  | Binary
  | Large_binary
  | Binary_view
  | Fixed_binary of int
  | Date_day
  | Date_ms
  | Time of Type.unit_
  | Timestamp of Type.unit_ * string
  | Duration of Type.unit_
  | Decimal of { precision : int; scale : int; bits : int }
  | List
  | Large_list
  | Fixed_list of int
  | Map
  | Struct
  | Unsupported of string

type encoding = { id : int; index_bits : int; index_signed : bool }

type field = {
  name : string;
  arrow : arrow;
  encoding : encoding option;
  children : field list;
  metadata : (string * string) list;
  extension : (string * string) option;
}

type schema = {
  fields : field list;
  metadata : (string * string) list;
  little : bool;
}

type batch = {
  length : int;
  nodes : (int * int) array;
  buffers : (int * int) array;
  compression : int option;
  variadic : int list;
}

type header =
  | Schema of schema
  | Dictionary of { id : int; delta : bool; batch : batch }
  | Batch of batch

type message = { header : header option; body : int; body_len : int }

(** A scalar's Arrow type, talon type and storage dtype. *)
type scalar = Scalar : arrow * 'a Type.t * ('b, 'c) Nx.dtype -> scalar
