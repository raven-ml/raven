(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** TIFF files. {!Nx_io.Tiff}, {!Nx_io.load_tiff} and {!Nx_io.save_tiff}
    document them. An IFD keeps its file's disk buffer, from which
    {!samples} reads. *)

type value =
  | Byte of int array
  | Ascii of string
  | Short of int array
  | Long of int array
  | Rational of (int * int) array
  | Sbyte of int array
  | Undefined of string
  | Sshort of int array
  | Slong of int array
  | Srational of (int * int) array
  | Float of float array
  | Double of float array
  | Long8 of int64 array
  | Slong8 of int64 array

type t
type compression = Uncompressed | Deflate

val read : string -> Rig.Buffer.t -> t list
(** [read path file] is the IFDs of the disk buffer [file] of [path].

    Raises [Failure] without the path as {!Nx_io.load_tiff} states. *)

val tags : t -> (int * value) list

val samples : t -> Nx.host Nx.packed
(** Raises [Failure] starting with the IFD's path as {!Nx_io.Tiff.samples}
    states. *)

val save :
  ?tile:int * int ->
  ?compression:compression ->
  ?tags:(int * value) list ->
  string ->
  ('v, 's, 'd) Nx.t ->
  unit
(** Raises [Invalid_argument] naming [Nx_io.save_tiff] and [Failure] as
    {!Nx_io.save_tiff} states. *)
