(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** WAV files. {!Nx_io.Audio} and {!Nx_io.load_audio} document them. A value
    keeps its file's disk buffer, from which {!window} and {!samples} read.
    Functions raise [Failure] without the path. *)

type t

val read : Rig.Buffer.t -> t
(** [read file] is the WAV file of the disk buffer [file].

    Raises [Failure] as {!Nx_io.load_audio} states. *)

val rate : t -> int
val length : t -> int
val channels : t -> int
val bits : t -> int
val channel_mask : t -> int

val samples : t -> Nx.host Nx.packed
(** Raises [Failure] as {!Nx_io.Audio.samples} states. *)

val window : t -> row:int -> rows:int -> Nx.host Nx.packed
(** [window t ~row ~rows] requires the rows to lie in [t]'s.

    Raises [Failure] as {!Nx_io.Audio.window} states. *)

val save :
  ?bits:int ->
  ?channel_mask:int ->
  string ->
  rate:int ->
  ('v, 's, 'd) Nx.t ->
  unit
(** Raises [Invalid_argument] naming [Nx_io.save_wav] and [Failure] as
    {!Nx_io.save_wav} states. *)
