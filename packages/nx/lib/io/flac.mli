(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** FLAC files. {!Nx_io.Audio} and {!Nx_io.load_audio} document them. A value
    keeps its file's disk buffer, from which {!samples} decodes. Functions raise
    [Failure] without the path. *)

type t

val read : Rig.Buffer.t -> t
(** [read file] is the FLAC file of the disk buffer [file], whose first 4 bytes
    are its marker.

    Raises [Failure] as {!Nx_io.load_audio} states. *)

val rate : t -> int
val length : t -> int
val channels : t -> int
val bits : t -> int
val channel_mask : t -> int

val samples : t -> Nx.host Nx.packed
(** Raises [Failure] as {!Nx_io.Audio.samples} states. *)
