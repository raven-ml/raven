(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Audio files. {!Nx_io.Audio} and {!Nx_io.load_audio} document them. *)

type t

val read : string -> Rig.Buffer.t -> t
(** [read path file] is the audio file of the disk buffer [file] of [path].

    Raises [Failure] without the path as {!Nx_io.load_audio} states. *)

val rate : t -> int
val length : t -> int
val channels : t -> int
val bits : t -> int
val channel_mask : t -> int

val samples : t -> Nx.host Nx.packed
(** Raises [Failure] starting with the path as {!Nx_io.Audio.samples} states. *)
