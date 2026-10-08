(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The lines a launched process and a machine's half write to their launcher.

    A launched process reports on the descriptor [RIG_REMOTE_REPORT] names:
    rig.remote writes [started], [closed] and [failed WHY]; an agent also writes
    [waiting] and [listening HOST:PORT] before them. A machine's half writes
    [rig-agent VERSION] first on its session, relays its agent's lines, and adds
    [died CAUSE]. Each line ends with a newline; a reason's own newlines are
    written as spaces.

    Used from one thread: readers share one buffer. *)

type t =
  | Agent of string  (** [rig-agent VERSION] *)
  | Started
  | Waiting
  | Listening of string  (** [HOST:PORT] *)
  | Closed
  | Failed of string
  | Died of string  (** ["killed by SIGSEGV"], ["exited with status 3"] *)

val to_string : t -> string
(** [to_string l] is [l]'s line, its newline included. *)

val of_string : string -> t option
(** [of_string s] is the line [s], without its newline, or [None]. *)

(** {1:readers Readers} *)

type reader
(** The type for the lines of a descriptor, read without blocking. *)

val reader : Unix.file_descr -> reader
(** [reader fd] reads the lines of [fd], which it makes non-blocking. *)

val fd : reader -> Unix.file_descr
(** [fd r] is [r]'s descriptor. *)

val read : reader -> string list
(** [read r] is the complete lines [fd r] has for now, without their newlines.
    At the end of [fd r] it closes it, and drops an unfinished line: a process
    that died while writing reported nothing. *)

val ended : reader -> bool
(** [ended r] is [true] once {!read} met the end of [fd r]. *)
