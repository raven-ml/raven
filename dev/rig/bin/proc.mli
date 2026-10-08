(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Processes: starting them, learning how they ended, ending them.

    Used from one thread: the launchers run no other. *)

val spawn :
  ?env:string array ->
  string ->
  string array ->
  stdin:Unix.file_descr ->
  stdout:Unix.file_descr ->
  stderr:Unix.file_descr ->
  int
(** [spawn ~env prog args ~stdin ~stdout ~stderr] starts [prog], found through
    [PATH], with [args] and [env] (defaults to this process's), and is its pid.
    Raises [Unix.Unix_error] if it cannot be run. *)

val pipe : unit -> Unix.file_descr * Unix.file_descr
(** [pipe ()] is a pipe whose ends close on exec. *)

val write : Unix.file_descr -> string -> unit
(** [write fd s] writes [s] on [fd], ignoring the system's errors: a reader that
    is gone reads nothing. *)

val fd_number : Unix.file_descr -> int
(** [fd_number fd] is [fd]'s number, as [RIG_REMOTE_REPORT] names it. *)

val inherited : Unix.file_descr -> (unit -> 'a) -> 'a
(** [inherited fd f] is [f ()], with [fd] passed on to the processes [f] starts.
*)

val reap : int -> Unix.process_status option
(** [reap pid] is how [pid] ended, if it has, without waiting. *)

val kill : int -> unit
(** [kill pid] kills [pid] (SIGKILL). [pid] must not have been reaped: a reaped
    pid may name another process. *)

val cause : Unix.process_status -> string
(** [cause st] is ["exited with status N"], or ["killed by SIGNAME"], SIGNAME a
    POSIX signal's name or ["signal N"]. *)

val status : Unix.process_status -> int
(** [status st] is the exit status a shell shows for [st]: its own, or 128 + N
    for signal N. *)

(** {1:signals Signals} *)

val signals : int list -> unit
(** [signals interrupts] handles SIGCHLD, SIGPIPE and each signal of
    [interrupts]: each wakes {!wait}, and the first interrupt is noted. A write
    to a pipe that has no reader fails instead of ending the process. A signal
    ignored when the process started stays ignored, except SIGCHLD: a launcher
    must see its children end. *)

val wait : ?until:float -> Line.reader list -> unit
(** [wait ~until rs] returns once a reader of [rs] has bytes or its end, a
    signal of {!signals} came, or the time is [until] (defaults to never). *)

val interrupted : unit -> int option
(** [interrupted ()] is the first interrupt the process got, at the first call
    after it came, and [None] at every other call. *)

val die_by : int -> 'a
(** [die_by s] restores [s]'s default action and sends it to this process. *)
