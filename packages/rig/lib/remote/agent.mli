(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Agents ({!Rig_remote.listen}, {!Rig_remote.serve}), and what the
    controller's side shares with them: reports to a launcher, and connecting
    within a bound.

    An agent's listener accepts on a thread of its own and runs each handshake
    on another. [serve] applies the controller's commands from its caller's
    domain, one after another. A [Handover] command's parts run there in
    order, each copy into the controller's memory sending its bytes back as
    it runs, and its [word] message follows once every part has run. *)

type t
(** {!Rig_remote.agent}. *)

val listen : key:string -> string -> int -> (t, string) result
(** {!Rig_remote.listen}. *)

val port : t -> int
(** {!Rig_remote.port}. *)

val serve :
  t ->
  (string * (unit -> (Rig.t list, string) result)) list ->
  (unit, string) result
(** {!Rig_remote.serve}. *)

(** {1:signals Signals}

    The library's calls that build its state run with the calling
    thread's signals blocked ([rig_mask] of [rig_edge.h]), and unblock them
    only across their waits. Its finalisers and exit handler take none,
    even in their waits. A command an agent serves runs as its caller's
    code does: the rig calls it makes block signals themselves. On Windows
    none of these block a signal ({!Rig}'s Errors and failure). *)

val masked : (unit -> 'a) -> 'a
(** [masked f] is [f ()] with the calling thread's signals blocked. *)

val shielded : (unit -> 'a) -> 'a
(** [shielded f] is [f ()] with the calling thread's signals blocked, its
    waits included ([rig_shield] of [rig_edge.h]). *)

val unblocked : ('a -> 'b) -> 'a -> 'b
(** [unblocked f x] is [f x], a wait, with the calling thread's signals as
    they were before the outermost {!masked}. *)

val spawn : ('a -> 'b) -> 'a -> Thread.t
(** [spawn f x] is [Thread.create f x], whose thread blocks every signal but
    a fault's for its life. *)

(** {1:reports Reports to a launcher} *)

type report
(** The type for the descriptor a launcher reads a process's reports on. *)

val report_var : string
(** [report_var] is the variable that names a launched process's report
    descriptor: ["RIG_REMOTE_REPORT"]. *)

val take : string -> string option
(** [take name] is the variable [name]'s value, taken out of the process's
    environment. *)

val report : string -> (report, string) result
(** [report fd] is the report descriptor [fd], a decimal number, which it sets
    close-on-exec. Only this process reports on it. [Error why] if [fd] is not
    an open descriptor. *)

val started : report option -> unit
(** [started r] writes ["started"] on [r], unless it wrote a line before. *)

val ended : report option -> [ `Closed | `Failed of string ] -> unit
(** [ended r e] writes ["closed"] or ["failed WHY"] on [r], unless it wrote an
    end before, and returns once the end is written: a call from another
    thread waits for one in progress. A call from a signal handler, finaliser
    or [Gc.Memprof] callback on the thread that writes on [r] waits for
    nothing. *)

(** {1:dialing Connecting} *)

val dial_tcp : s:float -> string -> int -> (Unix.file_descr, string) result
(** [dial_tcp ~s host port] is a TCP connection to [host] and [port], made
    within [s] seconds. [Error why] if [host] does not resolve, the system
    refuses, or no reply came in time. *)

val join_s : float
(** [join_s] is 10: the seconds a connection may take, and the seconds the
    job's other agents may take to connect to each other. *)
