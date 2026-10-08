(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Agents ({!Rig_remote.listen}, {!Rig_remote.serve}), and what the
    controller's side shares with them: the job's watcher, and connecting with a
    bound.

    An agent's listener accepts on a thread of its own and runs each handshake
    on another, at most 64 at once. [serve] applies the controller's commands
    from its caller's domain, one after another: each hand-over's parts run
    there, in order, before its copies' bytes and its word go back. *)

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
    close-on-exec. Only this process reports on it. [Error why] if [fd] is no
    open descriptor. *)

val started : report option -> unit
(** [started r] writes ["started"] on [r], unless it wrote a line before. *)

val ended : report option -> [ `Closed | `Failed of string ] -> unit
(** [ended r e] writes ["closed"] or ["failed WHY"] on [r], unless it wrote an
    end before. *)

(** {1:fate The job's fate} *)

val watch : report option -> Rig_remote_proxy.Link.job -> unit
(** [watch r j] starts a thread that, once [j] fails, reports the failure on [r]
    and then fails the process ({!Rig.fail}) with [j]'s root cause; and fails
    [j] with the process's failure ({!Rig.failure}) within a second of a
    device's loss. It ends once [j] failed or closed. *)

val dial_tcp : s:float -> string -> int -> (Unix.file_descr, string) result
(** [dial_tcp ~s host port] is a TCP connection to [host] and [port], made
    within [s] seconds. [Error why] if [host] does not resolve, the system
    refuses, or no answer came in time. *)

val join_s : float
(** [join_s] is the most seconds a connection, or the job's other agents'
    connections, take: 10. *)
