(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Agents ({!Rig_remote.listen}, {!Rig_remote.serve}), and what the
    controller's side shares with them: the job's watcher, connecting with a
    bound, and the key's bounds.

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

val watch : Rig_remote_proxy.Link.job -> unit
(** [watch j] starts a thread that fails the process ({!Rig.fail}) with [j]'s
    root cause once [j] fails, and fails [j] with the process's failure
    ({!Rig.failure}) within a second of a device's loss. It ends once [j] failed
    or closed. *)

val dial_tcp : s:float -> string -> int -> (Unix.file_descr, string) result
(** [dial_tcp ~s host port] is a TCP connection to [host] and [port], made
    within [s] seconds. [Error why] if [host] does not resolve, the system
    refuses, or no answer came in time. *)

val join_s : float
(** [join_s] is the most seconds a connection, or the job's other agents'
    connections, take: 10. *)

val check_key : string -> string -> unit
(** [check_key fn key] raises [Invalid_argument] under [Rig_remote.fn] if [key]
    has fewer than 16 or more than 4096 bytes. *)
