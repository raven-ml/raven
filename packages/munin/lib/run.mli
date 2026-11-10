(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Runs, as a listing reads them.

    A value of type {!t} is a run as it was when read: what its [start]
    record states, which never changes, its tags and notes, and its status
    ({!Store_format.section-status}). Reading one costs the first and last
    records of its log, whatever the log's length, its tag files and notes,
    and a probe of its lock. Everything else the run logged is its
    {!History}.

    {!Store.val-runs} and {!Store.val-run} read runs, and {!History.val-run}
    is a run as its history last read it. A value does no I/O: its accessors
    give what was read. *)

type status =
  [ `Running
    (** A process holds the run's lock. It may be a child the writer
        forked without [exec], which keeps the lock after the writer
        exits ({!Store_format.section-status}), so [`Running] proves a
        holder of the lock, and not that the program that started the run
        still runs. *)
  | `Silent
    (** The reader could not probe the lock
        ({!Store_format.section-status}), so a live writer and a dead one
        look the same. {!updated_at} is when the run last wrote. *)
  | `Crashed
    (** The log has no end and its writer has exited, since no process
        holds the lock or the writer's machine has booted since. *)
  | `Finished  (** The writer ended the run as finished. *)
  | `Failed of string  (** The writer ended the run with this error. *)
  | `Killed  (** The writer ended the run as killed. *) ]
(** The type for statuses ({!Store_format.section-status}). *)

type t = Listing.t
(** The type for runs as read. *)

(** {1:start What the start states} *)

val id : t -> string
(** [id r] is [r]'s id ({!Store_format.section-names}). *)

val dir : t -> string
(** [dir r] is the absolute path of [r]'s directory. *)

val experiment : t -> string
(** [experiment r] is the experiment [r] is an attempt at. *)

val name : t -> string option
(** [name r] is [r]'s name, if it was given one. *)

val parent : t -> string option
(** [parent r] is the id of the run [r] belongs to, if it named one. That
    run may since have been deleted. *)

val params : t -> (string * Value.t) list
(** [params r] is [r]'s params, in the order its start gave them. *)

val steps : t -> int option
(** [steps r] is the largest step [r] declared it will log, if it declared
    one. A reader shows progress as the latest step over it. *)

val provenance : t -> Provenance.t
(** [provenance r] is the provenance [r]'s start recorded. A resume
    records its own ({!History.resumes}). *)

val diff : t -> string option
(** [diff r] is the absolute path of [r]'s [git.diff]
    ({!Store_format.section-files}), if it had one when read. *)

val started_at : t -> float
(** [started_at r] is the time of [r]'s start. *)

(** {1:annotations Tags and notes} *)

val tags : t -> string list
(** [tags r] is [r]'s tags as read, sorted. *)

val notes : t -> string option
(** [notes r] is [r]'s notes as read, if it had any. *)

(** {1:status Status} *)

val status : t -> status
(** [status r] is [r]'s status as read. *)

val ended_at : t -> float option
(** [ended_at r] is the time of [r]'s end, if [r] had ended. *)

val updated_at : t -> float
(** [updated_at r] is when [r] last wrote: its log's modification time
    when read, from the clock of the file system that holds it. A time
    the program gave its samples does not change it. *)
