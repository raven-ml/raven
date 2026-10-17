(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Run events and their encoding as lines of JSON.

    A run's [events.jsonl] holds one event per line, each a JSON object
    whose [type] member names its case. A session appends events while
    other processes read the file, so a reader can meet a last line that is
    still being written; {!read_from} leaves it for a later read. A line
    that does not decode as an event is skipped. Timestamps are seconds
    since the Unix epoch. *)

type media_kind = [ `Image | `Audio | `Table | `File ]
(** The type for kinds of media, those of {!Run.media_entry}. *)

type event =
  | Metric of { step : int; timestamp : float; key : string; value : float }
  | Define_metric of {
      key : string;
      summary : Metric.summary;
      step_metric : string option;
      goal : Metric.goal option;
    }
  | Media of {
      step : int;
      timestamp : float;
      key : string;
      kind : media_kind;
      path : string;  (** Relative to the run's directory. *)
    }
  | Summary of (string * Value.t) list
  | Notes of string option
  | Tags of string list
  | Artifact_output of { name : string; version : string }
  | Artifact_input of { name : string; version : string }
  | Resumed of { at : float }
  | Finished of { status : string; ended_at : float }
      (** [status] is ["finished"], ["failed"] or ["killed"]. *)
(** The type for run events. *)

val encode : event -> string
(** [encode e] is [e] as one line of JSON, with no newline. *)

val last : string -> event option
(** [last path] is the event of the last whole line of [path], if that line
    decodes as one and starts within the file's last 4096 bytes. A missing
    [path], and one truncated while it is read, read as [None]. *)

val read_from : string -> pos:int -> event list * int
(** [read_from path ~pos] is the events of the whole lines of [path] after
    byte [pos], in order, and the byte offset after the last whole line, or
    the offset the read started at if there is none. If
    [path] is shorter than [pos], it was truncated or replaced, and the read
    starts at byte [0]. A missing [path] reads as [([], 0)]. *)
