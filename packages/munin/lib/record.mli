(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The records of a run's log ({!Store_format.section-log}).

    Encoding gives one line, ended by a line feed. Decoding takes a whole line
    without its line feed and is [None] for a damaged one. *)

type start = {
  format : int;
  experiment : string;
  name : string option;
  parent : string option;
  steps : int option;
  params : (string * Value.t) list;
  provenance : Provenance.t;
}

type status = [ `Finished | `Failed of string | `Killed ]

type t =
  | Start of start
  | Metric of {
      key : string;
      goal : Metric.goal option;
      step_metric : string option;
    }
  | Metrics of { step : int; values : (string * float) list }
  | Reading of { step : int; values : (string * float) list }
  | File of { key : string; step : int; path : string }
  | Summary of (string * Value.t) list
  | Use of { name : string; version : int }
  | Resume of { step : int; provenance : Provenance.t }
  | End of status

val encode : ?buf:Buffer.t -> time:float -> t -> string
(** [encode ?buf ~time r] is the line of [r], built in [buf], a scratch buffer
    the caller keeps between calls. *)

val decode : string -> (float * t) option

val start_format : string -> int option
(** [start_format line] is the [format] of a [start] line that is otherwise of a
    format this library cannot read, so that a reader names it. *)

val goal_name : Metric.goal -> string
val goal_of : Json.t -> Metric.goal option
val add_provenance : Buffer.t -> Provenance.t -> unit
val provenance : Json.t -> Provenance.t option
