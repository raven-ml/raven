(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The fold of a run's log: the representation of {!History.t}, and what a
    resumed writer learns from the log. {!History} documents each value. *)

type t
type kind = [ `Metric | `Reading ]
type file = { key : string; step : int; time : float; path : string }
type resume = { step : int; time : float; provenance : Provenance.t }

val read : ?limit:int -> ?keep:string list -> Listing.t -> t
(** [keep] names the keys whose samples are kept, every key by default. *)

val follow : ?limit:int -> t -> t
val restarted : t -> bool
val metrics : t -> string list
val readings : t -> string list
val goal : t -> string -> Metric.goal option

val defs : t -> (string * (Metric.goal option * string option)) list
(** [defs h] is each declared metric's latest goal and step metric. *)

val step_metric : t -> string -> string option
val series : t -> string -> Metric.series
val surviving : t -> string -> Metric.series
val orphans : t -> string -> Metric.series
val stats : t -> string -> Metric.stats option
val summary : t -> (string * Value.t) list
val files : t -> file list

val named_files : t -> string list
(** [named_files h] is the path, relative to the run's directory and joined
    by [/], of every [file] record, orphans included. *)

val uses : t -> (string * int) list
val resumes : t -> resume list
val damaged : t -> int
val run : t -> Listing.t
val offset : t -> int
val ended : t -> bool

val hint_keys : t -> (string * Hint.key) list
(** [hint_keys h] is the statistics hint's keys for [h]. *)

val kinds : t -> (string * kind) list
(** [kinds h] is the kind of every key a record named. *)

val accs : t -> (string * (kind * Acc.t)) list

val for_resume : Listing.t -> step:int -> t
(** [for_resume run ~step] is [run]'s history, keeping no samples, with a resume
    at [step] after the log's last line applied. *)
