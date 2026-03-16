(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Incremental run monitoring.

    [Run_monitor] polls a run's event log and maintains aggregated metric state.
    Used by the TUI and future web dashboard for live updates without re-reading
    the entire log. *)

type t
(** The type for run monitors. *)

type live_status =
  [ `Live  (** Events are still arriving. *)
  | `Stopped  (** No events for the timeout period (5 seconds). *)
  | `Done of Run.status  (** [Finished] event received. *) ]
(** The type for live run status. *)

val start : Run.t -> t
(** [start run] opens the run's event log for incremental reading. The event log
    file handle is acquired lazily on the first {!poll}. Must be paired with
    {!close}. *)

val poll : t -> unit
(** [poll t] reads new events since the last poll and updates state. *)

val close : t -> unit
(** [close t] releases the file handle. *)

(** {1:state State} *)

val live_status : t -> live_status
(** [live_status t] is the current run status based on event activity. *)

val metrics : t -> (string * Run.metric) list
(** [metrics t] is the latest metric value per key, sorted alphabetically. *)

val history : t -> string -> (int * float) list
(** [history t key] is the [(step, value)] history for [key] in chronological
    order. Returns the empty list if [key] has no samples. *)

val metric_defs : t -> (string * Run.metric_def) list
(** [metric_defs t] is the metric definitions declared so far, sorted
    alphabetically by key. *)

val best : t -> string -> Run.metric option
(** [best t key] is the best observation for [key] according to
    {!Session.define_metric} goal, or a heuristic if undefined (keys containing
    "loss" or "error" prefer lower values). Returns [None] if no samples exist
    for [key]. *)
