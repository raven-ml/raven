(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Incremental metric aggregation.

    The dashboard should not rescan all historical events every refresh. This
    store maintains the latest value per tag (and latest epoch) as new events
    arrive. It also stores full history for rendering charts and tracks best
    values for each metric. *)

type metric = { step : int; epoch : int option; value : float }
type best_value = { step : int; value : float }
type t

val create : ?initial_size:int -> unit -> t
(** [create ()] constructs a new store. *)

val update : t -> Kaun_runlog.Event.t list -> unit
(** [update store events] incorporates newly read events into the store. *)

val latest_epoch : t -> int option
(** [latest_epoch store] returns the maximum epoch observed so far (if any). *)

val latest_metrics : t -> (string * metric) list
(** [latest_metrics store] returns the latest metric per tag, sorted by tag. *)

val history_for_tag : t -> string -> (int * float) list
(** [history_for_tag store tag] returns the full history of (step, value) pairs
    for a given tag, in chronological order. Returns empty list if tag not
    found. *)

val best_for_tag : t -> string -> best_value option
(** [best_for_tag store tag] returns the best value for the given tag. For
    loss-like metrics (containing "loss" or "err"), returns the minimum. For
    other metrics, returns the maximum. *)

val best_metrics : t -> (string * best_value) list
(** [best_metrics store] returns the best value for each metric, sorted by tag.
    Uses heuristics to determine if min or max is "best" based on metric name.
*)

val clear : t -> unit
(** [clear store] drops all stored metrics/epochs. *)
