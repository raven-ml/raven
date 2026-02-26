(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Incremental metric aggregation.

    A store maintains per-tag latest values, full history, and best values as
    new {!Event}s arrive. This avoids rescanning all historical events on each
    dashboard refresh. *)

(** {1:types Types} *)

type metric = {
  step : int;  (** Training step. *)
  epoch : int option;  (** Training epoch, if any. *)
  value : float;  (** Metric value. *)
}
(** The type for the latest observation of a metric. *)

type best_value = {
  step : int;  (** Step at which the best value was observed. *)
  value : float;  (** The best value. *)
}
(** The type for the best observed value of a metric. *)

type t
(** The type for metric stores. *)

(** {1:constructors Constructors} *)

val create : ?initial_size:int -> unit -> t
(** [create ()] is an empty store.

    [initial_size] is a hint for the number of distinct tags expected. Defaults
    to [32]. *)

(** {1:updating Updating} *)

val update : t -> Event.t list -> unit
(** [update store evs] incorporates [evs] into [store]. *)

val clear : t -> unit
(** [clear store] drops all stored metrics and resets the epoch counter. *)

(** {1:querying Querying} *)

val latest_epoch : t -> int option
(** [latest_epoch store] is the maximum epoch observed so far, if any. *)

val latest_metrics : t -> (string * metric) list
(** [latest_metrics store] is the latest {!metric} per tag, sorted by tag name.
*)

val history_for_tag : t -> string -> (int * float) list
(** [history_for_tag store tag] is the [(step, value)] history for [tag] in
    chronological order, or the empty list if [tag] has not been observed. *)

val best_for_tag : t -> string -> best_value option
(** [best_for_tag store tag] is the best value for [tag], if any.

    {b Note.} "Best" is minimum for tags containing ["loss"] or ["error"],
    maximum otherwise. *)

val best_metrics : t -> (string * best_value) list
(** [best_metrics store] is the best value per tag, sorted by tag name. Uses the
    same heuristic as {!best_for_tag}. *)
