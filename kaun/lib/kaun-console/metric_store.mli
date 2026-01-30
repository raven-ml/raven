(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Incremental metric aggregation.

    The dashboard should not rescan all historical events every refresh. This
    store maintains the latest value per tag (and latest epoch) as new events
    arrive. *)

type metric = { step : int; epoch : int option; value : float }
type t

val create : ?initial_size:int -> unit -> t
(** [create ()] constructs a new store. *)

val update : t -> Kaun_runlog.Event.t list -> unit
(** [update store events] incorporates newly read events into the store. *)

val latest_epoch : t -> int option
(** [latest_epoch store] returns the maximum epoch observed so far (if any). *)

val latest_metrics : t -> (string * metric) list
(** [latest_metrics store] returns the latest metric per tag, sorted by tag. *)

val clear : t -> unit
(** [clear store] drops all stored metrics/epochs. *)
