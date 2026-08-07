(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Declared scalar metrics.

    A metric is a named scalar channel of a run. Declare one with
    {!Session.metric} and write samples through the returned handle: the key is
    spelled once, where the metric is declared. *)

(** {1:types Types} *)

type summary = [ `Min | `Max | `Mean | `Last | `None ]
(** The type for summary modes, i.e. how a run summary value is computed from a
    metric's history. *)

type goal = [ `Minimize | `Maximize ]
(** The type for goals, i.e. whether lower or higher values are better. *)

type sample = {
  step : int;  (** Step counter at which the sample was logged. *)
  timestamp : float;  (** Wall-clock time of the sample. *)
  value : float;  (** Scalar value. *)
}
(** The type for scalar observations. *)

type def = {
  summary : summary;  (** How the run summary value is computed from history. *)
  step_metric : string option;
      (** Key of another metric to use as x-axis (e.g. ["epoch"]). *)
  goal : goal option;  (** Whether lower or higher values are better. *)
}
(** The type for metric declarations, as read back with {!Run.metric_defs}. *)

type t
(** The type for metrics. A metric is bound to the session that declared it. *)

(** {1:writing Writing} *)

val key : t -> string
(** [key m] is the key [m] was declared with. *)

val log : t -> step:int -> ?timestamp:float -> float -> unit
(** [log m ~step v] appends the sample [v] at [step].

    [timestamp] defaults to [Unix.gettimeofday ()]. Silently ignored if the
    declaring session is closed. Use {!Session.log_metrics} to write several
    metrics at one step under a single timestamp. *)

(**/**)

(* Constructed by Session.metric, which supplies the event-log write as
   [append]. Taking the writer as a closure rather than a session is what keeps
   Metric free of a Session dependency, and so free of a cycle. *)

val make :
  key:string -> append:(step:int -> timestamp:float -> float -> unit) -> t

(**/**)
