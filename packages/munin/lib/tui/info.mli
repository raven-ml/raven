(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The dashboard's screen of a run's details. *)

val view :
  run:Munin.Run.t ->
  status:Theme.run_status ->
  elapsed_secs:float ->
  metric_defs:(string * Munin.Metric.def) list ->
  latest_metrics:(string * Munin.Metric.sample) list ->
  step_metrics:string list ->
  best_for_tag:(string -> float option) ->
  'a Mosaic.t
(** [view ~run ~status ~elapsed_secs ~metric_defs ~latest_metrics
    ~step_metrics ~best_for_tag] is [run]'s name, id, experiment, group,
    [status], start time in local time and [elapsed_secs] as its duration,
    then its tags, provenance, parameters, metric summary as
    {!Overview.view} shows it, and notes. *)
