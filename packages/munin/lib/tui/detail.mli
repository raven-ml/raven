(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The dashboard's screen for one metric. *)

val view :
  tag:string ->
  history_for_tag:(string -> (int * float) list) ->
  best:float option ->
  size:Mosaic.dimension Mosaic.size ->
  smooth:float option ->
  metric_def:Munin.Metric.def option ->
  'a Mosaic.t
(** [view ~tag ~history_for_tag ~best ~size ~smooth ~metric_def] is the
    chart of metric [tag] in a box of [size], over a row of its last, best
    and mean values and sample count, and a row of [metric_def]'s goal,
    summary and step metric. The best value's step is the first at which
    [tag] took the value [best].

    With [smooth] [Some alpha], the chart draws the history in grey under
    its exponential moving average, in which the newest sample weighs
    [alpha]. [alpha] must be in (0, 1]. *)
