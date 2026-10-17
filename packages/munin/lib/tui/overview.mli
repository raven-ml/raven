(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The dashboard's panel of a run's parameters and latest metric values,
    and the rows that {!Info} shares. *)

(** {1:rows Rows} *)

val key_style : Mosaic.Ansi.Style.t
val val_style : Mosaic.Ansi.Style.t

val section_header : string -> 'a Mosaic.t
(** [section_header label] is a rule that names a section [label]. *)

val kv_row : string -> string -> 'a Mosaic.t
(** [kv_row k v] is a row with [k] at its left and [v] at its right. *)

val format_value : Munin.Value.t -> string
(** [format_value v] is [v] as text. A float that is an integer below
    [1e9] in magnitude shows no decimals. *)

val format_summary_value :
  metric_defs:(string * Munin.Metric.def) list ->
  best_for_tag:(string -> float option) ->
  string ->
  Munin.Metric.sample ->
  'a Mosaic.t
(** [format_summary_value ~metric_defs ~best_for_tag k m] is the row of
    metric [k]. Its value is [k]'s best if its summary is [`Min] or [`Max]
    and a best is known, and [m]'s value otherwise. A badge names any
    summary but [`Last], and an arrow the goal. *)

val is_sys : string -> bool
(** [is_sys tag] is [true] iff [tag] names a system metric, in the sense
    that it starts with [sys/] and is longer than it. *)

(** {1:panel Panel} *)

val view :
  run:Munin.Run.t ->
  latest_metrics:(string * Munin.Metric.sample) list ->
  step_metrics:string list ->
  metric_defs:(string * Munin.Metric.def) list ->
  best_for_tag:(string -> float option) ->
  'a Mosaic.t
(** [view ~run ~latest_metrics ~step_metrics ~metric_defs ~best_for_tag] is
    [run]'s parameters and the rows of {!format_summary_value} for
    [latest_metrics], leaving out system metrics and [step_metrics]. It is
    an empty box if there is neither. *)
