(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Colours, styles and chart drawing shared by the dashboard's views. *)

(** {1:status Run status} *)

type run_status = Live | Stopped | Done | Failed | Killed
(** The type for run statuses as the dashboard shows them. A running run is
    [Live] while it logs and [Stopped] once it goes quiet. The others are
    the statuses of a run that ended. *)

val status_label : run_status -> string
val status_color : run_status -> Mosaic.Ansi.Color.t

(** {1:smooth Smoothing} *)

type smooth = Off | Light | Medium | Heavy
(** The type for levels of smoothing of a metric chart. *)

val smooth_alpha : smooth -> float option
(** [smooth_alpha s] is the weight of the newest sample in the exponential
    moving average of level [s], or [None] for [Off]. *)

val next_smooth : smooth -> smooth
(** [next_smooth s] is the level after [s], from [Heavy] back to [Off]. *)

val smooth_display : smooth -> int
(** [smooth_display s] is the number shown for [s], from [0] for [Off] to
    [3] for [Heavy]. *)

(** {1:styles Styles} *)

val muted_style : Mosaic.Ansi.Style.t
val axis_style : Mosaic.Ansi.Style.t
val grid_style : Mosaic.Ansi.Style.t

val header_bg : Mosaic.Ansi.Color.t
(** [header_bg] is the background of the header and footer bars. *)

(** {1:charts Charts} *)

val last_value : ('a * 'b) list -> 'b option
(** [last_value history] is the second component of the last pair of
    [history], if any. *)

val draw_metric_chart :
  compact:bool ->
  (int * float) list ->
  Grid.t ->
  width:int ->
  height:int ->
  unit
(** [draw_metric_chart ~compact history grid ~width ~height] draws
    [history], pairs of step and value, as a line chart on the [width] by
    [height] cells of [grid]. A [compact] chart has narrower margins, fewer
    ticks and one decimal on the y axis. An empty [history] draws
    nothing. *)
