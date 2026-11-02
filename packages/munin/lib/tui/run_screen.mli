(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Run and Zoom, one run's monitor (the Run, Charts and States sections of
   [Munin_tui]). The layout is a function of the terminal's size, the metrics
   and the marked runs; [requests] names what a poll prepares for it, and
   [paint] and [paint_zoom] draw from what was prepared, reading no series. *)

type view = {
  data : Data.t;
  now : float;
  focus : Munin.Run.t;
  entry : Data.entry option; (* the focus's, with every key *)
  marked : (char * Munin.Run.t) list;
  metric : string option; (* the metric cursor *)
  chart : Chart.view;
  zoom_at : int option; (* the step cursor, an index of the metric's series *)
  filter : string; (* keeps the metrics whose key holds each word *)
  light : bool;
}

val keep : filter:string -> string list -> string list
(** [keep ~filter keys] is the [keys] that hold each word of [filter], ignoring
    case. *)

val metrics : view -> string list
(** [metrics v] is the focus's metrics with a sample that the filter keeps, by
    name. *)

val cursor : view -> string option
(** [cursor v] is the metric cursor, on a metric the filter keeps. *)

val latest_step : view -> int option

val shown : view -> string list
(** [shown v] is the keys of the chart that draws the marked runs, which they
    keep. *)

val requests :
  view ->
  width:int ->
  height:int ->
  zoom:bool ->
  Data.chart list * Data.trend list * Data.zoom option
(** [requests v ~width ~height ~zoom] is what a poll prepares for Run, or for
    Zoom when [zoom]: the chart, a chart per strip shown, and the machine's
    readings over time. *)

val paint : view -> Matrix_grid.t -> width:int -> height:int -> unit
val paint_zoom : view -> Matrix_grid.t -> width:int -> height:int -> unit
