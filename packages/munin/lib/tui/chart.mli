(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Charts of a metric over the focus and the marked runs, as the Charts section
   of [Munin_tui] states, and one-row trends.

   A chart is prepared off the loop ([prepare]), which reads the series and
   reduces them, and drawn on it ([draw]), which lays out the reduced points
   alone. A prepared chart passed back to [prepare] is extended at the cost of
   the samples it did not read. *)

type axis = Step | Step_metric | Time

type view = {
  axis : axis;
  smooth : int; (* 0 off, else the index of the time constant's fraction *)
  log : bool; (* log y asked for *)
  window : (float * float) option;
}
(** The settings every chart of a screen shares. *)

type spec = { key : string; width : int; height : int; x_axis : bool }
(** A chart's metric and region, the row of marks included. *)

val x_of :
  axis -> Munin.History.t -> string -> Munin.Metric.series -> int -> float
(** [x_of axis h key s] is the x of each sample of [s], [key]'s series in [h],
    NaN for a sample without one on [axis]. *)

val bound : float -> float
(** [bound x] is the least of 1, 2 or 5 times a power of ten at or above
    [1.1 *. x], and [1.] for [x <= 0.]. *)

val domain : axis -> key:string -> Munin.History.t list -> float * float
(** [domain axis ~key hs] is the x domain of [key] over the histories [hs], the
    focus's first: to the focus's [steps] on the step axis when it declared them
    and no x passes them, else to the {!bound} of the largest x. *)

type prepared

val prepare :
  ?prev:prepared ->
  view ->
  spec ->
  Munin.History.t ->
  (char * Munin.History.t) list ->
  prepared
(** [prepare view spec focus marked] reduces [spec.key]'s series of [focus] and
    [marked] for the plot of [spec]'s region. With [prev], prepared for the same
    view, spec and runs, it reads only the samples [prev] did not. *)

val spec : prepared -> spec
val lo : prepared -> float
val hi : prepared -> float
val tau : prepared -> float option
val unpaired : prepared -> int

val log_y : prepared -> bool
(** [log_y p] is [true] iff [p] draws y on a log scale: asked for, and every
    value drawn above 0. *)

val latest : prepared -> float option
(** [latest p] is the focus's greatest x. *)

val left : int
(** [left] is the cells left of every plot, for its y labels: a region that
    starts [left] cells before a column has its plot start there. *)

type placed = { px : int; pw : int }
(** The plot's first column in the grid and its width, for what aligns with it.
*)

val draw :
  ?cursor:float ->
  ?inline:bool ->
  ?faint:bool ->
  light:bool ->
  prepared ->
  Matrix_grid.t ->
  x:int ->
  y:int ->
  placed option
(** [draw p g ~x ~y] draws in the region of [g] at [(x, y)] the row of
    non-finite marks and, below it, the chart: the focus solid and last, each
    marked run's level line with its letter, a dot after the latest sample, the
    y axis labelled at both extremes of the values drawn, and axes only when the
    spec asks for an x axis; [cursor] is a dotted vertical line. [inline] draws
    the plot from the region's first row, the marks in its blank cells, and
    [faint] the focus dim. It costs the reduced points, whatever the series'
    lengths, and is [None] for a region too small to hold a plot. *)

type versus = {
  last : float; (* the latest difference *)
  at : float; (* its x *)
  scale : float; (* the largest difference, which the row's edges stand for *)
  zero_row : int; (* the grid row of the zero rule *)
}

val draw_versus :
  light:bool ->
  prepared ->
  char ->
  Matrix_grid.t ->
  under:placed ->
  y:int ->
  rows:int ->
  versus option
(** [draw_versus p c g ~under ~y ~rows] draws, in the plot columns of [under],
    the focus's level minus the marked run [c]'s, per dot column that holds
    both, about a dim zero rule; [None] when no column holds both. *)

type trend

val trend :
  ?prev:trend ->
  ?domain:float * float ->
  axis ->
  width:int ->
  Munin.History.t ->
  string ->
  trend option
(** [trend axis ~width h key] is [key]'s series of [h] drawn in braille one row
    high and [width] cells wide, over its own finite extent or [domain], with
    [×] in a cell that holds a sample that is not finite. On the time axis it
    spans the run's active time. *)

val text : trend -> string
