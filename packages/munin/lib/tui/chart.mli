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

val latest : prepared -> float option
(** [latest p] is the focus's greatest x. *)

type drawn = {
  log : bool; (* y was drawn on a log scale *)
  range : (float * float) option;
      (* the y domain, when the plot is too short for two ticks *)
}

val draw :
  ?cursor:float ->
  light:bool ->
  prepared ->
  Matrix_grid.t ->
  x:int ->
  y:int ->
  drawn
(** [draw p g ~x ~y] draws in the region of [g] at [(x, y)] the row of
    non-finite marks and, below it, the chart; [cursor] is a dotted vertical
    line. It costs the reduced points, whatever the series' lengths. *)

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

val palette : char -> Ansi.Color.t
(** [palette c] is the colour of the marked run lettered [c]. *)
