(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The dashboard's state, its transitions, what it wants read and what the
    screen shows.

    The state holds only sources: what the user chose, what the terminal and
    the screen's elements reported, and what the last poll read. Everything
    else, the pane's lines, the screen's shape, the charts, is computed from
    it where it is read. *)

(** {1:state State} *)

type slot =
  | Large  (** The large chart, at charts and zoom. *)
  | Small of int  (** The small chart of this index. *)
  | Strip  (** Every strip. *)
  | Trend  (** Every trend of the runs pane. *)
  | Rows  (** The runs pane's lines. *)
  | Strip_rows  (** The strips' lines. *)
  | Centre  (** The centre's content, inside its padding. *)
(** The type for the screen's elements whose size the model reads. *)

type at = Latest | Step of int
(** The type for zoom's step cursor: the cursor's run's latest step, or a
    step. *)

type depth =
  | Charts
  | Strips of { strip : string option }
      (** The strip under the cursor; [None]: the chart's metric. *)
  | Zoom of { metric : string; at : at; narrow : int }
      (** [narrow] halvings of the x domain about [at] ({!Plot.window}). *)
(** The type for the centre's depths. *)

type layer =
  | Screen
  | Filtering of Mosaic.Editor.t  (** The filter as it is typed. *)
  | Info of Mosaic.Scroll.t  (** Everything about the run, and its scroll. *)
  | Help
(** The type for what is over the screen. *)

type pane = Auto | Shown | Hidden
(** The type for a side pane's toggle: [Auto] follows the terminal's
    width. *)

type cursor =
  | Unchosen  (** Before the first listing: the newest run. *)
  | Waiting  (** Waiting for a run without an end that has not crashed. *)
  | On of string  (** The run of this id. *)
(** The type for the run the user chose. *)

type poll = Idle | Reading | Again
(** The type for the poll in flight: none, one, or one and another after
    it. *)

(** Overlays: four slots, [a] to [d], each run in one at most. *)
module Overlays : sig
  type t

  val empty : t
  (** [empty] has no overlay. *)

  val toggle : string -> t -> t option
  (** [toggle id o] is [o] with [id] taken off, or put in the first free
      slot; [None] when every slot is taken. *)

  val runs : t -> (char * string) list
  (** [runs o] is each overlay with its letter, by letter. *)
end

module Smap : Map.S with type key = string

type t = {
  store : Munin.Store.t;
  clock : unit -> float;
  size : (int * int) option;  (** The terminal's columns and lines. *)
  slots : (slot * (int * int)) list;  (** The sizes the elements reported. *)
  look : Theme.look;
  data : Data.t;
  poll : poll;
  cursor : cursor;
  metric : string option;  (** The chosen tab. *)
  depth : depth;
  layer : layer;
  overlays : Overlays.t;
  colours : int Smap.t;  (** Each run drawn so far, its colour. *)
  others : bool;  (** The experiment's other runs in grey. *)
  smooth : bool;
  scales : Runs.Sset.t;  (** The metrics whose y scale [y] turned over. *)
  filter : Runs.filter;
  order : Runs.order;
  folds : Runs.folds;
  runs_pane : pane;
  run_pane : pane;
  top : int;  (** The first line the runs pane shows. *)
  strips_top : int;  (** The first line the strips show. *)
  notice : string option;  (** Why the last key did nothing. *)
}
(** The type for the dashboard's state. *)

val make :
  store:Munin.Store.t -> clock:(unit -> float) -> scheme:[ `Dark | `Light ] ->
  cursor:cursor -> t
(** [make ~store ~clock ~scheme ~cursor] is the state before any report. *)

(** {1:transitions Transitions} *)

type action =
  | Vertical of int  (** The run at charts; the strip at strips. *)
  | Horizontal of int  (** The metric at charts; the step at zoom. *)
  | Run of int  (** The run, at any depth. *)
  | First
  | Last
  | Deeper
  | Back
  | Overlay
  | Others
  | Clear
  | Smooth
  | Scale
  | Narrow
  | Widen
  | Filter
  | Keep_filter
  | Clear_filter
  | Order
  | Fold
  | Info
  | Toggle_runs
  | Toggle_run
  | Help
  | Close
  | Quit
(** The type for what a key does. [Quit] is the program's. *)

val act : action -> t -> t
(** [act a m] is [m] after [a]. An action whose object is missing leaves
    [m], with a notice when the user may not see why. *)

val polled : Data.t -> t -> t
(** [polled d m] is [m] with the poll's [d], the cursor and the folds
    decided on the first listing, and the runs pane kept on its cursor. *)

val resized : int -> int -> t -> t
(** [resized w h m] is [m] on a terminal of [w] columns and [h] lines. *)

val scrolled_rows : Mosaic.Scroll.t -> t -> t
(** [scrolled_rows s m] is [m] with the runs pane scrolled to [s]. *)

val scrolled_strips : Mosaic.Scroll.t -> t -> t
(** [scrolled_strips s m] is [m] with the strips scrolled to [s]. *)

val sized : slot -> int -> int -> t -> t
(** [sized s w h m] is [m] with [s] of [w] by [h] cells. *)

val edited : Mosaic.Editor.t -> t -> t
(** [edited e m] is [m] with the filter as typed in [e]. *)

val scrolled_info : Mosaic.Scroll.t -> t -> t
(** [scrolled_info s m] is [m] with the info's scroll [s]. *)

val palette : Theme.look -> t -> t
(** [palette look m] is [m] in [look]. *)

val styled : Mosaic.Ansi.Styling.t -> t -> t
(** [styled s m] is [m] with the screen written at the styling [s]. *)

(** {1:messages Messages} *)

type msg =
  | Key of action
  | Tick  (** The poll's timer. *)
  | Polled of Data.t
  | Resized of int * int
  | Sized of slot * int * int
  | Edited of Mosaic.Editor.t
  | Submitted
  | Info_scrolled of Mosaic.Scroll.t
  | Rows_scrolled of Mosaic.Scroll.t
  | Strips_scrolled of Mosaic.Scroll.t
  | Palette of Mosaic.Ansi.Palette.t
  | Styling of Mosaic.Ansi.Styling.t
(** The type for the program's messages. *)

(** {1:derived What the screen shows} *)

type side = Beside of int | Over of int | Off
(** The type for where the run pane is: beside the centre with a width, over
    it, or hidden. *)

type hero = Big of int | Line
(** The type for the hero: big digits with this many facts, or one line. *)

type panes = {
  runs : int option;  (** The runs pane's width. *)
  run : side;
  nav : bool;  (** A line under the top bar names the cursor's run. *)
  hero : hero;
  smalls : int;  (** The small charts under the large one. *)
}

type frame =
  | Unsized  (** No size reported yet. *)
  | Tiny  (** Too small: one line per live run. *)
  | Reading  (** No listing yet. *)
  | Empty  (** A store without runs. *)
  | Waiting_run  (** Waiting for a run. *)
  | Panes of panes
(** The type for the screen's shape. *)

val frame : t -> frame
(** [frame m] is the screen's shape at [m]'s size, depth and toggles. *)

val kept : t -> Facts.run list
(** [kept m] is the runs the filter keeps, in order. *)

val lines : t -> Runs.line list
(** [lines m] is the runs pane's lines. *)

val focus : t -> Facts.run option
(** [focus m] is the cursor's run: the chosen run if it is kept, else the
    first stop. *)

val key : t -> string -> string option
(** [key m e] is the key metric of the experiment [e]. *)

val metric : t -> string option
(** [metric m] is the chart's metric: the chosen tab if the focus has it,
    else its experiment's key metric, else its first metric. *)

val metrics : t -> string list
(** [metrics m] is the focus's metrics, the tabs. *)

val strips : t -> (string * Data.axis) list
(** [strips m] is the strips at the strips depth: each metric, then, while
    the focus is live, its machine's readings. *)

val strip : t -> string option
(** [strip m] is the strip under the cursor at strips: the one chosen if
    the focus has it, else the chart's metric. *)

val colour : t -> string -> int option
(** [colour m id] is the run's colour, once it was drawn. *)

val drawn : t -> (Plot.role * string) list
(** [drawn m] is the runs the charts draw, in drawing order: the other runs
    of the experiment with [r], the overlays, the focus. *)

val suffixes : t -> string Smap.t
(** [suffixes m] is, by id, the end of the id of each run whose name another
    run listed or drawn shares: the fewest of its last characters, four at
    the least, that tell the runs of that name apart. *)

val label : t -> Facts.run -> string
(** [label m f] is [f]'s name, followed by its {!suffixes} entry when it has
    one. *)

val label_in : t -> Facts.run -> int -> string
(** [label_in m f w] is [label m f] in at most [w] cells: the name is cut,
    ending in […], before its suffix is. *)

type chart = {
  slot : slot;
  key : string;
  axis : Data.axis;
  x : float * float;  (** The x domain shown. *)
  runs : (Plot.role * string) list;  (** In drawing order. *)
  smooth : bool;
  log : bool;
  size : (int * int) option;  (** The slot's size, once reported. *)
}
(** The type for a chart of the screen. *)

val charts : t -> chart list
(** [charts m] is the charts the centre draws at its depth. *)

val trend : t -> Facts.run -> chart option
(** [trend m f] is the run's trend in the runs pane: its experiment's key
    metric over its own steps, in its colour while the charts draw the
    run. *)

val trends : t -> (string * chart) list
(** [trends m] is the trend of each run of the lines the runs pane shows,
    by run. *)

val plot : t -> chart -> Plot.t
(** [plot m c] is what [c] draws: each run's reductions of [c]'s series
    over its x domain, at the dot counts a poll made them; the draw keeps
    the one made at the size it draws. *)

val strip_height : int
(** [strip_height] is a strip's lines. *)

val strip_gap : int
(** [strip_gap] is the lines between two strips. *)

val zoom_step : t -> int option
(** [zoom_step m] is the step under zoom's cursor. *)

val zoom_window : t -> (float * float) option
(** [zoom_window m] is the steps zoom shows. *)

val legend : t -> Data.readout option
(** [legend m] is the values the legend prints at charts, or the readout at
    zoom: each run drawn but the grey ones, at the focus's latest step or at
    zoom's cursor. *)

val hero : t -> Data.readout list
(** [hero m] is the smoothed values the hero prints. *)

val readouts : t -> Data.readout list
(** [readouts m] is {!legend} and {!hero}. *)

val compare : t -> Data.compare option
(** [compare m] is the strips' differences to the first overlay. *)

val wants : t -> Data.wants
(** [wants m] is what the screen draws, for the poll. *)
