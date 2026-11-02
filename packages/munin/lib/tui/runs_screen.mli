(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Runs, the store's runs as rows under a header per experiment (the Runs
   section of [Munin_tui]).

   A [table] holds the rows, their lines and the columns that fit, built once
   per change of what it shows; [paint] draws it in time linear in the lines on
   screen. *)

type order = Started | Value | State | Name

type view = {
  data : Data.t;
  now : float;
  root : string;
  filter : string;
  metric : string option; (* the focus metric *)
  order : order;
  reverse : bool;
  light : bool;
}

type table

val table : view -> width:int -> typing:bool -> table
(** [table v ~width ~typing] is Runs for [v] at [width], in time linear in the
    store's runs and their columns. *)

val row_ids : table -> string list
(** [row_ids t] is the ids of [t]'s rows, in order. *)

val metrics : Data.t -> Munin.Run.t list -> string list
(** [metrics d rows] is the metrics any of [rows] has, by name. *)

val default_metric : Data.t -> Munin.Run.t list -> string option
(** [default_metric d rows] is the metric with a goal that the most of [rows]
    have, the first by name among equals. *)

val rows : view -> Munin.Run.t list
(** [rows v] is the runs the filter keeps, in [v]'s order, grouped by
    experiment. *)

val pane_request : table -> height:int -> Munin.Run.t -> Data.chart option
(** [pane_request t ~height r] is the chart of [r]'s pane, when it shows, asked
    for as kept. *)

val paint :
  table ->
  Matrix_grid.t ->
  height:int ->
  cursor:string option ->
  marks:(char * string) list ->
  unit

(* Shared with the other screens *)

val name : Munin.Run.t -> string
(** [name r] is [r]'s name, else its id. *)

val value_text : Munin.Value.t -> string
(** [value_text v] is a param's value as the program gave it. *)

val contains : sub:string -> string -> bool
(** [contains ~sub s] is [true] iff [sub], in lowercase, is part of [s],
    ignoring case. *)

val words : string -> string list
(** [words s] is [s]'s words, in lowercase. *)

val tilde : string -> string
(** [tilde path] is [path] with [$HOME] written [~]. *)
