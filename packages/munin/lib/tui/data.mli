(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What the dashboard has read of a store, and the reads of a poll.

   A poll is one [read], off the loop, of a [request] the stack of screens
   makes: the listing while Runs is shown, the histories the shown screen
   follows, the statistics of ended runs not read before, and the charts, trends
   and step cursor the shown screen draws, prepared from the histories as
   [Chart.prepare] prepares them. [apply] folds its result in, on the loop, and
   times arrivals at the request's time. The loop then draws from what was
   prepared, at a cost that does not grow with a series.

   A history stays held while a screen of the stack needs its run and the run is
   in the store, and a poll that does not follow it, or fails to read it, leaves
   it as it was. A chart asked for as [kept] outlives the request while its run
   has ended and is in the store, one per run, if its focus's history had read
   the run's end: an ended run's log never changes, so the chart holds and its
   history need not. Values are immutable: a read
   never changes the data the loop holds. *)

type keys =
  | All  (** Every key's samples. *)
  | Keys of string list  (** These keys' samples, none for [[]]. *)

type entry = {
  history : Munin.History.t;
  keys : keys;
  watch : Watch.t;
  rate : float option;
      (* the steps per second, for a history that keeps every key *)
  pauses : (float * float) list; (* Fact.pauses, for the same *)
}

type t

val empty : t
val listing : t -> Munin.Run.t list
val entry : t -> string -> entry option

val run : t -> string -> Munin.Run.t option
(** [run d id] is the run [id] as last read: its history's, else the listing's.
*)

val stats : t -> Munin.Run.t -> (string * Munin.History.key_stats) list option
(** [stats d r] is [r]'s keys' statistics, from its history or its hint, if
    read. *)

val word : t -> now:float -> Munin.Run.t -> string
(** [word d ~now r] is [r]'s state word. *)

(* Requests *)

type chart = {
  focus : string;
  marked : (char * string) list;
  view : Chart.view;
  spec : Chart.spec;
  kept : bool;
      (* kept, once its run has ended, after the screen stops asking for it *)
}

type trend = {
  run : string;
  key : string;
  axis : Chart.axis;
  width : int;
  domain : (float * float) option;
}

type zoom = {
  focus : string;
  others : string list; (* the marked runs *)
  key : string;
  axis : Chart.axis;
  index : int option; (* a sample of the focus's series, else its last *)
}

type request

val request :
  t ->
  time:float ->
  list:bool ->
  follow:(string * keys) list ->
  keep:string list ->
  charts:chart list ->
  trends:trend list ->
  zoom:zoom option ->
  request
(** [request d ~time ~list ~follow ~keep ~charts ~trends ~zoom] is the reads of
    a poll at [time]: the listing when [list], with the statistics of its runs,
    the histories of [follow] with at least the keys given, and what the charts,
    trends and zoom need. The histories of [keep] are held and not read. *)

type result

val read : Munin.Store.t -> request -> result
(** [read store r] does the reads of [r]. A run that cannot be read keeps what
    was held of it and is named in [errors]. *)

val apply : t -> result -> t
val errors : result -> (string * string) list

(* What the loop draws *)

val chart : t -> chart -> Chart.prepared option
(** [chart d c] is the chart [c] as last prepared, else one prepared of the same
    runs and metric for another region or view. *)

val whole : t -> chart -> bool
(** [whole d c] is [true] iff [c] was prepared for its region and view from a
    history of its focus that had read the run's end. *)

val trend : t -> trend -> string option

type cursor = {
  at : zoom;
  x : float option; (* the cursor's x, NaN-free *)
  values : (string * float option) list;
      (* each run's value at or before x, by id *)
}

val cursor : t -> cursor option
