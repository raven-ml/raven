(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The dashboard's grid of metric charts.

    The charts are laid out one or two to a row and paged in batches of as
    many rows as the screen holds. A chart is selected by its index in the
    current batch. {!update} and {!selected_tag} take the number of charts,
    so the state stays valid as metrics appear. *)

type state = {
  screen_width : int;  (** In cells. *)
  screen_height : int;  (** In cells. *)
  current_batch : int;  (** From [0]. *)
  selected : int;  (** The selected chart's index in the current batch. *)
}
(** The type for the grid's state. *)

type msg =
  | Resize of int * int  (** The new width and height, in cells. *)
  | Next_batch
  | Prev_batch
  | Select_left
  | Select_right
  | Select_up
  | Select_down
(** The type for the grid's messages. *)

val initial_state : unit -> state
(** [initial_state ()] is the first batch of an 80 by 24 screen, with its
    first chart selected. *)

val update : msg -> state -> total_metrics:int -> state
(** [update msg s ~total_metrics] is [s] after [msg], for a grid of
    [total_metrics] charts. A move past the first or last batch, or off the
    edge of the batch, leaves [s] as it was. [Next_batch] and [Prev_batch]
    select the first chart of the batch they move to, and [Resize] brings
    the batch and the selection back within the charts. *)

val selected_tag : state -> total_metrics:int -> all_tags:'a list -> 'a option
(** [selected_tag s ~total_metrics ~all_tags] is the element of [all_tags]
    at the selected chart, if any. *)

val view :
  state ->
  metric_tags:string list ->
  history_for_tag:(string -> (int * float) list) ->
  best_for_tag:(string -> float option) ->
  goal_for_tag:(string -> [< `Maximize | `Minimize ] option) ->
  'a Mosaic.t
(** [view s ~metric_tags ~history_for_tag ~best_for_tag ~goal_for_tag] is the
    current batch of the charts of [metric_tags], under a header for each
    run of tags that share the prefix before their first ['/']. Each title
    gives the last value, and the best value and goal's arrow if known. It
    is a waiting message if [metric_tags] is empty. *)
