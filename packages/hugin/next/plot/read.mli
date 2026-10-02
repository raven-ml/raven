(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Reading a mark's channels to the host as rows. *)

module Ticks := Hugin_next_kit.Ticks

(** {1:context Context} *)

type ctx = {
  theme : Theme.t;
  scales : Resolved.fitted array;
  frozen : Ticks.t array;  (** Per scale. *)
}

(** {1:reading Reading} *)

type reader
(** Reads the channels of one mark, each evaluated once and, when [whole], read
    to the host at most once. *)

val reader : whole:bool -> Figure.mark -> reader

type sel = All | Rows of int array  (** Flat indices of the mark's shape. *)

val rows :
  ?only:string list ->
  ctx ->
  reader ->
  id:Common.id ->
  Coord.projection ->
  axes:Rows.fitted option * Rows.fitted option ->
  warn:(string -> unit) ->
  int option array ->
  sel ->
  Rows.t
(** [rows ~only ctx rd ~id proj ~axes ~warn reads sel] is the rows [sel] of the
    mark of [rd], in a panel whose x and y read the scales [axes], whose binding
    [i] reads the scale [reads.(i)], with the bindings of the roles [only], all
    by default. *)

val swatch :
  ctx ->
  Figure.mark ->
  id:Common.id ->
  Coord.projection ->
  warn:(string -> unit) ->
  scale:int ->
  reads:(int -> bool) ->
  n:int ->
  k:int ->
  float ->
  Rows.t
(** [swatch ctx m ~id proj ~warn ~scale ~reads ~n ~k u] is the one row of the
    swatch of [m] for entry [k] of [n] at the normalised value [u] of the scale
    [scale], which the bindings of the indices [reads] read. A colour role bound
    to other data takes the ink at half its opacity, so that the swatch shows
    only what the legend does. *)

(** {1:ranges Ranges} *)

val colors : ctx -> int -> float -> Hugin_next_gg.Color.t
(** [colors ctx i u] is the colour a colour role gives [u] on the scale [i]. *)

val area : ctx -> int -> float -> float
(** [area ctx i u] is the area in square points that a role of area gives the
    normalised value [u] of the scale [i], [nan] if [u] is. *)

val levels : ctx -> int -> float array option
(** [levels ctx i] is the levels of the scale [i] if it is stepped: the
    normalised ends of its domain and its frozen major ticks between them, in
    increasing order. Its ranges are constant between consecutive levels. *)
