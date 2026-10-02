(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Reading a mark's channels to the host as rows. *)

module Ticks := Hugin_next_kit.Ticks
module Scale := Hugin_next_kit.Scale

(** {1:context Context} *)

type ctx = {
  theme : Theme.t;
  density : float;
  scales : Resolved.fitted array;
  frozen : Ticks.t array;  (** Per scale. *)
}

type scale_of = int -> int option
(** The index in [scales] of the scale that the binding of an index reads, in a
    panel. *)

(** {1:reading Reading} *)

type reader
(** Reads the tensors of one mark, each source at most once when [whole]. *)

val reader : whole:bool -> Figure.mark -> reader

type sel = All | Rows of int array  (** Flat indices of the mark's shape. *)

val facet : reader -> string -> (int array * (int -> string)) option
(** [facet rd role] is, if the mark binds the facet [role] to data, an identity
    of each row's category, [min_int] where it is missing, and the name of the
    category of an identity. *)

val rows :
  ctx ->
  reader ->
  id:Common.id ->
  Coord.projection ->
  warn:(string -> unit) ->
  scale_of ->
  sel ->
  Rows.t
(** [rows ctx rd ~id proj ~warn scale_of sel] is the rows [sel] of the mark of
    [rd]. *)

val mark : reader -> Figure.mark

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
    [scale], which the bindings of the indices [reads] read. *)

val memo : bool array -> int array -> (int -> 'a) -> int -> 'a
(** [memo miss ids f] is [f], applied at most once per identity of [ids] that
    [miss] does not mark, on such identities. *)

(** {1:ranges Ranges} *)

val colors : ctx -> Resolved.fitted -> float -> Hugin_next_gg.Color.t
(** [colors ctx s u] is the colour a colour role gives [u] on [s]. *)

val stroked : Figure.mark -> bool
(** [stroked m] is [true] iff [m] paints its symbols outlined only. *)
