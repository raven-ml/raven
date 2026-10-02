(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The interface marks are made with. *)

module P2 := Hugin_next_gg.P2
module Path := Hugin_next_gg.Path
module Color := Hugin_next_gg.Color
module Text := Hugin_next_text.Text
module Picture := Hugin_next_vg.Picture
module Scale := Hugin_next_kit.Scale

(** {1:bindings Bindings} *)

type binding = Figure.binding

val bind :
  ?imply:'d Scale.t ->
  ?guide:bool ->
  ('d, 'r) Role.t ->
  ('d, 'r) Channel.t ->
  binding

(** {1:rows Rows} *)

type rows = Figure.rows

val id : rows -> Common.id
val length : rows -> int
val shape : rows -> int array
val index : rows -> int array
val get : rows -> ('d, 'r) Role.t -> 'r array option
val normalized : rows -> ('d, 'r) Role.t -> float array option
val range : rows -> ('d, 'r) Role.t -> (float -> 'r) option
val ticks : rows -> ('d, 'r) Role.t -> float array option
val scale : rows -> [ `X | `Y ] -> 'd Scale.kind -> 'd Scale.t option
val positions : rows -> float array * float array
val points : rows -> float array * float array
val extent : rows -> [ `X | `Y ] -> float array * float array
val projection : rows -> Coord.projection
val project : rows -> Path.t -> Path.t
val series : rows -> rows list
val theme : rows -> Theme.t

val text :
  ?halign:Text.Layout.halign ->
  ?valign:Text.Layout.valign ->
  rows ->
  Color.t ->
  P2.t ->
  Text.t ->
  Picture.t

val warn : rows -> string -> unit

(** {1:reducers Reducers} *)

type reducer = Figure.reducer

val m4 : reducer
val cells : reducer
val raster : reducer

(** {1:making Making marks} *)

val broadcast : ?shape:int array -> binding list -> int array

val v :
  name:string ->
  ?reduce:reducer ->
  ?coord:Coord.t ->
  ?shape:int array ->
  ?swatch:(rows -> Picture.t) ->
  binding list ->
  (rows -> Picture.t) ->
  Figure.t
