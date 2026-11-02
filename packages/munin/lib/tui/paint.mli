(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Text drawn into the dashboard's grid. *)

val text :
  ?style:Ansi.Style.t ->
  Matrix_grid.t ->
  x:int ->
  y:int ->
  w:int ->
  string ->
  unit
(** [text g ~x ~y ~w s] draws [s] cut to [w] cells at [(x, y)], and nothing
    outside the grid. *)

val plain : Ansi.Style.t
val bold : Ansi.Style.t
val dim : Ansi.Style.t
val reverse : Ansi.Style.t

val alert : Ansi.Style.t
(** Red and bold, for what needs attention. *)

val word : string -> Ansi.Style.t
(** [word w] is the style of the state word [w]. *)
