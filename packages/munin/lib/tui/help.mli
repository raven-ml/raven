(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Help, over the screen it opens on, dimmed: the keys of every screen, the keys
   of this one, and the words and glyphs of the screens, from [scroll] lines
   down when it does not fit. *)

val paint :
  screen:[ `Runs | `Run | `Zoom | `Details ] ->
  light:bool ->
  scroll:int ->
  Matrix_grid.t ->
  width:int ->
  height:int ->
  unit
