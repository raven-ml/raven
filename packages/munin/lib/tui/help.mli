(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Help, over the screen it opens on: the keys, the keys that screen lacks, and
   the words and glyphs of the screens. *)

val paint :
  screen:[ `Runs | `Run | `Zoom | `Details ] ->
  Matrix_grid.t ->
  width:int ->
  height:int ->
  unit
