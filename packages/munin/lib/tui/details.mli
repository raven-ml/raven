(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Details, every fact about one run, from [scroll] lines down. *)

val paint :
  word:string ->
  scroll:int ->
  Munin.Run.t ->
  stats:(string * Munin.History.key_stats) list ->
  Munin.History.t option ->
  marked:(char * Munin.Run.t) list ->
  Matrix_grid.t ->
  width:int ->
  height:int ->
  unit
