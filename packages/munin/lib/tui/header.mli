(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The dashboard's top bar. *)

val view :
  run_id:string ->
  run_name:string option ->
  tags:string list ->
  latest_epoch:int option ->
  total_epochs:int option ->
  latest_step:int option ->
  elapsed_secs:float ->
  status:Theme.run_status ->
  'a Mosaic.t
(** [view ~run_id ~run_name ~tags ~latest_epoch ~total_epochs ~latest_step
    ~elapsed_secs ~status] is the bar naming the run, by [run_name] with
    [run_id] after it or by [run_id] alone, then its tags, its epoch out of
    [total_epochs] if known, its step, [elapsed_secs] as [HH:MM:SS] and
    [status]. *)
