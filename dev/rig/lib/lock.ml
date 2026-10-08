(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = int

external create : unit -> t = "caml_rig_lock_new"
external try_take : t -> bool = "caml_rig_lock_try" [@@noalloc]
external take : t -> unit = "caml_rig_lock_take"
external give : t -> unit = "caml_rig_lock_give" [@@noalloc]
external wait : t -> unit = "caml_rig_lock_wait"
external broadcast : t -> unit = "caml_rig_lock_broadcast" [@@noalloc]

let protect l f =
  if not (try_take l) then take l;
  match f () with
  | v ->
      give l;
      v
  | exception e ->
      let bt = Printexc.get_raw_backtrace () in
      give l;
      Printexc.raise_with_backtrace e bt

let busy l =
  if try_take l then begin
    give l;
    false
  end
  else true
