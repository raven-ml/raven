(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t

external create : unit -> t = "caml_rig_lock_new"
external try_take : t -> bool = "caml_rig_lock_try" [@@noalloc]
external take : t -> unit = "caml_rig_lock_take"
external release : t -> unit = "caml_rig_lock_give" [@@noalloc]
external wait : t -> unit = "caml_rig_lock_wait"
external broadcast : t -> unit = "caml_rig_lock_broadcast" [@@noalloc]

let hold l = if not (try_take l) then take l
let protect l f x =
  hold l;
  match f x with
  | v ->
      release l;
      v
  | exception e ->
      release l;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())

let rec await l f x =
  hold l;
  match f x with
  | Some v ->
      release l;
      v
  | None ->
      wait l;
      await l f x
  | exception e ->
      release l;
      Printexc.raise_with_backtrace e (Printexc.get_raw_backtrace ())
