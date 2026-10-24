(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

external hold : unit -> unit = "caml_rig_edge_hold" [@@noalloc]
external unhold : unit -> unit = "caml_rig_edge_unhold" [@@noalloc]
external reraise : exn -> 'a = "%reraise"

type t = Mutex.t

let create = Mutex.create

(* [take m] takes [m], counted from before the take, so that a signal handler
   that runs inside a contended take sees it held. An exception that ends the
   take gives the count back first, having taken nothing: one a handler
   raised, or the error-checking [Mutex]'s [Sys_error] for a take by the
   thread that holds [m], which only a call that skipped its check makes and
   which raises [Invalid_argument]. *)
let take m =
  hold ();
  match Mutex.lock m with
  | () -> ()
  | exception e -> (
      unhold ();
      match e with
      | Sys_error why when String.starts_with ~prefix:"Mutex.lock:" why ->
          invalid_arg
            "Rig_pci: a lock taken again by the thread that holds it, from a \
             signal handler, finaliser or Gc.Memprof callback"
      | e -> reraise e)

(* [try_take m] takes [m] as [take] does if it is free, and is whether it
   took it. *)
let try_take m =
  hold ();
  match Mutex.try_lock m with
  | true -> true
  | false ->
      unhold ();
      false
  | exception e ->
      unhold ();
      reraise e

(* Gives [m] back, then its count, with nothing between that allocates or
   polls. *)
let give m =
  Mutex.unlock m;
  unhold ()

(* [protect m f] is [f ()] holding [m], taken as the last step before the
   handler and given back as the first step of both exits, where nothing
   allocates or polls, so an exception anywhere in [f], an asynchronous one
   too, leaves [m] free and the count right. *)
let protect m f =
  take m;
  match f () with
  | x ->
      give m;
      x
  | exception e ->
      give m;
      reraise e

