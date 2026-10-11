(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  events : (int * Def.event) list option Atomic.t;
      (* [None] is closed. Closing detaches history from pending callbacks. *)
}

(* The host's clock, [rig_now_ns] of [rig_edge.h]. *)
external now : unit -> (int[@untagged])
  = "caml_rig_edge_now_byte" "caml_rig_edge_now"
[@@noalloc]

let profiles : t list Atomic.t = Atomic.make []
let numbers = Atomic.make 0
let active () = Atomic.get profiles
let is_open p = Option.is_some (Atomic.get p.events)

let enabled () =
  match Atomic.get profiles with [] -> false | ps -> List.exists is_open ps

(* Sets the list to [f] of it without the stopped profiles. *)
let rec change f =
  let ps = Atomic.get profiles in
  if not (Atomic.compare_and_set profiles ps (f (List.filter is_open ps))) then
    change f

let start () =
  let p = { events = Atomic.make (Some []) } in
  change (fun ps -> ps @ [ p ]);
  p

(* The exchange is the stop: an exception that ends the call after it, at an
   allocation, leaves [p] in the list stopped, and the next change drops it. *)
let stop p =
  let events = Atomic.exchange p.events None in
  change Fun.id;
  match events with None -> [] | Some events -> events

let rec push a x =
  match Atomic.get a with
  | None -> ()
  | Some l as before ->
      if not (Atomic.compare_and_set a before (Some (x :: l))) then push a x

let add ps e =
  let n = Atomic.fetch_and_add numbers 1 in
  List.iter (fun p -> push p.events (n, e)) ps

let record e = match active () with [] -> () | ps -> add ps e
