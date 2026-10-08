(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  counters : string list;
  trace : bool;
  events : (int * Def.event) list Atomic.t;
}

external now : unit -> int = "caml_rig_now"

let profiles : t list Atomic.t = Atomic.make []
let numbers = Atomic.make 0
let active () = Atomic.get profiles
let enabled () = match Atomic.get profiles with [] -> false | _ -> true

let rec change f =
  let ps = Atomic.get profiles in
  if not (Atomic.compare_and_set profiles ps (f ps)) then change f

let start ~counters ~trace =
  let p = { counters; trace; events = Atomic.make [] } in
  change (fun ps -> ps @ [ p ]);
  p

let stop p = change (List.filter (fun p' -> p' != p))

let rec push a x =
  let l = Atomic.get a in
  if not (Atomic.compare_and_set a l (x :: l)) then push a x

let add ps e =
  let n = Atomic.fetch_and_add numbers 1 in
  List.iter (fun p -> push p.events (n, e)) ps

let record e = match active () with [] -> () | ps -> add ps e
let add_all ps es = List.iter (add ps) es
