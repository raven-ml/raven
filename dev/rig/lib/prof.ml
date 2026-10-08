(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type t = {
  counters : string list;
  trace : bool;
  lock : Mutex.t;
  mutable events : (int * Def.event) list;
}

external now : unit -> int = "caml_rig_now"

let profiles : t list Atomic.t = Atomic.make []
let numbers = Atomic.make 0
let active () = Atomic.get profiles
let enabled () = Atomic.get profiles <> []

let rec change f =
  let ps = Atomic.get profiles in
  if not (Atomic.compare_and_set profiles ps (f ps)) then change f

let start ~counters ~trace =
  let p = { counters; trace; lock = Mutex.create (); events = [] } in
  change (fun ps -> ps @ [ p ]);
  p

let stop p = change (List.filter (fun p' -> p' != p))

let add ps e =
  let n = Atomic.fetch_and_add numbers 1 in
  List.iter
    (fun p -> Mutex.protect p.lock (fun () -> p.events <- (n, e) :: p.events))
    ps

let record e = match active () with [] -> () | ps -> add ps e
let add_all ps es = List.iter (add ps) es
