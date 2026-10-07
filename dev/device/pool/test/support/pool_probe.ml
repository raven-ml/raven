(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Recorded jobs *)

type call = { lo : int64; hi : int64; worker : int; thread : int }

type job = { calls : call list; count : int; overlaps : int }

external record_raw :
  int -> int64 -> int64 -> (int64 * int64 * int * int) array * int * int
  = "probe_record"

let record ~threads ~total ~chunks =
  let calls, count, overlaps = record_raw threads total chunks in
  let call (lo, hi, worker, thread) = { lo; hi; worker; thread } in
  { calls = List.map call (Array.to_list calls); count; overlaps }

external visibility : jobs:int -> threads:int -> int * int = "probe_visibility"

external nested : threads:int -> outer:int -> inner:int -> int * int * int * int
  = "probe_nested"

external balance : int -> bool = "probe_balance"

(* Jobs observed from another domain *)

external reset : unit -> unit = "probe_reset"

external hold : only_worker:bool -> bool = "probe_hold"
external hold_arrived : unit -> int = "probe_hold_arrived"
external hold_release : unit -> unit = "probe_hold_release"

external counted : threads:int -> total:int64 -> chunks:int64 -> unit
  = "probe_counted"

external counted_calls : unit -> int = "probe_counted_calls"

(* The host *)

external cores : unit -> int = "probe_cores"
external performance_cores : unit -> int = "probe_performance_cores"
external cgroup_cpus : string -> int = "probe_cgroup_cpus"

external sysctl : string -> int = "probe_sysctl"

external active_processors : unit -> int = "probe_active_processors"

external pinned_cores : unit -> int * int = "probe_pinned_cores"

let needs_two_cores () =
  if cores () < 2 then Windtrap.skip ~reason:"the host has one core" ()

(* Waiting *)

let timeout = 30.

let settle seconds read ok =
  let deadline = Unix.gettimeofday () +. seconds in
  let rec poll () =
    let v = read () in
    if ok v || Unix.gettimeofday () > deadline then v
    else (
      Unix.sleepf 0.001;
      poll ())
  in
  poll ()

let within seconds why ready =
  if not (settle seconds ready Fun.id) then
    Windtrap.failf "after %gs, %s" seconds why

let finishes what f =
  let result = Atomic.make None in
  let d =
    Domain.spawn (fun () ->
        Atomic.set result
          (Some (match f () with v -> Ok v | exception e -> Error e)))
  in
  within 10. (what ^ " had not returned") (fun () ->
      Option.is_some (Atomic.get result));
  Domain.join d;
  match Option.get (Atomic.get result) with Ok v -> v | Error e -> raise e

let while_held f =
  reset ();
  let held = Domain.spawn (fun () -> hold ~only_worker:false) in
  match
    within 10. "the held job did not begin" (fun () -> hold_arrived () >= 1);
    f ()
  with
  | v ->
      hold_release ();
      Windtrap.equal ~msg:"the held job ran until released" Windtrap.bool true
        (Domain.join held);
      v
  | exception e ->
      hold_release ();
      ignore (Domain.join held);
      raise e
