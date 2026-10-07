(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What the pool's suites share: probes of nx_pool.h (pool_probe_stubs.c), and
   their waits. *)

(* Recorded jobs *)

type call = { lo : int64; hi : int64; worker : int; thread : int }

(* [calls] in the order they began; [thread] numbers the threads in order of
   their first call, the caller's 0. [count] is every call, [overlaps] the calls
   that began while another of their worker ran. *)
type job = { calls : call list; count : int; overlaps : int }

external record_raw :
  int -> int64 -> int64 -> (int64 * int64 * int * int) array * int * int
  = "probe_record"

let record ~threads ~total ~chunks =
  let calls, count, overlaps = record_raw threads total chunks in
  let call (lo, hi, worker, thread) = { lo; hi; worker; thread } in
  { calls = List.map call (Array.to_list calls); count; overlaps }

(* [visibility ~jobs ~threads] is (values bodies read stale, values the caller
   read stale) over [jobs] jobs. *)
external visibility : jobs:int -> threads:int -> int * int = "probe_visibility"

(* [nested ~threads ~outer ~inner] is (outer units run, inner jobs not run in
   one call, inner calls off their body's thread or not worker 0, inner units
   not run once). *)
external nested : threads:int -> outer:int -> inner:int -> int * int * int * int
  = "probe_nested"

(* [balance chunks] is whether, in a job of [chunks] chunks of one unit on two
   threads, the units after the call that runs unit 0 ran while it lasted. *)
external balance : int -> bool = "probe_balance"

(* Jobs observed from another domain *)

external reset : unit -> unit = "probe_reset"

(* [hold ~only_worker] runs a job of two chunks on two threads whose chunks, or
   with [only_worker] the worker's once both began, wait for [hold_release]. It
   is whether they were released before patience ran out. *)
external hold : only_worker:bool -> bool = "probe_hold"
external hold_arrived : unit -> int = "probe_hold_arrived"
external hold_release : unit -> unit = "probe_hold_release"

external counted : threads:int -> total:int64 -> chunks:int64 -> unit
  = "probe_counted"

external counted_calls : unit -> int = "probe_counted_calls"

(* The host *)

external cores : unit -> int = "probe_cores"
external performance_cores : unit -> int = "probe_performance_cores"

(* The probes of a host fact answer -1 where the host lacks it. *)

(* [sysctl name] is the integer the sysctl [name] reads (macOS). *)
external sysctl : string -> int = "probe_sysctl"

(* [active_processors ()] is the processors active in every group (Windows). *)
external active_processors : unit -> int = "probe_active_processors"

(* [pinned_cores ()] pins the calling thread to one CPU of its affinity, reads
   the cores, restores the affinity and reads them again (Linux). *)
external pinned_cores : unit -> int * int = "probe_pinned_cores"

(* Calls nx_pool_cores only when run: test_pool's affinity child must call
   nothing of the pool before it pins itself. *)
let needs_two_cores () =
  if cores () < 2 then Windtrap.skip ~reason:"the host has one core" ()

(* Waiting *)

(* Each test's limit in seconds, past the 10 s that a probe or a poll waits
   before it fails with what it waited for. *)
let timeout = 30.

(* [settle seconds read ok] reads [read ()] every millisecond until [ok] holds
   of the value or [seconds] have passed, and is the last value read. *)
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

(* Polls [ready] for at most [seconds], and fails with [why] if it never
   holds. *)
let within seconds why ready =
  if not (settle seconds ready Fun.id) then
    Windtrap.failf "after %gs, %s" seconds why

(* [finishes what f] is [f ()], run on a domain of its own, and fails the test
   if it has not returned after 10 s: a job that waits forever fails the test
   instead of hanging it. *)
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

(* [while_held f] is [f ()], run while another domain's job of two threads runs:
   every chunk of it waits until [f] has returned. *)
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
