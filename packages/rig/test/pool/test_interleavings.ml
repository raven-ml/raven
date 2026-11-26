(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The suite of rig_pool.h's protocol under interleavings held at its hook
   points, through the scenarios of rig_pool_hooked_probe_stubs.c. *)

open Windtrap
module P = Rig_pool_probe
module H = Rig_pool_hooked_probe

(* [held cores names expected run ()] runs [run] under a watchdog, on a host of
   at least [cores] cores, and checks that its timeline held and that each
   named value is [expected]'s. *)
let held cores names expected run () =
  if P.cores () < cores then
    skip ~reason:(Printf.sprintf "the host has fewer than %d cores" cores) ();
  let timed_out, values = P.finishes "the scenario" run in
  equal ~msg:"the hold that gave up" string "" timed_out;
  equal
    (list (pair string int))
    (List.combine names expected)
    (List.combine names (Array.to_list values))

let scenario names expected run () = held 3 names expected run ()

(* Entering and closing *)

let test_stale_add =
  scenario
    [
      "the next job's count at its close";
      "closed set at that close";
      "the undo began before the next job returned";
      "the count closed after";
      "first job's chunks not run once";
      "next job's chunks not run once";
      "next job's chunks off worker 0";
      "the held worker entered a job";
    ]
    [ 1; 1; 1; 1; 0; 0; 0; 0 ] H.stale_add

let late_entry_names =
  [
    "entered the second job";
    "first job's chunks not run once";
    "second job's chunks not run once";
    "second job's chunks off workers 0 and 1";
    "second job's chunks on the held worker";
  ]

let test_late_entry_thread =
  scenario late_entry_names [ 1; 0; 0; 0; 8 ] (fun () -> H.late_entry 1)

let test_late_entry_other =
  scenario late_entry_names [ 1; 0; 0; 0; 0 ] (fun () -> H.late_entry 2)

let entering_tests =
  group "entering and closing"
    [
      test
        "an add that finds the job closed counts in the next job, which waits \
         for its undo"
        test_stale_add;
      test
        "a worker that saw a job enters the next before its publication and \
         claims its chunks"
        test_late_entry_thread;
      test
        "a worker that saw a job enters the next one, of which it is no \
         thread, and claims nothing"
        test_late_entry_other;
    ]

(* Waiting and waking *)

let test_caller_sleeps =
  scenario [ "the caller set waiting"; "chunks not run once" ] [ 1; 0 ]
    (fun () -> H.caller_sleeps ~held:false)

let test_caller_held =
  scenario
    [
      "the caller set waiting";
      "the worker left while the caller held";
      "chunks not run once";
    ] [ 1; 1; 0 ] (fun () -> H.caller_sleeps ~held:true)

let wake (order : H.wake_order) =
  scenario [ "worker 1 ran a chunk"; "chunks not run once" ] [ 1; 0 ] (fun () ->
      H.wake order)

let test_narrow_burst =
  scenario
    [
      "worker 2 parked during the burst";
      "worker 2 ran a chunk of the wide job";
      "narrow jobs' chunks not run once";
      "wide job's chunks not run once";
    ]
    [ 1; 1; 0; 0 ] H.narrow_burst

let test_narrow_wake =
  scenario
    [
      "worker 1 ran a chunk"; "worker 2 parked again"; "chunks not run once";
    ]
    [ 1; 0; 0 ] H.narrow_wake

(* A job's caller that finds no chunk left, its worker leaving right after its
   close: a blocking job's caller sets waiting at once, a compute job's spins
   and sees the leave first (rig_pool.h's Blocking jobs). *)
let caller_closes ~blocking expected =
  held 2 [ "the caller set waiting"; "chunks not run once" ] [ expected; 0 ]
    (fun () -> H.caller_closes ~blocking)

(* A compute job's waiting thread spins this long before it sleeps ([spin_ns] in
   rig_pool.c). *)
let compute_spin_ns = 100_000

(* A blocking job's worker that finds no chunk parks at once: in one of 7 jobs
   at least, the time from its leave to its decision to park is under half a
   compute worker's spin, which a spin there would take whole in each. *)
let test_worker_parks () =
  if P.cores () < 2 then skip ~reason:"the host has fewer than 2 cores" ();
  let timed_out, values = P.finishes "the scenario" H.worker_parks in
  equal ~msg:"the hold that gave up" string "" timed_out;
  less ~msg:"the least ns from a worker's leave to its park" int
    ~than:(compute_spin_ns / 2) values.(0)

(* The second job's thread finds the pool taken, with none of its chunks run,
   and waits there for the first job's end. *)
let test_take_waits () =
  held 2
    [
      "the second job's thread found the pool taken";
      "the second job's chunks run by then";
      "first job's chunks not run once";
      "second job's chunks not run once";
    ]
    [ 1; 0; 0; 0 ] H.take_waits ()

let waiting_tests =
  group "waiting and waking"
    [
      test
        "a caller that outlasts its spin sleeps, and the last worker out wakes \
         it"
        test_caller_sleeps;
      test
        "a leave that lands between the caller's waiting and its read ends the \
         wait"
        test_caller_held;
      test "a worker that set its parked bit before a publication runs the job"
        (wake Bit_first);
      test
        "a worker that decided to park before a publication, and set its bit \
         after, runs the job"
        (wake Decision_first);
      test "a worker asleep before a publication is woken and runs the job"
        (wake Asleep);
      test "a worker that narrow jobs leave out parks, and a wide job wakes it"
        test_narrow_burst;
      test
        "a job wakes only its own parked workers: one it leaves out sleeps on, \
         sampled for 20 ms"
        test_narrow_wake;
      test
        "a blocking job's caller sleeps as soon as its claims find no chunk: \
         it waits though its worker left right after its close"
        (caller_closes ~blocking:true 1);
      test
        "a compute job's caller spins first: its worker's leave right after \
         its close ends the wait before it sleeps"
        (caller_closes ~blocking:false 0);
      test
        "a blocking job's worker sleeps as soon as it finds no chunk, sampled \
         over 7 jobs"
        test_worker_parks;
      test
        "a compute job of more than one thread waits for another thread's \
         compute job to end"
        test_take_waits;
    ]

(* Fork *)

let forked ?(cores = 3) names expected run () =
  P.needs_fork ();
  held cores names expected run ()

let fork_tests =
  group "fork"
    [
      test
        "a child forked while a worker holds the parking mutex runs a job on \
         every core"
        (forked [ "the child's chunks not run once" ] [ 0 ] H.fork_parked);
      test
        "fork waits for another thread's job, and the child runs a job on \
         every core"
        (forked ~cores:2
           [
             "fork returned after the job's body";
             "the child's chunks not run once";
           ]
           [ 1; 0 ] H.fork_running);
      test
        "a pool whose memory cannot be had runs each job on the caller, and \
         is made at a later job"
        (forked [ "the child's exit status" ] [ 0 ] H.unmade);
    ]

let () =
  exit
    (run "rig_pool.h interleavings"
       [ entering_tests; waiting_tests; fork_tests ])
