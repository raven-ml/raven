(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The suite of nx_pool.h's protocol under interleavings held at its hook
   points, through the scenarios of nx_pool_hooked_probe_stubs.c. *)

open Windtrap
module P = Nx_pool_probe
module H = Nx_pool_hooked_probe

(* [scenario names expected run ()] runs [run] under a watchdog and checks that
   its timeline held and that each named value is [expected]'s. *)
let scenario names expected run () =
  if H.cores () < 3 then skip ~reason:"the host has fewer than three cores" ();
  let timed_out, values = P.finishes "the scenario" run in
  equal ~msg:"the hold that gave up" string "" timed_out;
  equal
    (list (pair string int))
    (List.combine names expected)
    (List.combine names (Array.to_list values))

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
  group ~timeout:P.timeout "entering and closing"
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

let waiting_tests =
  group ~timeout:P.timeout "waiting and waking"
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
    ]

(* Fork *)

let fork_tests =
  group ~timeout:P.timeout "fork"
    [
      test
        "a child forked while a worker holds the parking mutex runs a job on \
         every core"
        (scenario [ "the child's chunks not run once" ] [ 0 ] H.fork_parked);
      test
        "fork waits for another thread's job, and the child runs a job on \
         every core"
        (scenario
           [
             "fork returned after the job's body";
             "the child's chunks not run once";
           ]
           [ 1; 0 ] H.fork_running);
    ]

let () =
  exit
    (run "nx_pool.h interleavings"
       [ entering_tests; waiting_tests; fork_tests ])
