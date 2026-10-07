(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The suite of nx_pool.h's threads, through the probes of thread_probe_stubs.c:
   what a body sees on a worker, and the workers' life. *)

open Windtrap
module P = Pool_probe
module T = Thread_probe

let cores = P.cores ()
let needs_two_cores = P.needs_two_cores

(* Bodies on a worker *)

let test_worker_mask () =
  needs_two_cores ();
  let worked, mask = T.worker_mask () in
  equal ~msg:"a worker ran a chunk" bool true worked;
  equal ~msg:"(signal, blocked)"
    (list (pair string bool))
    (List.map (fun (s, _) -> (s, not (List.mem s T.faults))) mask)
    mask

(* [in_child s] is the values of scenario [s], once its child answered. *)
let in_child scenario =
  let status, values = T.in_child scenario in
  equal ~msg:"how the child ended" string "exit 0" status;
  values

let test_stack () =
  needs_two_cores ();
  let v = in_child T.Stack in
  equal ~msg:"a worker ran the chunk" int 1 v.(0)

let test_faults () =
  needs_two_cores ();
  let v = in_child T.Faults in
  equal ~msg:"a worker ran the chunk" int 1 v.(0);
  equal ~msg:"(signal, its handler ran on the worker)"
    (list (pair string bool))
    (List.map (fun s -> (s, true)) T.faults)
    (List.mapi (fun k s -> (s, v.(1) land (1 lsl k) <> 0)) T.faults)

let worker_tests =
  group ~timeout:P.timeout "workers"
    [
      test "a body on a worker has 8 MiB of stack" test_stack;
      test "workers block every signal but those a body raises itself"
        test_worker_mask;
      test "a fault signal a body raises on a worker runs its handler there"
        test_faults;
    ]

(* Threads *)

let needs_thread_states () =
  if T.running_threads () < 0 then
    skip ~reason:"the system does not report its threads' states" ()

(* [settle_to n] polls until at most [n] threads other than this one run, for at
   most 5 s, and is the last count. *)
let settle_to n = P.settle 5. T.running_threads (fun k -> k <= n)

let test_made_once () =
  needs_two_cores ();
  let v = in_child T.Threads in
  if v.(0) < 0 then skip ~reason:"the system does not count its threads" ();
  equal ~msg:"threads after a job of one thread, as before any" int v.(0) v.(1);
  greater ~msg:"threads after the first job of two" int ~than:v.(0) v.(2);
  equal ~msg:"threads after jobs on every core, as after the first of two" int
    v.(2) v.(3)

(* An idle pool's threads park. A job then wakes a parked worker, and a caller
   whose job a worker holds parks until the job ends. *)
let test_parks () =
  needs_thread_states ();
  needs_two_cores ();
  ignore (P.record ~threads:cores ~total:64L ~chunks:8L);
  equal ~msg:"threads running once the pool is idle" int 0 (settle_to 0);
  P.reset ();
  let finished = Atomic.make false in
  let caller =
    Domain.spawn (fun () ->
        Fun.protect
          ~finally:(fun () -> Atomic.set finished true)
          (fun () -> P.hold ~only_worker:true))
  in
  P.within 10. "a parked worker was not woken for the job" (fun () ->
      P.hold_arrived () = 2);
  equal ~msg:"threads running while a worker holds the job: the caller parked"
    int 1 (settle_to 1);
  P.hold_release ();
  P.within 10. "the parked caller was not woken at the job's end" (fun () ->
      Atomic.get finished);
  equal ~msg:"the job was released" bool true (Domain.join caller)

(* A burst of jobs of two threads takes the caller and one worker. The other
   workers, which the job before the burst kept spinning and which take part in
   none of its jobs, park while it runs: once at most two threads run beside
   this one, the count read every millisecond stays there. A worker that a
   wakeup reaches runs for a moment, so the median is what counts. *)
let test_narrow_burst () =
  needs_thread_states ();
  if cores < 3 then skip ~reason:"every worker takes part" ();
  P.reset ();
  let burst = Domain.spawn T.burst in
  let samples =
    Fun.protect
      ~finally:(fun () ->
        T.burst_stop ();
        Domain.join burst)
      (fun () ->
        ignore (settle_to 2);
        List.init 51 (fun _ ->
            Unix.sleepf 0.001;
            T.running_threads ()))
  in
  let median = List.nth (List.sort Int.compare samples) 25 in
  at_most
    ~msg:
      (Printf.sprintf
         "threads running beside the burst's caller and its worker, of %s"
         (String.concat " " (List.map string_of_int samples)))
    int ~than:2 median

(* The child answers after a job of the parent's on every core; the parent's
   pool then runs a job that needs a worker. *)
let test_fork_child () =
  needs_two_cores ();
  let v = in_child T.Job in
  equal ~msg:"(a worker ran a chunk, units not run once)" (pair int int) (1, 0)
    (v.(0), v.(1));
  equal
    ~msg:"in the parent, the other chunks ran on a worker while chunk 0 lasted"
    bool true
    (P.finishes "the parent's job after the fork" (fun () -> P.balance 16))

(* Nothing signals that fork waits, so the test samples: once the domain is
   about to fork, fork has still not returned 50 ms later. *)
let test_fork_waits () =
  needs_two_cores ();
  let forking = Atomic.make false and forked = Atomic.make false in
  let forker =
    P.while_held (fun () ->
        let d =
          Domain.spawn (fun () ->
              Atomic.set forking true;
              T.fork ();
              Atomic.set forked true)
        in
        P.within 10. "the domain did not reach fork" (fun () ->
            Atomic.get forking);
        Unix.sleepf 0.05;
        equal ~msg:"fork returned while a job ran" bool false
          (Atomic.get forked);
        d)
  in
  P.within 10. "fork did not return once the job ended" (fun () ->
      Atomic.get forked);
  Domain.join forker

let thread_tests =
  group ~timeout:P.timeout "threads"
    [
      test
        "the workers are made at the first job of more than one thread and \
         live until the process exits"
        test_made_once;
      test "the pool's threads park when idle, and a job wakes them" test_parks;
      test "a worker that a burst of narrow jobs leaves out parks"
        test_narrow_burst;
      test
        "a child made by fork runs its jobs on workers of its own, and the \
         parent on its own"
        test_fork_child;
      test
        "fork waits for a running job of more than one thread to end, sampled \
         for 50 ms"
        test_fork_waits;
    ]

let () = exit (run "nx_pool.h threads" [ worker_tests; thread_tests ])
