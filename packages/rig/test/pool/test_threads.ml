(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The suite of rig_pool.h's threads, through the probes of
   rig_pool_thread_probe_stubs.c: what a body sees on a worker, and the workers'
   life. *)

open Windtrap
module P = Rig_pool_probe
module T = Rig_pool_thread_probe

let cores = P.cores ()
let kinds = [ P.Compute; P.Blocking ]
let kind_name = function P.Compute -> "compute" | P.Blocking -> "blocking"

(* Bodies on a worker *)

let test_worker_mask kind =
  P.needs_two_cores ();
  let worked, mask = T.worker_mask kind in
  equal ~msg:"a worker ran a chunk" bool true worked;
  equal ~msg:"(signal, blocked)"
    (list (pair string bool))
    (List.map (fun (s, _) -> (s, not (List.mem s T.faults))) mask)
    mask

(* [in_child kind s] is the values of scenario [s] with jobs of [kind], once its
   child answered. *)
let in_child ?(kind = P.Compute) scenario =
  P.needs_fork ();
  let status, values = T.in_child kind scenario in
  equal ~msg:"how the child ended" string "exit 0" status;
  values

let test_stack kind =
  P.needs_two_cores ();
  let v = in_child ~kind T.Stack in
  equal ~msg:"a worker ran the chunk" int 1 v.(0)

let test_faults kind =
  P.needs_two_cores ();
  let v = in_child ~kind T.Faults in
  equal ~msg:"a worker ran the chunk" int 1 v.(0);
  equal ~msg:"(signal, its handler ran on the worker)"
    (list (pair string bool))
    (List.map (fun s -> (s, true)) T.faults)
    (List.mapi (fun k s -> (s, v.(1) land (1 lsl k) <> 0)) T.faults)

let worker_tests =
  group ~timeout:P.timeout "workers"
    [
      cases "a body on a worker has 8 MiB of stack" ~id:kind_name kinds
        test_stack;
      cases "workers block every signal but those a body raises itself"
        ~id:kind_name kinds test_worker_mask;
      cases "a fault signal a body raises on a worker runs its handler there"
        ~id:kind_name kinds test_faults;
    ]

(* Threads *)

let needs_thread_states () =
  if T.running_threads () < 0 then
    skip ~reason:"the system does not report its threads' states" ()

(* [settle_to n] polls until at most [n] threads other than this one run, for at
   most 5 s, and is the last count. *)
let settle_to n = P.settle 5. T.running_threads (fun k -> k <= n)

let test_made_once kind =
  P.needs_two_cores ();
  let v = in_child ~kind T.Threads in
  if v.(0) < 0 then skip ~reason:"the system does not count its threads" ();
  equal ~msg:"threads after a job of one thread, as before any" int v.(0) v.(1);
  greater ~msg:"threads after the first job of two" int ~than:v.(0) v.(2);
  equal ~msg:"threads after jobs on every core, as after the first of two" int
    v.(2) v.(3)

(* An idle pool's threads park. A job then wakes a parked worker, and a caller
   whose job a worker holds parks until the job ends. *)
let test_parks kind =
  needs_thread_states ();
  P.needs_two_cores ();
  ignore (P.record ~kind ~threads:cores ~total:64L ~chunks:8L ());
  equal ~msg:"threads running once the pool is idle" int 0 (settle_to 0);
  P.reset ();
  let finished = Atomic.make false in
  let caller =
    Domain.spawn (fun () ->
        Fun.protect
          ~finally:(fun () -> Atomic.set finished true)
          (fun () -> P.hold ~kind ~only_worker:true ()))
  in
  P.within 10. "a parked worker was not woken for the job" (fun () ->
      P.hold_arrived () = 2);
  equal ~msg:"threads running while a worker holds the job: the caller parked"
    int 1 (settle_to 1);
  P.hold_release ();
  P.within 10. "the parked caller was not woken at the job's end" (fun () ->
      Atomic.get finished);
  equal ~msg:"the job was released" bool true (Domain.join caller)

(* The child answers after a job of the parent's on every core; the parent's
   pool then runs a job that needs a worker. *)
let test_fork_child kind =
  P.needs_two_cores ();
  let v = in_child ~kind T.Job in
  equal ~msg:"(a worker ran a chunk, units not run once)" (pair int int) (1, 0)
    (v.(0), v.(1));
  equal ~msg:"in the parent, a job's two chunks ran at once" bool true
    (P.finishes "the parent's job after the fork" (fun () -> P.meet kind))

(* fork returns while another thread's blocking job runs, and the child runs a
   blocking job of its own on its own workers. *)
let test_fork_beside_blocking () =
  P.needs_two_cores ();
  P.needs_fork ();
  let during, status = T.fork_beside_blocking () in
  equal ~msg:"how the child ended" string "exit 0" status;
  equal ~msg:"fork returned while the blocking job ran" bool true during

(* The child limits its own threads to none, so the pool makes no worker. Its
   jobs run on the calling thread, and the pool makes none once the limit is
   lifted. *)
let test_unmade () =
  if not (T.limits_threads ()) then
    skip ~reason:"the system cannot limit a process's own threads" ();
  P.needs_two_cores ();
  let v = in_child T.Limited in
  if v.(0) = 0 then skip ~reason:"the thread limit does not bind this user" ();
  equal ~msg:"units not run once by each job" int 0 v.(1);
  equal ~msg:"calls on a worker" int 0 v.(2);
  equal ~msg:"threads after both jobs" int 1 v.(3)

let thread_tests =
  group ~timeout:P.timeout "threads"
    [
      cases
        "a kind's workers are made at its first job of more than one thread \
         and live until the process exits"
        ~id:kind_name kinds test_made_once;
      cases "the pool's threads park when idle, and a job wakes them"
        ~id:kind_name kinds test_parks;
      cases
        "a child made by fork runs its jobs on workers of its own, and the \
         parent on its own"
        ~id:kind_name kinds test_fork_child;
      test
        "fork returns while another thread's blocking job runs, and the child \
         runs its blocking jobs on workers of its own"
        test_fork_beside_blocking;
      test
        "workers that cannot be made are missing from every job; with none, \
         the calling thread runs every chunk"
        test_unmade;
    ]

let () = exit (run "rig_pool.h threads" [ worker_tests; thread_tests ])
