(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The host pool's jobs, as a kernel runs them.

   Launches are empty jobs: what the pool costs a caller, alone (with one chunk,
   and with the 8 a kernel's serial job may pass), on the performance cores and
   on every core. Thumper calls a case back to back, so workers that a job's
   predecessor ran still spin; the rows 5 us and 150 us apart keep the caller
   busy that long before each job, within the spin window and past it, where the
   job wakes parked workers. A job of 2 threads runs while the other workers are
   parked, and pays no wake for them.

   Claims are a job of 65,536 empty chunks: one claim each.

   Compute jobs are 65,536 units of 64 dependent multiply-adds, 8 chunks a
   thread: on one thread, on the performance cores, on every core, with the cost
   skewed (the costliest units first, the threads balanced by claiming), and
   against one spinning thread per core that another part of the program
   started. *)

external cores : unit -> int = "pool_bench_cores"
external performance_cores : unit -> int = "pool_bench_performance_cores"
external empty : int -> int -> int -> unit = "pool_bench_empty" [@@noalloc]
external empty_after : int -> int -> unit = "pool_bench_empty_after" [@@noalloc]

external compute : int -> int -> int -> bool -> unit = "pool_bench_compute"
[@@noalloc]

external load_start : int -> unit = "pool_bench_load_start"
external load_stop : unit -> unit = "pool_bench_load_stop"

let cores = cores ()
let fast = performance_cores ()
let us = 1_000
let units = 1 lsl 16
let chunks_per_thread = 8

let launch =
  let job threads () = empty threads threads threads in
  let after gap () = empty_after gap cores in
  Thumper.group "launch"
    [
      Thumper.bench "empty-1-thread" (job 1);
      Thumper.bench "empty-1-thread-8-chunks" (fun () -> empty 1 8 8);
      Thumper.bench "empty-2-threads" (job 2);
      Thumper.bench "empty-performance-cores" (job fast);
      Thumper.bench "empty-all-cores" (job cores);
      Thumper.bench "empty-all-cores-5us-apart" (after (5 * us));
      Thumper.bench "empty-all-cores-150us-apart" (after (150 * us));
    ]

let claim =
  let chunks = 65_536 in
  Thumper.group "claim"
    [
      Thumper.bench "65536-empty-chunks-all-cores" (fun () ->
          empty cores chunks chunks);
    ]

let compute =
  let job ?(skewed = false) threads () =
    compute threads units (chunks_per_thread * threads) skewed
  in
  Thumper.group "compute"
    [
      Thumper.bench "balanced-1-thread" (job 1);
      Thumper.bench "balanced-performance-cores" (job fast);
      Thumper.bench "balanced-all-cores" (job cores);
      Thumper.bench "skewed-performance-cores" (job ~skewed:true fast);
      (* The load's own CPU time would swamp the job's. *)
      Thumper.bench_with_setup
        ~metrics:Thumper.Metric.[ wall_time; alloc_words ]
        ~setup:(fun () -> load_start cores)
        ~teardown:load_stop "balanced-performance-cores-under-load"
        (fun () -> job fast ());
    ]

(* The process's CPU time counts what spinning workers spend. A batch of at
   least 50 ms spans several of the scheduler's time slices, which steadies the
   rows a competing load shares the cores with. *)
let config =
  Thumper.Config.(
    default |> batch_floor 0.05
    |> metrics Thumper.Metric.[ wall_time; cpu_time; alloc_words ])

let () = exit (Thumper.run ~config "nx_pool" [ launch; claim; compute ])
