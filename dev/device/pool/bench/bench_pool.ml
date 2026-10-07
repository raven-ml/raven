(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The host pool's jobs, as a kernel runs them.

   Launches are empty jobs: what the pool costs a caller, alone (with one chunk,
   and with the 8 a kernel's serial job may pass), on the performance cores and
   on every core. A job of 2 threads runs while the other workers are parked,
   and pays no wake for them. A job of one 2 us chunk per core needs every
   thread to start within a chunk's time, or another thread runs its chunk after
   its own. Thumper calls a case back to back, so workers that a job's
   predecessor ran still spin; the rows 5 us and 150 us apart keep the caller
   busy that long before each job, within the spin window and past it, where the
   job wakes parked workers.

   Claims are a job of 65,536 empty chunks: one claim each.

   Compute jobs are 65,536 units of 64 dependent multiply-adds, 8 chunks a
   thread: on one thread, on the performance cores, on every core, with the cost
   skewed (the costliest units first, the threads balanced by claiming), and
   against one spinning process per core, another program on the host.

   A floor row runs the job of the row above it on threads of the bench's own,
   each with a fixed share of the chunks and nothing claimed: what announcing a
   job and waiting for every thread cost without the pool. *)

(* Competing load

   A process of this executable started with [spin_flag] spins until it is
   killed or its parent ends, and runs nothing else of the bench. Spinning
   domains would join the bench's collections; processes share nothing with it
   but the cores. Windows tells a process nothing of its parent, so there a
   spinner whose bench was killed spins until it is killed too. *)

let spin_flag = "--spin"

let spin parent =
  print_char '.';
  flush stdout;
  let x = ref 1 in
  while Sys.win32 || Unix.getppid () = parent do
    for _ = 1 to 1 lsl 20 do
      x := Sys.opaque_identity ((!x * 3) + 1)
    done
  done;
  exit 0

let () =
  match Sys.argv with
  | [| _; flag; parent |] when flag = spin_flag -> spin (int_of_string parent)
  | _ -> ()

(* [load_start n] is [n] spinning processes, once each spins. *)
let load_start n =
  let exe = Sys.executable_name in
  let args = [| exe; spin_flag; string_of_int (Unix.getpid ()) |] in
  let r, w = Unix.pipe ~cloexec:true () in
  let pids =
    Array.init n (fun _ ->
        Unix.create_process exe args Unix.stdin w Unix.stderr)
  in
  Unix.close w;
  let b = Bytes.create n in
  let rec await got =
    if got < n then
      match Unix.read r b got (n - got) with
      | 0 -> failwith "a spinning process ended before it spun"
      | k -> await (got + k)
  in
  Fun.protect ~finally:(fun () -> Unix.close r) (fun () -> await 0);
  pids

let load_stop pids =
  Array.iter (fun pid -> Unix.kill pid Sys.sigkill) pids;
  Array.iter (fun pid -> ignore (Unix.waitpid [] pid)) pids

(* Jobs *)

external empty : int -> int -> int -> unit = "pool_bench_empty" [@@noalloc]
external busy : int -> unit = "pool_bench_busy" [@@noalloc]

external compute : int -> int -> int -> bool -> unit = "pool_bench_compute"
[@@noalloc]

external floor_start : int -> unit = "pool_bench_floor_start"
external floor_stop : unit -> unit = "pool_bench_floor_stop"
external floor_empty : int -> int -> unit = "pool_bench_floor_empty" [@@noalloc]

external floor_compute : int -> int -> unit = "pool_bench_floor_compute"
[@@noalloc]

let cores = Pool_probe.cores ()
let fast = Pool_probe.performance_cores ()
let us = 1_000
let units = 1 lsl 16
let chunks_per_thread = 8

(* 24 units, about 2 us on a performance core. *)
let chunk_units = 24

(* [floor name f] is a floor row: [f] on [cores] floor threads. *)
let floor name f =
  Thumper.bench_with_setup
    ~setup:(fun () -> floor_start cores)
    ~teardown:floor_stop name f

let launch =
  let job threads () = empty threads threads threads in
  let chunks () = compute cores (chunk_units * cores) cores false in
  let after gap () =
    busy gap;
    chunks ()
  in
  Thumper.group "launch"
    [
      Thumper.bench "empty-1-thread" (job 1);
      Thumper.bench "empty-1-thread-8-chunks" (fun () -> empty 1 8 8);
      Thumper.bench "empty-2-threads" (job 2);
      Thumper.bench "empty-performance-cores" (job fast);
      Thumper.bench "empty-all-cores" (job cores);
      floor "floor-empty-all-cores" (fun () -> floor_empty cores cores);
      Thumper.bench "2us-chunks-all-cores" chunks;
      floor "floor-2us-chunks-all-cores" (fun () ->
          floor_compute (chunk_units * cores) cores);
      Thumper.bench "2us-chunks-all-cores-5us-apart" (after (5 * us));
      Thumper.bench "2us-chunks-all-cores-150us-apart" (after (150 * us));
    ]

let claim =
  let chunks = 65_536 in
  Thumper.group "claim"
    [
      Thumper.bench "65536-empty-chunks-all-cores" (fun () ->
          empty cores chunks chunks);
      floor "floor-65536-empty-chunks-all-cores" (fun () ->
          floor_empty chunks chunks);
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
      Thumper.bench_with_setup
        ~setup:(fun () -> load_start cores)
        ~teardown:load_stop "balanced-performance-cores-under-load"
        (fun _ -> job fast ());
    ]

(* The process's CPU time counts what spinning workers spend. A batch of at
   least 50 ms spans several of the scheduler's time slices, which steadies the
   rows a competing load shares the cores with. *)
let config =
  Thumper.Config.(
    default |> batch_floor 0.05
    |> metrics Thumper.Metric.[ wall_time; cpu_time; alloc_words ])

let () = exit (Thumper.run ~config "nx_pool" [ launch; claim; compute ])
