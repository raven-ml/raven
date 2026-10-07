(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The suite of nx_pool.h, through the probes of pool_probe_stubs.c. *)

open Windtrap
module P = Pool_probe

(* A process this executable starts, to see the first call of nx_pool_cores
   under a pinned affinity. It must call nothing of the pool before, so this
   comes first. *)
let child_variable = "NX_POOL_TEST_CHILD"

let () =
  match Sys.getenv_opt child_variable with
  | Some "affinity" ->
      let first, later = P.pinned_cores () in
      Printf.printf "%d %d\n%!" first later;
      exit 0
  | _ -> ()

let cores = P.cores ()
let needs_two_cores = P.needs_two_cores

(* Jobs by nx_pool.h *)

(* c, the chunks of a job. *)
let chunk_count ~total ~chunks =
  if total <= 0L then 0L else Int64.min (Int64.max chunks 1L) total

(* t, the most threads of a job. *)
let thread_bound ~threads ~total ~chunks =
  let c = chunk_count ~total ~chunks in
  Int64.to_int
    (Int64.min
       (Int64.of_int (max threads 1))
       (Int64.min (Int64.of_int cores) c))

(* Chunk i is [floor (i * total / c), floor ((i + 1) * total / c)). With total =
   q c + r, floor (i * total / c) = i q + floor (i r / c), whose products stay
   below total and c * c. *)
let partition ~total ~chunks =
  let c = chunk_count ~total ~chunks in
  if c = 0L then []
  else
    let q = Int64.div total c and r = Int64.rem total c in
    let bound i = Int64.(add (mul i q) (div (mul i r) c)) in
    List.init (Int64.to_int c) (fun i ->
        let i = Int64.of_int i in
        (bound i, bound (Int64.succ i)))

(* The bounds of the chunks, floor (i * total / c) for 0 <= i <= c. *)
let chunk_bounds ~total ~chunks =
  match partition ~total ~chunks with
  | [] -> []
  | chunks -> 0L :: List.map snd chunks

(* [cut bounds ranges] is each of [ranges] cut at the [bounds] strictly inside
   it. Disjoint ranges of whole chunks that cover the units cut into the chunks,
   each once; a range that ends inside a chunk, or is empty, leaves a piece that
   is no chunk, and ranges that overlap leave a chunk twice. *)
let cut bounds ranges =
  let bounds = Array.of_list bounds in
  let n = Array.length bounds in
  (* The index of the first bound above [x]. *)
  let rec above x lo hi =
    if lo >= hi then lo
    else
      let mid = (lo + hi) / 2 in
      if bounds.(mid) > x then above x lo mid else above x (mid + 1) hi
  in
  let rec pieces i start hi =
    if i < n && bounds.(i) < hi then
      (start, bounds.(i)) :: pieces (i + 1) bounds.(i) hi
    else [ (start, hi) ]
  in
  List.concat_map (fun (lo, hi) -> pieces (above lo 0 n) lo hi) ranges

let chunks_ran (job : P.job) =
  List.sort compare (List.map (fun (c : P.call) -> (c.lo, c.hi)) job.calls)

let distinct l = List.sort_uniq compare l

(* Waiting *)

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
  P.within 10. (what ^ " had not returned") (fun () ->
      Option.is_some (Atomic.get result));
  Domain.join d;
  match Option.get (Atomic.get result) with Ok v -> v | Error e -> raise e

(* Cores *)

let test_core_bounds () =
  let fast = P.performance_cores () in
  at_least ~msg:"cores" int ~than:1 cores;
  at_least ~msg:"performance cores" int ~than:1 fast;
  at_most ~msg:"performance cores" int ~than:cores fast

let test_macos_cores () =
  let physical = P.sysctl "hw.physicalcpu" in
  if physical < 0 then skip ~reason:"the host has no hw.physicalcpu" ();
  equal ~msg:"cores" int physical cores;
  let fast = P.sysctl "hw.perflevel0.physicalcpu" in
  if fast < 0 then skip ~reason:"the host has cores of one kind" ();
  equal ~msg:"performance cores" int fast (P.performance_cores ())

let test_other_performance_cores () =
  if P.sysctl "hw.perflevel0.physicalcpu" >= 0 then
    skip ~reason:"the host reports its performance cores" ();
  equal int cores (P.performance_cores ())

(* A process of this executable pins itself to one CPU, reads the cores, then
   restores its affinity and reads them again. *)
let test_affinity () =
  let env =
    Array.append (Unix.environment ()) [| child_variable ^ "=affinity" |]
  in
  let exe = Sys.executable_name in
  let ((out, _, _) as process) =
    Unix.open_process_args_full exe [| exe |] env
  in
  let line = In_channel.input_line out in
  ignore (Unix.close_process_full process);
  if line = Some "-1 -1" then skip ~reason:"the host has no affinity to pin" ();
  equal
    ~msg:"cores at the first call, on one CPU, and after the affinity returned"
    (option string) (Some "1 1") line

let test_windows_cores () =
  let active = P.active_processors () in
  if active < 0 then skip ~reason:"the host has no processor groups" ();
  equal ~msg:"cores" int active cores

let cores_tests =
  group "cores"
    [
      test "the cores and the performance cores are within their bounds"
        test_core_bounds;
      test
        "on macOS the cores are the physical cores and the performance cores \
         those of hw.perflevel0"
        test_macos_cores;
      test "where the host reports no performance cores, they are the cores"
        test_other_performance_cores;
      test "on Windows the cores are the active processors of every group"
        test_windows_cores;
      test
        "on Linux the cores are bounded by the affinity at the first call, and \
         a later change is not seen"
        test_affinity;
    ]

(* Chunks *)

let pp_int64 ppf x = Format.fprintf ppf "%LdL" x

let int64_extremes =
  Int64.
    [
      min_int;
      succ min_int;
      -1L;
      0L;
      1L;
      div max_int 2L;
      succ (div max_int 2L);
      pred max_int;
      max_int;
    ]

let threads_gen =
  Gen.frequency
    [
      (6, Gen.int_range (-2) (cores + 2));
      ( 1,
        Gen.of_list ~pp:Format.pp_print_int
          [ Int32.(to_int min_int); -1; 0; 1; Int32.(to_int max_int) ] );
    ]

let total_gen =
  Gen.frequency
    [
      (4, Gen.int64_range (-2L) 300L);
      (2, Gen.int64_range 0L 100_000L);
      (2, Gen.of_list ~pp:pp_int64 int64_extremes);
    ]

let chunks_gen =
  Gen.frequency
    [
      (4, Gen.int64_range (-2L) 70L);
      (1, Gen.int64_range 0L 4096L);
      (1, Gen.of_list ~pp:pp_int64 int64_extremes);
    ]

let pp_job ppf (threads, total, chunks) =
  Format.fprintf ppf "threads %d, total %LdL, chunks %LdL" threads total chunks

let job_gen = Gen.with_pp pp_job (Gen.triple threads_gen total_gen chunks_gen)

(* A drawn job records its every call. *)
let max_drawn_chunks = 16_384L

let assume_recordable (_, total, chunks) =
  assume (chunk_count ~total ~chunks <= max_drawn_chunks)

let job_examples =
  [
    (4, 0L, 3L);
    (3, -5L, 2L);
    (0, 10L, 0L);
    (cores + 3, 5L, 8L);
    (2, Int64.max_int, 3L);
    (1, 1000L, 7L);
  ]

let test_bounds ((threads, total, chunks) as job) =
  assume_recordable job;
  let c = chunk_count ~total ~chunks
  and t = thread_bound ~threads ~total ~chunks in
  cover "no unit" (total <= 0L);
  cover "fewer than one chunk" (total > 0L && chunks < 1L);
  cover "more chunks than units" (total > 0L && chunks > total);
  cover "units times chunks past 64 bits"
    (c > 1L && total > Int64.div Int64.max_int c);
  cover "fewer than one thread" (total > 0L && threads < 1);
  cover "more threads than cores" (c > Int64.of_int cores && threads > cores);
  cover "a job on one thread" (t = 1);
  if cores > 1 then cover "a job on several threads" (t > 1);
  let ran = P.record ~threads ~total ~chunks in
  let ranges = chunks_ran ran in
  at_most ~msg:"calls" int ~than:(Int64.to_int c) ran.count;
  equal ~msg:"the calls' ranges cut at the chunk bounds"
    (list (pair int64 int64))
    (partition ~total ~chunks)
    (cut (chunk_bounds ~total ~chunks) ranges);
  if t = 1 then
    equal ~msg:"the calls of a job on one thread"
      (list (pair int64 int64))
      [ (0L, total) ]
      ranges

(* Calls in the order they began, which is the order of their claims on one
   thread. *)
let test_claim_order ((threads, total, chunks) as job) =
  assume_recordable job;
  let ran = P.record ~threads ~total ~chunks in
  let threads = distinct (List.map (fun (c : P.call) -> c.thread) ran.calls) in
  let los th =
    List.filter_map
      (fun (c : P.call) -> if c.thread = th then Some c.lo else None)
      ran.calls
  in
  (* A job of more chunks than threads, on more than one thread, has a thread
     run several: its calls are one a chunk. *)
  if cores > 1 then
    cover "a thread ran several chunks"
      (List.exists (fun th -> List.length (los th) >= 2) threads);
  List.iter
    (fun th ->
      equal
        ~msg:(Printf.sprintf "the chunks thread %d ran, in its order" th)
        (list int64)
        (distinct (los th))
        (los th))
    threads

let test_balance () =
  needs_two_cores ();
  equal ~msg:"the other chunks ran while chunk 0 lasted" bool true
    (P.balance 16)

let top = Int64.max_int

let wide_jobs =
  [
    (top, 2L, [ (0L, 4611686018427387903L); (4611686018427387903L, top) ]);
    ( top,
      3L,
      [
        (0L, 3074457345618258602L);
        (3074457345618258602L, 6148914691236517204L);
        (6148914691236517204L, top);
      ] );
    ( Int64.pred top,
      7L,
      [
        (0L, 1317624576693539400L);
        (1317624576693539400L, 2635249153387078801L);
        (2635249153387078801L, 3952873730080618202L);
        (3952873730080618202L, 5270498306774157603L);
        (5270498306774157603L, 6588122883467697004L);
        (6588122883467697004L, 7905747460161236405L);
        (7905747460161236405L, Int64.pred top);
      ] );
    (5L, 8L, [ (0L, 1L); (1L, 2L); (2L, 3L); (3L, 4L); (4L, 5L) ]);
    (3L, top, [ (0L, 1L); (1L, 2L); (2L, 3L) ]);
  ]

let test_wide_job (total, chunks, expected) =
  let ran = chunks_ran (P.record ~threads:cores ~total ~chunks) in
  equal ~msg:"the calls' ranges cut at the stated bounds"
    (list (pair int64 int64))
    expected
    (cut (0L :: List.map snd expected) ran)

let chunk_tests =
  group "chunks"
    [
      prop ~count:300 ~examples:job_examples
        "a job calls ranges of whole chunks of the stated bounds that cover \
         its units once, one range on one thread, and nothing for no unit"
        job_gen test_bounds;
      cases
        ~name:(fun (total, chunks, _) ->
          Printf.sprintf "total %Ld, chunks %Ld" total chunks)
        "a job whose total times its chunks passes 64 bits, or of more chunks \
         than units, calls ranges of the stated chunks"
        wide_jobs test_wide_job;
      prop ~examples:job_examples "each thread claims its chunks in index order"
        job_gen test_claim_order;
      test
        "a thread that finishes early runs the chunks a slower one would have"
        test_balance;
    ]

(* Workers *)

let test_workers ((threads, total, chunks) as job) =
  assume_recordable job;
  let t = thread_bound ~threads ~total ~chunks in
  let ran = P.record ~threads ~total ~chunks in
  let pairs =
    distinct (List.map (fun (c : P.call) -> (c.worker, c.thread)) ran.calls)
  in
  let workers = distinct (List.map fst pairs)
  and threads_used = distinct (List.map snd pairs) in
  cover "fewer than one thread" (total > 0L && threads < 1);
  cover "more threads than cores" (threads > cores && t = cores);
  List.iter
    (fun w ->
      at_least ~msg:"a worker index" int ~than:0 w;
      less ~msg:"a worker index" int ~than:t w)
    workers;
  List.iter
    (fun (w, th) ->
      if th = 0 then equal ~msg:"the calling thread's worker" int 0 w)
    pairs;
  equal ~msg:"workers, each on one thread" (list int) workers
    (List.map fst pairs);
  equal ~msg:"threads, each one worker" (list int) threads_used
    (List.sort compare (List.map snd pairs));
  equal ~msg:"calls begun while another of their worker ran" int 0 ran.overlaps

let test_visibility () =
  let bodies, caller = P.visibility ~jobs:2000 ~threads:cores in
  equal ~msg:"values bodies read stale" int 0 bodies;
  equal ~msg:"values the caller read stale" int 0 caller

let worker_tests =
  group "workers"
    [
      prop ~examples:job_examples
        "a call's worker is below the job's threads, the caller's 0, one per \
         thread, and its calls never overlap"
        job_gen test_workers;
      test
        "bodies see the caller's writes made before the job, and the caller \
         the bodies' once it returns"
        test_visibility;
    ]

(* Scheduling *)

let test_one_thread_at_once () =
  needs_two_cores ();
  P.while_held (fun () ->
      let ran =
        finishes "a job of one thread" (fun () ->
            P.record ~threads:1 ~total:100L ~chunks:10L)
      in
      equal ~msg:"(lo, hi, worker, thread) of each call"
        (list (quad int64 int64 int int))
        [ (0L, 100L, 0, 0) ]
        (List.map
           (fun (c : P.call) -> (c.lo, c.hi, c.worker, c.thread))
           ran.calls))

let test_nested () =
  let outer = 16 and inner = 8 in
  equal
    ~msg:
      "(outer units run, inner jobs not run in one call, misplaced inner \
       calls, inner units not run once)"
    (quad int int int int) (outer, 0, 0, 0)
    (finishes "a job whose bodies begin jobs" (fun () ->
         P.nested ~threads:cores ~outer ~inner))

(* Nothing signals that a job waits, so the test samples: once the domain is
   about to begin its job, none of its chunks has run 50 ms later. *)
let test_waits () =
  needs_two_cores ();
  let beginning = Atomic.make false in
  let waiting =
    P.while_held (fun () ->
        let d =
          Domain.spawn (fun () ->
              Atomic.set beginning true;
              P.counted ~threads:2 ~total:4L ~chunks:4L)
        in
        P.within 10. "the domain did not begin its job" (fun () ->
            Atomic.get beginning);
        Unix.sleepf 0.05;
        equal ~msg:"chunks run while another thread's job ran" int 0
          (P.counted_calls ());
        d)
  in
  P.within 10. "the waiting job did not run once the other ended" (fun () ->
      P.counted_calls () = 4);
  Domain.join waiting

let pp_small_job ppf (threads, total, chunks) =
  Format.fprintf ppf "threads %d, total %LdL, chunks %LdL" threads total chunks

let job_commands =
  let small_job =
    Gen.with_pp pp_small_job
      (Gen.triple
         (Gen.int_range (-1) (cores + 1))
         (Gen.int64_range (-1L) 2000L)
         (Gen.int64_range (-1L) 64L))
  in
  [
    command "job"
      (small_job @-> returns (list (pair int64 int64)))
      (fun (_, total, chunks) -> partition ~total ~chunks)
      (fun (threads, total, chunks) ->
        cut
          (chunk_bounds ~total ~chunks)
          (chunks_ran (P.record ~threads ~total ~chunks)));
  ]

let scheduling_tests =
  group "scheduling"
    [
      test "a job of one thread runs at once while another thread's job runs"
        test_one_thread_at_once;
      test
        "a job begun from a body of a job of more than one thread runs at once \
         on the body's thread as worker 0, in one call"
        test_nested;
      test
        "a job of more than one thread waits for another thread's job to end, \
         sampled for 50 ms"
        test_waits;
      stateful ~domains:2 ~count:30
        "jobs from two domains each call their chunks once" job_commands;
    ]

let () =
  exit
    (run "nx_pool.h"
       [ cores_tests; chunk_tests; worker_tests; scheduling_tests ])
