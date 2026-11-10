(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The CUDA rows against this machine's baseline. A call runs a row's kernel a
   fixed count of times behind a hold and returns once they are done: its wall
   time is their span on the GPU and a submission's round trip. [bench_cuda.exe
   gate] prints the device times themselves, and [bench_cuda.exe compare] the
   idle times of the rows that PyTorch can time too (compare.py). *)

module S = Nx_cuda_support
module D = Nx_array.Dtype

(* Runs per call: about 2 ms of a memory-bound row at 500 GB/s; a peak row's run
   spans 1-2 ms itself. *)
let count (p : Rows.prepared) =
  match p.work with
  | Bytes 0 -> 1000
  | Bytes n -> Int.max 1 (1_000_000_000 / n)
  | Flops _ -> 1

let case (r : Rows.row) =
  let setup () =
    let g = S.gpu () in
    let p = r.make g in
    (g, p, count p)
  in
  Thumper.bench_with_setup ~setup r.name (fun (g, (p : Rows.prepared), count) ->
      S.device_time g p.run ~count)

(* The host's share of a call: 64 calls of the smallest contraction issued
   behind a hold, then their work once released. A call that finds nx.cuda's
   caches warm allocates its launch, which dies with it. *)
let host =
  let bf16 = D.Any D.Bfloat16 and f32 = D.Any D.Float32 in
  let setup () =
    let g = S.gpu () in
    let _, _, _, run = Rows.plan g bf16 f32 bf16 1 1 1 in
    (g, run)
  in
  [
    Thumper.bench_with_setup ~setup "issue/contract-64" (fun (g, run) ->
        S.issue g ~count:64 run);
  ]

(* Gate *)

(* The gate: a table of each row's device time per run under two protocols, its
   rate, and its distance to the floor kernel that moves its bytes, timed in the
   same run, with the SM clock each protocol ended at.

   bench_cuda.exe gate [PATTERN...]

   A row runs if its name starts with a PATTERN, or is a PATTERN that ends with
   $ but for it; every row runs without one. First the GPU's clock settles on
   the row: it runs 100 ms at a time, at most ten times, until two clocks read
   after it agree within 1%. Then:

   - idle: the median of 30 measurements, each of enough runs to span 10 ms of
   the GPU's time, after the GPU idled 20 ms, as compare.py times PyTorch, so
   that both run in the same power state; a run longer than 100 ms runs alone
   in a measurement, measured as often as 3 s of runs allow, at least 5
   times;
   - sustained: the median of 10 such measurements back to back, after 1 s of
   the row without pause, as a training loop runs it: a GPU busy without pause
   reaches its power cap and lowers its clock.

   The runs that settle the clock and the second of the row run without a
   hold: they measure nothing, and a hold waits for the host to make every
   call it covers. A second of contract/bfloat16-4096x4096x65536 is 76 calls,
   each allocating its 128 MiB of partials, some for 150 ms: past the hold's
   2 s on a loaded host.

   bench_cuda.exe compare [PATTERN...]

   prints, for each selected row with a spec, its name, its idle time in
   microseconds and its spec, separated by tabs: what compare.py reads. A
   row whose runs a hold cannot queue, as a call of more launches than the
   stream holds behind it, is timed from the host instead, each run
   submitted and waited for, and adds a fourth field, [wall]. *)

let strf = Printf.sprintf
let span = 10e-3
let warm = 0.1
let idle = 20e-3
let long_span = 3.

let median xs =
  let a = Array.of_list xs in
  Array.sort compare a;
  a.(Array.length a / 2)

(* Seconds per run of [r] when idle and the clock it ended at, once the clock
   settled on [r]; with the count of runs that spans a given time and the
   measurement, for {!time} to go on with. *)
let time_idle g r =
  let t = S.device_time g r ~count:1 in
  let t = if t < 1e-3 then S.device_time g r ~count:3 else t in
  let long = t > warm in
  let runs s =
    if long then 1 else Int.max 3 (int_of_float (Float.ceil (s /. t)))
  in
  let rec settle last tries =
    S.busy g r ~count:(runs warm);
    let c = S.sm_clock g in
    if tries > 1 && Float.abs (c -. last) > 0.01 *. c then settle c (tries - 1)
  in
  settle 0. 10;
  let measure () = S.device_time g r ~count:(runs span) in
  let measurements =
    if long then Int.max 5 (Int.min 30 (int_of_float (long_span /. t))) else 30
  in
  let idle =
    median
      (List.init measurements (fun _ ->
           Unix.sleepf idle;
           measure ()))
  in
  (idle, S.sm_clock g, runs, measure)

(* Seconds per run of [r] when idle and sustained, and the clock each ended
   at. *)
let time g r =
  let idle, idle_clock, runs, measure = time_idle g r in
  S.busy g r ~count:(runs 1.);
  let sustained = median (List.init 10 (fun _ -> measure ())) in
  ((idle, idle_clock), (sustained, S.sm_clock g))

let rate (p : Rows.prepared) t =
  match p.work with
  | Bytes 0 -> ""
  | Bytes n -> strf "%7.1f GB/s" (float n /. t *. 1e-9)
  | Flops f -> strf "%7.1f TF/s" (f /. t *. 1e-12)

let row g (r : Rows.row) =
  let p = r.make g in
  let (t, c), (ts, cs) = time g p.run in
  let floor =
    match p.floor with
    | None -> ""
    | Some f ->
        let (tf, _), _ = time g f in
        strf "floor %9.2f us  x%.2f" (tf *. 1e6) (t /. tf)
  in
  Printf.printf
    "%-34s idle %10.2f us %s [%.0f MHz]  sustained %10.2f us %s [%.0f MHz]  %s\n\
     %!"
    r.name (t *. 1e6) (rate p t) c (ts *. 1e6) (rate p ts) cs floor

let selected patterns (r : Rows.row) =
  patterns = []
  || List.exists
       (fun p ->
         match String.ends_with ~suffix:"$" p with
         | true -> String.sub p 0 (String.length p - 1) = r.name
         | false -> String.starts_with ~prefix:p r.name)
       patterns

let gate patterns =
  let g = S.gpu () in
  Printf.printf "# %s, %d SMs\n%!" (S.arch g) (S.sms g);
  List.iter (row g) (List.filter (selected patterns) Rows.compared)

(* Compare *)

let fst4 (a, _, _, _) = a

(* Seconds per run of [r] from the host: the median of 5 runs, each
   submitted and waited for, after one. It counts the host's share of the
   call, and is for a call that a hold cannot hold. *)
let wall g r =
  S.run g r;
  median
    (List.init 5 (fun _ ->
         let t0 = Unix.gettimeofday () in
         S.run g r;
         Unix.gettimeofday () -. t0))

let compare_rows patterns =
  let g = S.gpu () in
  let one (r : Rows.row) =
    match r.spec with
    | None -> ()
    | Some spec ->
        (* A hold that runs out, as the host stalls on a loaded machine or
           a call submits more launches than the stream queues behind it,
           times the row once more, then from the host, with a note on
           stderr and a fourth field. *)
        let measure () =
          let p = r.make g in
          fst4 (time_idle g p.run)
        in
        let t =
          try (measure (), "")
          with Failure _ -> (
            try (measure (), "")
            with Failure m ->
              Printf.eprintf "%s: %s; timed from the host\n%!" r.name m;
              let p = r.make g in
              (wall g p.run, "\twall"))
        in
        let t, note = t in
        Printf.printf "%s\t%.3f\t%s%s\n%!" r.name (t *. 1e6)
          (Rows.spec_text spec) note
  in
  List.iter one (List.filter (selected patterns) Rows.compared)

let () =
  (* Looks for the GPU without opening it: the bench's rows open it in thumper's
     workers, after it forks them. *)
  if not (Sys.file_exists "/dev/nvidiactl") then begin
    print_endline "nx.cuda bench: no NVIDIA GPU";
    exit 0
  end;
  match List.tl (Array.to_list Sys.argv) with
  | "gate" :: patterns -> gate patterns
  | "compare" :: patterns -> compare_rows patterns
  | _ -> exit (Thumper.run "nx_cuda" (host @ List.map case Rows.all))
