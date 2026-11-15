(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The AMD rows against this machine's baseline. A call runs a row's kernel
   behind a hold as many times as span 2 ms of the GPU and returns once they are
   done: its wall time is their span on the GPU and a submission's round trip.
   [bench_amd.exe gate] prints the device times themselves, [bench_amd.exe
   compare] the idle times of the rows that PyTorch can time too (compare.py),
   and [bench_amd.exe verify] each row's check alone.

   A row whose kernel nx.amd declines fails the suite, as a wrong result does;
   gate, compare and verify print it as declined and go on. *)

module S = Nx_amd_support
module D = Nx_array.Dtype

(* Runs [r] [chunk] seconds at a time, [cap] seconds at most, until two clocks
   read after consecutive chunks agree within 1%, [runs s] runs spanning [s]
   seconds. 20 ms of idle lower the R9700's clock to 1.7 GHz, where 10 ms of
   work leave it, from the 2.9-3.3 GHz that work keeps it at. *)
let settle g r ~runs ~chunk ~cap =
  let rec go last left =
    S.busy g r ~count:(runs chunk);
    let c = S.clock g in
    if left >= 2. *. chunk && Float.abs (c -. last) > 0.01 *. c then
      go c (left -. chunk)
  in
  go 0. cap

(* A row's first settle: 100 ms at a time, 1 s at most. Runs longer than [warm]
   run one at a time. *)
let warm = 0.1
let settle_row g r ~runs = settle g r ~runs ~chunk:warm ~cap:1.

(* The runs of [r] that span [s] seconds, [t] seconds a run: one for a run
   longer than {!warm}, else at least three. *)
let runs_of t s =
  if t > warm then 1 else Int.max 3 (int_of_float (Float.ceil (s /. t)))

(* The GPU time a suite call spans, so that its round trip, some 20 us, is a
   hundredth of it. *)
let call_span = 2e-3

(* A row's case. Its setup times one run to count the runs a call makes, then
   settles the clock on the row: thumper's calls follow one another with no idle
   in between, so the clock stays where the row keeps it. *)
let case (r : Rows.row) =
  let setup () =
    let g = S.gpu () in
    let p =
      try r.make g
      with Rows.Declined -> failwith (r.name ^ ": nx.amd declines it")
    in
    p.check ();
    let t = S.device_time g p.run ~count:1 in
    settle_row g p.run ~runs:(runs_of t);
    (g, p, Int.max 1 (int_of_float (Float.ceil (call_span /. t))))
  in
  Thumper.bench_with_setup ~setup r.name (fun (g, (p : Rows.prepared), count) ->
      S.device_time g p.run ~count)

(* The host's share of a call: 64 calls of the smallest contraction issued
   behind a hold, then their work once released. A call that finds nx.amd's
   caches warm allocates its launch, which dies with it. *)
let host =
  let bf16 = D.Any D.Bfloat16 and f32 = D.Any D.Float32 in
  let setup () =
    let g = S.gpu () in
    match Rows.plan g bf16 f32 bf16 1 1 1 with
    | _, _, _, run -> (g, run)
    | exception Rows.Declined ->
        failwith "issue/contract-64: nx.amd declines it"
  in
  Thumper.bench_with_setup ~setup "issue/contract-64" (fun (g, run) ->
      S.issue g ~count:64 run)

let suite () = host :: List.map case Rows.all

(* Gate *)

(* The gate: a table of each row's device time per run under two protocols, its
   rate, and its distance to the floor kernel that moves its bytes, timed in the
   same run, with the shader clock each protocol ended at.

   bench_amd.exe gate [PATTERN...]

   A row runs if its name starts with a PATTERN, or is a PATTERN that ends with
   $ but for it; every row runs without one. A row nx.amd declines prints so.
   First the GPU's clock settles on the row ({!settle}). Then:

   - idle: the median of 30 measurements, each of enough runs to span 10 ms of
   the GPU's time, after the GPU idled 20 ms and then ran the row unmeasured
   until two clocks read 50 ms apart agree within 1%, 300 ms at most, as
   compare.py times PyTorch, which runs 300 ms, reading no clock: the clock that
   idling lowered climbs back, where a 10 ms span alone would run at the idle
   clock; a run longer than 100 ms runs alone in a measurement, after one
   unmeasured run, measured as often as 3 s of runs allow, at least 5 times;

   - sustained: the median of 10 such measurements back to back, after 1 s of
   the row without pause, as a training loop runs it: a GPU busy without pause
   reaches its power cap and lowers its clock.

   The runs that settle the clock and the second of the row run without a hold:
   they measure nothing, and a hold waits for the host to make every call it
   covers.

   bench_amd.exe compare [PATTERN...]

   prints, for each selected row with a spec, its name, its idle time in
   microseconds, its floor kernel's idle time in microseconds ([-] for a row
   with none) and its spec, separated by tabs: what compare.py reads. A row
   nx.amd declines prints [declined] in place of its times. A row whose runs a
   hold cannot queue is timed from the host instead, each run submitted and
   waited for, and adds a fifth field, [wall]. *)

let strf = Printf.sprintf
let span = 10e-3
let idle = 20e-3

(* The row's unmeasured work after each idle: 50 ms at a time until the clock
   holds, 300 ms at most. A fixed 100 ms once left the bfloat16 peak at 1.84 GHz
   of the 2.93 it sustains. *)
let ramp g r ~runs = settle g r ~runs ~chunk:0.05 ~cap:0.3
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
  let runs = runs_of t in
  settle_row g r ~runs;
  let measure () = S.device_time g r ~count:(runs span) in
  let measurements =
    if long then Int.max 5 (Int.min 30 (int_of_float (long_span /. t))) else 30
  in
  let idle =
    median
      (List.init measurements (fun _ ->
           Unix.sleepf idle;
           ramp g r ~runs;
           measure ()))
  in
  (idle, S.clock g, runs, measure)

(* Seconds per run of [r] when idle and sustained, and the clock each ended
   at. *)
let time g r =
  let idle, idle_clock, runs, measure = time_idle g r in
  S.busy g r ~count:(runs 1.);
  let sustained = median (List.init 10 (fun _ -> measure ())) in
  ((idle, idle_clock), (sustained, S.clock g))

let rate (p : Rows.prepared) t =
  match p.work with
  | Bytes 0 -> ""
  | Bytes n -> strf "%7.1f GB/s" (float n /. t *. 1e-9)
  | Flops f -> strf "%7.1f TF/s" (f /. t *. 1e-12)

let row g (r : Rows.row) =
  match r.make g with
  | exception Rows.Declined -> Printf.printf "%-34s declined\n%!" r.name
  | p ->
      p.check ();
      let (t, c), (ts, cs) = time g p.run in
      let floor =
        match p.floor with
        | None -> ""
        | Some f ->
            let (tf, _), _ = time g f in
            strf "floor %9.2f us  x%.2f" (tf *. 1e6) (t /. tf)
      in
      Printf.printf
        "%-34s idle %10.2f us %s [%.0f MHz]  sustained %10.2f us %s [%.0f \
         MHz]  %s\n\
         %!"
        r.name (t *. 1e6) (rate p t) c (ts *. 1e6) (rate p ts) cs floor;
      Gc.full_major ()

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
  Printf.printf "# %s, %d compute units\n%!" (S.arch g) (S.compute_units g);
  List.iter (row g) (List.filter (selected patterns) Rows.compared)

(* Check *)

(* bench_amd.exe verify [PATTERN...] runs each selected row once and checks its
   result, timing nothing: the rows' correctness on a GPU no timing hold covers.
   It prints each row's name and [ok] or [declined], and fails at the first
   wrong result. *)
let verify patterns =
  let g = S.gpu () in
  let one (r : Rows.row) =
    match r.make g with
    | exception Rows.Declined -> Printf.printf "%s\tdeclined\n%!" r.name
    | p ->
        p.check ();
        Option.iter (S.run g) p.floor;
        Printf.printf "%s\tok\n%!" r.name;
        Gc.full_major ()
  in
  List.iter one (List.filter (selected patterns) Rows.compared)

(* Compare *)

let fst4 (a, _, _, _) = a

(* Seconds per run of [r] from the host: the median of 5 runs, each submitted
   and waited for, after one. It counts the host's share of the call, and is for
   a call that a hold cannot hold. *)
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
    | Some spec -> (
        let text = Rows.spec_text spec in
        match r.make g with
        | exception Rows.Declined ->
            Printf.printf "%s\tdeclined\t-\t%s\n%!" r.name text
        | p ->
            p.check ();
            (* A hold that runs out, as the host stalls on a loaded machine or a
               call submits more launches than the argument segment queues
               behind it, times the row once more, then from the host, with a
               note on stderr and a fourth field. *)
            let measure () = fst4 (time_idle g p.run) in
            let t, note =
              try (measure (), "")
              with Failure _ -> (
                try (measure (), "")
                with Failure m ->
                  Printf.eprintf "%s: %s; timed from the host\n%!" r.name m;
                  (wall g p.run, "\twall"))
            in
            let floor =
              match p.floor with
              | None -> "-"
              | Some f -> strf "%.3f" (fst4 (time_idle g f) *. 1e6)
            in
            Printf.printf "%s\t%.3f\t%s\t%s%s\n%!" r.name (t *. 1e6) floor text
              note;
            Gc.full_major ())
  in
  List.iter one (List.filter (selected patterns) Rows.compared)

let () =
  (* Looks for the GPU without opening it: the bench's rows open it in thumper's
     workers, after it forks them. *)
  if not (Sys.file_exists "/dev/kfd") then begin
    print_endline "nx.amd bench: no AMD GPU";
    exit 0
  end;
  match List.tl (Array.to_list Sys.argv) with
  | "gate" :: patterns -> gate patterns
  | "compare" :: patterns -> compare_rows patterns
  | "verify" :: patterns -> verify patterns
  | _ -> exit (Thumper.run "nx_amd" (suite ()))
