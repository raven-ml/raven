(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The AMD rows against this machine's baseline. A call runs a row's kernel
   behind a hold a fixed number of times, about 2 ms of the GPU ({!runs}), and
   returns once they are done: its wall time is their span on the GPU and a
   submission's round trip.
   [bench_amd.exe gate] prints the device times themselves, [bench_amd.exe
   compare] the idle times of the rows that PyTorch can time too (compare.py),
   and [bench_amd.exe verify] each row's check alone, which the bench alias runs
   once over the suite's rows before timing them: a trial checks nothing.

   A row whose kernel nx.amd declines fails the suite, as a wrong result fails
   verify; gate, compare and verify print it as declined and go on. *)

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

(* The runs a suite call of each row makes: about 2 ms of the R9700's time at
   a settled clock, from one run of each row on nonnormal (2026-11-16). They
   are fixed, and in the case's name ([.../x<runs>]), so that a call's work
   never follows a trial's measurement, and other runs are another case, of no
   baseline. *)
let runs =
  [
    ("floor/launch", 3774);
    ("floor/submission", 1134);
    ("floor/copy-256MB", 2);
    ("floor/read-256MB", 5);
    ("floor/wmma-bf16-peak", 1);
    ("contract/bfloat16-256x256x256", 239);
    ("contract/bfloat16-4096x4096x4096", 1);
    ("contract/float32-256x256x256", 153);
    ("contract/float32-4096x4096x4096", 1);
    ("contract/bfloat16-4096x4096x4096-kf", 1);
    ("contract/bfloat16-4096x4096x4096-ff", 1);
    ("contract/bfloat16-256x64x64x64", 88);
    ("contract/float32-256x64x64x64", 89);
    ("contract/float64-256x64x64x64", 11);
    ("contract/int8-4096x4096x4096", 2);
    ("contract/bfloat16-512x5120x2880", 10);
    ("contract/bfloat16-1x5120x2880", 61);
    ("contract/float8_e4m3fn-1x5120x2880", 71);
    ("contract/bfloat16-1x201088x2880", 1);
    ("contract/bfloat16-256x256x65536", 2);
    ("contract/bfloat16-2048x2048x65536", 1);
    ("contract/bfloat16-4096x1x4096", 32);
    ("contract/float32-1x5120x2880", 26);
    ("contract/float64-1024x1024x1024", 1);
    ("contract/float64-1x5120x2880", 10);
    ("contract/int32-1024x1024x1024", 5);
    ("contract/int8-1x5120x2880", 74);
    ("contract/int32-1x5120x2880", 26);
    ("contract/bfloat16-8x5120x2880", 62);
    ("contract/float32-20480x1x2880", 5);
    ("contract/float8_e4m3fn-4096x4096x4096", 2);
    ("contract/int4-4096x4096x4096", 1);
    ("contract/bfloat16-131072x16x16x16", 1);
    ("reduce/float32-sum-1024x1024-axes0.1", 155);
    ("reduce/float32-sum-8192x8192-axes0.1", 4);
    ("reduce/float32-sum-1024x1024-axes1", 128);
    ("reduce/float32-sum-1024x1024-axes0", 115);
    ("reduce/float32-max-1024x1024-axes1", 122);
    ("reduce/float64-sum-512x1024-axes0.1", 87);
    ("reduce/int32-sum-1024x1024-axes0.1", 111);
    ("reduce/int32-sum-256x256-axes0.1", 137);
    ("reduce/float32-sum-4096-axes0", 187);
    ("reduce/float32-sum-16384x64-axes1", 57);
    ("scan/float32-sum-1024x1024-axis1", 46);
    ("scan/float32-sum-1024x1024-axis0", 20);
    ("scan/float32-sum-1048576-axis0", 11);
    ("scan/float32-sum-67108864-axis0", 1);
    ("map/add-float32-1048576", 47);
    ("map/add-float32-67108864", 1);
    ("map/add-bfloat16-134217728", 1);
    ("map/bcast_add-float32-1024x1024", 43);
    ("map/t_add-float32-1024x1024", 40);
    ("map/fused4-float32-1048576", 47);
    ("map/pad9-float32-1x64x64x64", 10);
    ("reduce/argmax-float32-1024x1024", 20);
    ("reduce/logsumexp-float32-1024x1024", 9);
    ("reduce/logsumexp-float32-8192x8192", 1);
    ("reduce/moments-float32-1024x1024", 9);
    ("reduce/window2-float32-1x64x256x256", 3);
    ("reduce/sqdiff-float32-1024x1024", 15);
    ("scan/logcumsumexp-float32-1024x1024", 15);
    ("scan/sqcumsum-float32-1024x1024", 7);
    ("reduce/bfloat16-sum-2048x1024-axes1", 161);
    ("scan/float32-max-1024x1024-axis1", 42);
    ("map/nx-chain4-f32-67108864", 1);
    ("map/nx-add-f32-1048576", 45);
    ("sort/sort-f32-1x1M", 3);
    ("sort/sort-f32-1x16M", 1);
    ("sort/sort-i64-1x16M-range2^20", 1);
    ("sort/sort-u8-1x256M", 1);
    ("sort/sort-f32-65536x128", 3);
    ("sort/sort-f32-4096x4096-axis0", 1);
    ("sort/topk-f32-1x16M-k8", 2);
    ("sort/topk-f32-64x131072-k50", 4);
    ("sort/unique-i64-4Mx4-range16", 1);
    ("move/gather_emb_f32", 9);
    ("move/gather_rand_f32", 1);
    ("move/gather_rand64k_f32", 217);
    ("move/ranges_f32", 4);
    ("move/scatter_add_one_f32", 1);
    ("move/scatter_add_rand1m_f32", 1);
    ("move/scatter_add_emb_f32", 1);
    ("move/scatter_set_perm_f32", 1);
    ("move/scatter_max_rand4m_f32", 3);
    ("move/scatter_max_perm_f32", 1);
    ("move/assemble_cat_f32", 2);
    ("move/fold_col2im_f32", 1);
    ("fft/c2c-c64-4096x1024", 9);
    ("fft/c2c-c64-2^20", 16);
    ("fft/c2c-c64-2^24", 1);
    ("fft/c2c-c64-2048x2048-axes01", 4);
    ("fft/r2c-c64-4096x1024", 16);
    ("fft/c2r-c64-4096x513", 15);
    ("fft/c2c-c64-4096x1009", 4);
    ("fft/c2c-c64-2^20+7", 2);
    ("fft/c2c-c128-2^20", 4);
    ("linalg/lu/f32/4096x32x32", 3);
    ("linalg/qr/f32/4096x32x32", 1);
    ("linalg/chol/f32/4096x32", 11);
    ("linalg/solve/f32/4096x32x32", 12);
    ("linalg/lu/c64/1x1024x1024", 1);
    ("linalg/qr/f32/1x2048x1024", 1);
    ("linalg/eigh/f32/1024x32x32", 1);
    ("linalg/svd/f32/1024x32x32", 1);
    ("linalg/eig/f32/1024x32x32", 1);
  ]

let runs_of_row (r : Rows.row) =
  match List.assoc_opt r.name runs with
  | Some n -> n
  | None -> invalid_arg ("Bench_amd.runs: no runs for " ^ r.name)

(* A row's case. Its setup settles the clock on the row, 50 ms at a time and
   300 ms at most: thumper's calls follow one another with no idle in between,
   so the clock stays where the row keeps it, and the settle is the trial's
   warmup ({!config}). *)
let case (r : Rows.row) =
  let count = runs_of_row r in
  let setup () =
    let g = S.gpu () in
    let p =
      try r.make g
      with Rows.Declined -> failwith (r.name ^ ": nx.amd declines it")
    in
    settle g p.run
      ~runs:(runs_of (call_span /. float count))
      ~chunk:0.05 ~cap:0.3;
    (g, p)
  in
  Thumper.bench_with_setup ~setup
    (Printf.sprintf "%s/x%d" r.name count)
    (fun (g, (p : Rows.prepared)) -> S.device_time g p.run ~count)

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

(* thumper's protocol with no warmup time but its 3 batches: each case's setup
   settled the clock on its row. *)
let config = Thumper.Config.(warmup 0. default)

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
   Without a PATTERN it runs the suite's rows. It prints each row's name and
   [ok] or [declined], and fails at the first wrong result. *)
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
  match patterns with
  | [] -> List.iter one Rows.all
  | _ -> List.iter one (List.filter (selected patterns) Rows.compared)

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
  | _ -> exit (Thumper.run ~config "nx_amd" (suite ()))
