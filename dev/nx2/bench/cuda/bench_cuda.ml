(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The CUDA rows against this machine's baseline. A call runs a row's kernel a
   fixed count of times behind a hold and returns once they are done: its wall
   time is their span on the GPU and a submission's round trip. [bench_cuda.exe
   gate] prints the device times themselves. *)

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

(* The binding's host share of a call: the plan of a contraction alone, and the
   enqueue of 64 launches of the smallest one behind a hold, its wait outside
   the timed call. *)
let host =
  let bf16 = D.Any D.Bfloat16 and f32 = D.Any D.Float32 in
  let planner () =
    let a, b, y, _ = Rows.plan (S.gpu ()) bf16 f32 bf16 4096 4096 4096 in
    S.planner ~a ~b ~y
      ~batch:[ (0, 0) ]
      ~contracting:[ (2, 2) ]
      ~acc:(D.code D.Float32) ()
  in
  let plan =
    Thumper.bench_with_setup ~setup:planner "plan/contract-bf16-4096" (fun p ->
        p ())
  in
  let setup () =
    let g = S.gpu () in
    let _, _, _, run = Rows.plan g bf16 f32 bf16 1 1 1 in
    (g, run)
  in
  let fill =
    Thumper.bench_with_setup ~setup "fill/contract-64" (fun (g, run) ->
        S.enqueue g ~count:64 run)
  in
  [ plan; fill ]

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
   the GPU's time, after the GPU idled 20 ms, as the reference's ref.py times
   PyTorch, so that both run in the same power state; - sustained: the median of
   10 such measurements back to back, after 1 s of the row without pause, as a
   training loop runs it: a GPU busy without pause reaches its power cap and
   lowers its clock. *)

let strf = Printf.sprintf
let span = 10e-3
let warm = 0.1
let idle = 20e-3

let median xs =
  let a = Array.of_list xs in
  Array.sort compare a;
  a.(Array.length a / 2)

(* Seconds per run of [r] when idle and sustained, and the clock each ended
   at. *)
let time g r =
  let t = S.device_time g r ~count:3 in
  let runs s = Int.max 3 (int_of_float (Float.ceil (s /. t))) in
  let rec settle last tries =
    ignore (S.device_time g r ~count:(runs warm));
    let c = S.sm_clock g in
    if tries > 1 && Float.abs (c -. last) > 0.01 *. c then settle c (tries - 1)
  in
  settle 0. 10;
  let measure () = S.device_time g r ~count:(runs span) in
  let idle =
    median
      (List.init 30 (fun _ ->
           Unix.sleepf idle;
           measure ()))
  in
  let idle_clock = S.sm_clock g in
  ignore (S.device_time g r ~count:(runs 1.));
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

let gate patterns =
  let selected (r : Rows.row) =
    patterns = []
    || List.exists
         (fun p ->
           match String.ends_with ~suffix:"$" p with
           | true -> String.sub p 0 (String.length p - 1) = r.name
           | false -> String.starts_with ~prefix:p r.name)
         patterns
  in
  let g = S.gpu () in
  let kernels, bytes = S.library_size g in
  Printf.printf "# %s, %d SMs; nx.cuda: %d kernels, contract's, %d bytes\n%!"
    (S.arch g) (S.sms g) kernels bytes;
  List.iter (row g) (List.filter selected Rows.all)

let () =
  (* Looks for the GPU without opening it: the bench's rows open it in thumper's
     workers, after it forks them. *)
  if not (Sys.file_exists "/dev/nvidiactl") then begin
    print_endline "nx.cuda bench: no NVIDIA GPU";
    exit 0
  end;
  S.hold_gpu ();
  match List.tl (Array.to_list Sys.argv) with
  | "gate" :: patterns -> gate patterns
  | _ -> exit (Thumper.run "nx_cuda" (host @ List.map case Rows.all))
