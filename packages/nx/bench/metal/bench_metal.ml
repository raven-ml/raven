(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* The Metal rows against this machine's baseline. A call of a case issues a
   fixed count of its row's calls, about 2 ms of memory traffic at 400 GB/s
   (fewer for a row whose call runs at a fraction of its floor, and at most 4096
   for one of few bytes), one after another into the same destinations, and
   waits for the last: its time is theirs, each call's submission included, and
   its name ends with [/xC], its C calls. Each row has a floor case,
   [floor/copy-N/xC], the copy of its N bytes by nx.metal, timed alike.
   [bench_metal.exe verify [PATTERN...]] makes each row, or each whose name
   contains a pattern, and checks its result against nx.cpu's, once: a loop's
   bits, a contraction's within its bound, then a contraction's GPU time against
   its estimate on the M1 Max, an attention's bits or within its bound. It times
   nothing else, and a trial's setup checks nothing. [bench_metal.exe compare]
   times the rows for compare.py, which times MLX and PyTorch alike and
   interleaves them (see Compare). [bench_metal.exe first] prints the times of a
   process's first calls, each to its end: opening GPU 0 with nx.metal's
   metallib, a float32 contraction of 64 rows, columns and terms, that call
   again, and a float16 one, of another family, with Metal's cache as the
   machine holds it: warm once the machine has run them. [bench_metal.exe cold]
   times one call of a contraction and one of a copy, each after the GPU idles,
   against the margin plan.ml states for a GPU's low clock. [bench_metal.exe
   launches [PATTERN...]] prints the launches of one call of each row, or of
   each whose name starts with a pattern.

   Calls are timed chained: each starts once the one before has ended, so a
   row's time is one call's, never several calls overlapping. rig.metal runs
   every command after the one before has completed, so nx.metal's calls are
   chained as issued. A bar from another library is timed alike, each of its
   calls reading the previous call's output.

   Cases are measured in this process, one after another: a worker forked
   without exec cannot reach the Metal compiler service, which makes the
   kernels' pipelines, so forked cases fail once a run has used it. A case that
   raises stops the run with its name and exception; a crash ends the process
   unnamed, and [verify], which prints each row it checked, names the row a
   crash follows. Each setup first collects the cases before it, so their
   buffers are freed before it makes its own. *)

module D = Nx_array.Dtype

(* Thumper exposes in-process measurement only through [Private], whose suites
   and command line its public ones cannot be mixed with. *)
module T = Thumper.Private

let device () =
  match Rig.open_ (Rig_metal.gpu 0) with
  | Ok d when Nx_metal.computes_on d -> d
  | Ok _ -> failwith "nx.metal does not compute on Metal GPU 0"
  | Error e -> failwith e

(* Calls a case issues for [bytes] a call, a call taking [slow] times its floor:
   about 2 ms at 400 GB/s, and at most [most_calls], since a call of few bytes
   takes the 5-15 µs of a submission whatever its bytes. *)
let most_calls = 4096

let count ?(slow = 1) bytes =
  Int.min most_calls (Int.max 1 (800_000_000 / bytes / slow))

(* A case's name: its row's and the calls it issues, so that a case issuing
   other calls is another case, of no baseline. *)
let name row count = Printf.sprintf "%s/x%d" row count

let calls run wait ~count =
  for _ = 1 to count do
    run ()
  done;
  wait ()

(* The device, once the previous cases' buffers are freed. *)
let fresh_device () =
  Gc.full_major ();
  device ()

let seconds () = float (Rig.Profile.now ()) *. 1e-9

(* An idle GPU takes about 1.2 s of work to reach its clock on an M1 Max, and a
   timed run starts idle: a row timed before then reads high (a 5.5 ms call read
   6.4 ms first). [warm ()] copies 64 MiB back to back for [warm_s] seconds, so
   that the first case meets the clock the later ones do; the clock stays up
   through the gaps between cases. A case's setup forces [warmed], so a run that
   measures nothing (thumper's [list], a refused command line) leaves the GPU
   idle. *)
let warm_s = 1.5

let warm () =
  let g = device () in
  let run, wait = Rows.copy_floor g (64 lsl 20) in
  let t0 = seconds () in
  while seconds () -. t0 < warm_s do
    calls run wait ~count:16
  done

let warmed = lazy (warm ())

let case (r : Rows.row) =
  let setup () =
    Lazy.force warmed;
    r.make ~check:false (fresh_device ())
  in
  let count = count ~slow:r.slow r.bytes in
  T.Bench.bench_with_setup ~setup (name r.name count)
    (fun (p : Rows.prepared) -> calls p.run p.wait ~count)

(* The floors: a copy of each count of bytes a row moves. *)
let floors =
  let bytes =
    List.sort_uniq compare (List.map (fun (r : Rows.row) -> r.bytes) Rows.all)
  in
  List.map
    (fun n ->
      let setup () =
        Lazy.force warmed;
        Rows.copy_floor (fresh_device ()) n
      in
      let count = count n in
      T.Bench.bench_with_setup ~setup
        (name (Printf.sprintf "floor/copy-%d" n) count)
        (fun (run, wait) -> calls run wait ~count))
    bytes

(* Checks the rows whose names contain one of [patterns], every row without any,
   each once the rows before it are collected, so that their buffers are freed
   before it makes its own. *)
let verify patterns =
  let g = device () in
  let chosen (r : Rows.row) =
    patterns = []
    || List.exists
         (fun p ->
           let n = String.length p in
           let rec at i =
             i + n <= String.length r.name
             && (String.sub r.name i n = p || at (i + 1))
           in
           at 0)
         patterns
  in
  List.iter
    (fun (r : Rows.row) ->
      Gc.full_major ();
      ignore (r.make ~check:true g : Rows.prepared);
      Printf.printf "%s: checked\n%!" r.name)
    (List.filter chosen Rows.all)

(* Compare *)

(* bench_metal.exe compare --specs [PATTERN...] prints each selected row's name,
   bytes and spec, separated by tabs, and times nothing. A row is selected if
   its name starts with a PATTERN, or is a PATTERN that ends with $ but for it;
   every row is without one.

   bench_metal.exe compare ROW row|floor makes the row's call, or its floor's
   copy, and prints [ready]. Then, for each line [COUNT] it reads, it issues
   COUNT calls back to back from an idle GPU and prints [s] and their seconds,
   separated by a tab, until its input ends. compare.py holds the protocol and
   interleaves these measurements with its references'. *)

type side = Row | Floor

let selected patterns (r : Rows.row) =
  patterns = []
  || List.exists
       (fun p ->
         match String.ends_with ~suffix:"$" p with
         | true -> String.sub p 0 (String.length p - 1) = r.name
         | false -> String.starts_with ~prefix:p r.name)
       patterns

let specs patterns =
  List.iter
    (fun (r : Rows.row) -> Printf.printf "%s\t%d\t%s\n%!" r.name r.bytes r.spec)
    (List.filter (selected patterns) Rows.all)

let serve name side =
  let r =
    match List.find_opt (fun (r : Rows.row) -> r.name = name) Rows.all with
    | Some r -> r
    | None -> failwith ("bench_metal compare: no row " ^ name)
  in
  let g = device () in
  let run, wait =
    match side with
    | Row ->
        let p = r.make ~check:false g in
        (p.run, p.wait)
    | Floor -> Rows.copy_floor g r.bytes
  in
  print_endline "ready";
  let rec loop () =
    match In_channel.input_line stdin with
    | None -> ()
    | Some line ->
        let count = int_of_string line in
        let t0 = seconds () in
        calls run wait ~count;
        Printf.printf "s\t%.9f\n%!" (seconds () -. t0);
        loop ()
  in
  loop ()

(* bench_metal.exe launches [PATTERN...] prints each selected row's name and the
   launches one of its calls plans, as {!Nx_metal.Plan.traced} records them,
   separated by a tab. It times nothing. Each row is made once the rows before
   it are collected, as in {!verify}. *)
let launches patterns =
  let g = device () in
  List.iter
    (fun (r : Rows.row) ->
      Gc.full_major ();
      let p = r.make ~check:false g in
      let (), ls = Nx_metal.Plan.traced p.run in
      p.wait ();
      Printf.printf "%s\t%d\n%!" r.name (List.length ls))
    (List.filter (selected patterns) Rows.all)

(* This process's first calls, of zeroed operands that no kernel wrote. *)
let first () =
  let ms t = float (Rig.Profile.now () - t) /. 1e6 in
  let t = Rig.Profile.now () in
  let g = device () in
  Printf.printf "open: %.1f ms\n%!" (ms t);
  let timed what dt =
    let c =
      { Rows.batch = 1; m = 64; n = 64; k = 64; dt; db = dt; nk = false }
    in
    let zeros shape =
      let (D.Any d) = dt in
      let x = Nx_array.create g d (Iarray.of_array shape) in
      Rig.Buffer.fill (Nx_array.buffer x) '\000';
      Nx_array.Any x
    in
    let sa, sb, sy = Rows.shapes c in
    let a = zeros sa and b = zeros sb and dst = zeros sy in
    let t = Rig.Profile.now () in
    Rows.done_ what (Rows.contract_on (module Nx_metal) c dst a b);
    Rows.wait_all [ dst ] ();
    Printf.printf "%s: %.1f ms\n%!" what (ms t)
  in
  timed "first float32 contraction" (D.Any D.Float32);
  timed "second float32 contraction" (D.Any D.Float32);
  timed "first float16 contraction" (D.Any D.Float16)

let () =
  if
    Sys.os_type <> "Unix"
    || not (Sys.file_exists "/System/Library/Frameworks/Metal.framework")
  then begin
    print_endline "nx.metal bench: no Metal";
    exit 0
  end;
  match List.tl (Array.to_list Sys.argv) with
  | "verify" :: patterns -> verify patterns
  | [ "first" ] -> first ()
  | [ "cold" ] -> Rows.low_clock (device ())
  | "compare" :: "--specs" :: patterns -> specs patterns
  | "launches" :: patterns -> launches patterns
  | [ "compare"; name; "row" ] -> serve name Row
  | [ "compare"; name; "floor" ] -> serve name Floor
  | _ ->
      let config = T.Config.(fork false default) in
      exit
        (T.Cli.main ~config ~argv:Sys.argv ~name:"nx_metal"
           (List.map case Rows.all @ floors))
