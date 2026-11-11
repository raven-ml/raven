(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What a user calls on a store: folding a long run, following its last records,
   listing many runs cold of their logs and again with a previous listing, and
   logging a record; and what the dashboard does on a key, a resize and a
   frame.

   The stores are written once, before any row, in a temporary directory removed
   at exit; each row's worker reads them. A fold costs the bytes of its log, so
   the fold row's 10^5 steps of 12 keys shows a regression that 10^6 steps
   would, within the suite's time. [--tiny] shrinks every input to see that the
   rows work, with no timing worth keeping. *)

open Munin

let tiny = Array.exists (( = ) "--tiny") Sys.argv

let argv =
  Array.of_list (List.filter (( <> ) "--tiny") (Array.to_list Sys.argv))

let steps = if tiny then 1_000 else 100_000
let tail = if tiny then 100 else 1_000
let runs = if tiny then 50 else 10_000
let samples = if tiny then 10_000 else 1_000_000
let keys = List.init 12 (fun i -> Printf.sprintf "train/k%02d" i)

(* A provenance of its own keeps git out of every start. *)
let provenance =
  {
    Provenance.command = [ "bench_munin" ];
    cwd = "/";
    hostname = None;
    pid = 0;
    ocaml = Sys.ocaml_version;
    raven = "bench";
    os = "bench";
    arch = "bench";
    cpu = None;
    git = None;
    seeds = [];
    env = [];
  }

let rec rm_rf p =
  match Unix.lstat p with
  | exception Unix.Unix_error _ -> ()
  | { Unix.st_kind = Unix.S_DIR; _ } ->
      Array.iter (fun e -> rm_rf (Filename.concat p e)) (Sys.readdir p);
      Unix.rmdir p
  | _ -> Sys.remove p

(* Workers are forked from this process and exit through [at_exit] too: only the
   process that made a store removes it. *)
let temp_store name =
  let root = Filename.temp_dir ("munin-bench-" ^ name) "" in
  let owner = Unix.getpid () in
  at_exit (fun () -> if Unix.getpid () = owner then rm_rf root);
  Store.open_ ~root ()

(* A finished run of [steps] records of 12 keys. *)
let long_run () =
  let store = temp_store "long" in
  let s = Session.start ~store ~provenance ~experiment:"long" () in
  let handles = List.map (fun k -> Session.metric s ~goal:`Minimize k) keys in
  for step = 0 to steps - 1 do
    Session.log_metrics s ~step
      (List.mapi (fun i m -> (m, Float.of_int (step + i) /. 7.)) handles)
  done;
  Session.finish s;
  Option.get (Store.run store (Session.id s))

(* [before_tail run] is the byte offset of [run]'s log where its last [tail]
   records, the end included, begin. *)
let before_tail run =
  let log =
    In_channel.with_open_bin
      (Filename.concat (Run.dir run) "events.jsonl")
      In_channel.input_all
  in
  let rec back i n =
    if n = 0 then i + 1 else back (String.rindex_from log (i - 1) '\n') (n - 1)
  in
  back (String.length log - 1) tail

(* [age t p] sets the times of [p] and everything under it to [t]. *)
let rec age t p =
  if Sys.is_directory p then
    Array.iter (fun e -> age t (Filename.concat p e)) (Sys.readdir p);
  Unix.utimes p t t

(* A store of [runs] finished runs: one run, its directory copied. Its files are
   aged a minute, past any file system's time resolution, as a dashboard's store
   is from its second refresh on: a listing then takes a run's tags and notes
   from the previous one after a stat of each. *)
let many_runs () =
  let store = temp_store "many" in
  let s =
    Session.start ~store ~provenance ~experiment:"many" ~tags:[ "a" ] ()
  in
  let m = Session.metric s "loss" in
  for step = 0 to 9 do
    Session.log m ~step (Float.of_int step)
  done;
  Session.finish s;
  let runs_dir = Filename.concat (Store.root store) "runs" in
  let src = Filename.concat runs_dir (Session.id s) in
  let rec copy src dst =
    if Sys.is_directory src then begin
      Unix.mkdir dst 0o755;
      Array.iter
        (fun e -> copy (Filename.concat src e) (Filename.concat dst e))
        (Sys.readdir src)
    end
    else
      Out_channel.with_open_bin dst (fun oc ->
          Out_channel.output_string oc
            (In_channel.with_open_bin src In_channel.input_all))
  in
  for i = 1 to runs - 1 do
    copy src
      (Filename.concat runs_dir (Printf.sprintf "%s-%05d" (Session.id s) i))
  done;
  age (Unix.gettimeofday () -. 60.) runs_dir;
  store

(* Five finished runs of [samples] steps of four goal metrics, an experiment
   whose charts draw four metrics. The four runs after the first link its
   files, so that the store holds the samples once. *)
let five_runs () =
  let store = temp_store "five" in
  let s = Session.start ~store ~provenance ~experiment:"five" () in
  let goals = List.filteri (fun i _ -> i < 4) keys in
  let handles = List.map (fun k -> Session.metric s ~goal:`Minimize k) goals in
  for step = 0 to samples - 1 do
    Session.log_metrics s ~step
      (List.mapi (fun i m -> (m, Float.of_int (step + i) /. 7.)) handles)
  done;
  Session.finish s;
  let runs_dir = Filename.concat (Store.root store) "runs" in
  let src = Filename.concat runs_dir (Session.id s) in
  let rec link src dst =
    if Sys.is_directory src then begin
      Unix.mkdir dst 0o755;
      Array.iter
        (fun e -> link (Filename.concat src e) (Filename.concat dst e))
        (Sys.readdir src)
    end
    else Unix.link src dst
  in
  for i = 1 to 4 do
    link src (Filename.concat runs_dir (Printf.sprintf "%s-%d" (Session.id s) i))
  done;
  store

(* The dashboard *)

(* A session's inputs. [press k] is the key [k] as the loop takes it: the
   update, then the frame drawn; the reads it asks for are dropped, so a
   press times none. [read k] is the key, then each read it asks for and
   its result applied, and [resize] a resize, then the same. *)
type screen = {
  press : Matrix.Input.Keystroke.key -> unit;
  read : Matrix.Input.Keystroke.key -> unit;
  resize : cols:int -> rows:int -> unit;
}

(* [on_screen app f] is [f screen], in a session of [app] on a terminal of 160
   by 48 once its first reads are done. *)
let on_screen app f =
  let module T = Matrix_test in
  let term = T.Terminal.create ~cols:160 ~rows:48 () in
  let held = Queue.create () in
  Mosaic_test.run term app
    ~process_perform:(fun job -> Queue.push job held)
    ~quiesce:ignore
  @@ fun s ->
  let reads () =
    T.settle s;
    while not (Queue.is_empty held) do
      (Queue.pop held) ();
      T.settle s
    done
  in
  reads ();
  let press k =
    T.input s (T.Terminal.key term k);
    T.settle s;
    Queue.clear held
  in
  let read k =
    T.input s (T.Terminal.key term k);
    reads ()
  in
  let resize ~cols ~rows =
    T.resize s ~cols ~rows;
    reads ()
  in
  f { press; read; resize }

let char c = Matrix.Input.Keystroke.Char (Uchar.of_char c)

(* The first run with the four others overlaid, their reads applied. *)
let overlay_four screen =
  for _ = 1 to 4 do
    screen.read Matrix.Input.Keystroke.Down;
    screen.read (char ' ')
  done;
  screen.read (char 'g')

let suite ~run ~store ~on_run ~on_runs ~on_five ~on_others =
  let limit = before_tail run in
  let listing = Store.runs store in
  (* Down and back up, so that each call leaves the screen as it found it. *)
  let down_up press =
    press Matrix.Input.Keystroke.Down;
    press Matrix.Input.Keystroke.Up
  in
  [
    Thumper.bench (Printf.sprintf "fold %d steps of 12 keys" steps) (fun () ->
        History.read run);
    (* Statistics alone, so that each call follows the same history and costs
       its new bytes; a follow that keeps series copies them on a second follow
       of one history. *)
    Thumper.bench_with_setup
      ~setup:(fun () -> History.read ~limit ~series:[] run)
      (Printf.sprintf "follow the last %d records" tail)
      History.follow;
    Thumper.bench (Printf.sprintf "list %d runs" runs) (fun () ->
        Store.runs store);
    Thumper.bench (Printf.sprintf "list %d runs again, unchanged" runs)
      (fun () -> Store.runs ~prev:listing store);
    (* A key costs what is on screen, whatever the series' lengths and the
       store's size: a regression that brings per-sample or per-run work back to
       the loop shows at these sizes. *)
    Thumper.bench
      (Printf.sprintf "dashboard: down, up on Run of %d steps of 12 keys" steps)
      (fun () -> down_up on_run.press);
    Thumper.bench (Printf.sprintf "dashboard: down, up on Runs of %d runs" runs)
      (fun () -> down_up on_runs.press);
    (* A key that changes what the screen wants costs its reads, from the
       histories held and the last listing: one that lists the store or
       reads every run again shows at this size. *)
    Thumper.bench
      (Printf.sprintf "dashboard: down, up on Runs of %d runs, reads applied" runs)
      (fun () -> down_up on_runs.read);
    (* A resize changes every chart's dot columns, and each run's series is
       reduced again: the cursor's run and four overlays, as a comparison
       draws them. *)
    Thumper.bench
      (Printf.sprintf "dashboard: resize, 4 charts of 5 runs of %d samples" samples)
      (fun () ->
        on_five.resize ~cols:150 ~rows:44;
        on_five.resize ~cols:160 ~rows:48);
    (* Every run of the experiment drawn in grey: a frame draws a line of
       each, as many as the experiment has. *)
    Thumper.bench
      (Printf.sprintf "dashboard: down, up at charts with r over %d runs" runs)
      (fun () -> down_up on_others.press);
    Thumper.bench_with_setup
      ~setup:(fun () ->
        (* Made in the worker, which removes it in the teardown. *)
        let root = Filename.temp_dir "munin-bench-log" "" in
        let store = Store.open_ ~root () in
        let s = Session.start ~store ~provenance ~experiment:"log" () in
        let handles = List.map (fun k -> Session.metric s k) keys in
        (root, s, List.map (fun m -> (m, 0.5)) handles))
      ~teardown:(fun (root, s, _) ->
        Session.finish s;
        rm_rf root)
      "log a record of 12 keys"
      (fun (_, s, pairs) -> Session.log_metrics s ~step:1 pairs);
  ]

let config = Thumper.Config.(default |> deadline 60.)

let () =
  let run = long_run () in
  let store = many_runs () in
  let five = five_runs () in
  let run_app =
    Munin_tui.app ~clock:Unix.gettimeofday
      (Store.open_ ~root:(Filename.dirname (Filename.dirname (Run.dir run))) ())
      (Munin_tui.Run (Some (Run.id run)))
  in
  let runs_app () = Munin_tui.app ~clock:Unix.gettimeofday store Munin_tui.Runs in
  let five_app = Munin_tui.app ~clock:Unix.gettimeofday five Munin_tui.Runs in
  let code =
    on_screen run_app @@ fun on_run ->
    on_screen (runs_app ()) @@ fun on_runs ->
    on_screen five_app @@ fun on_five ->
    overlay_four on_five;
    on_screen (runs_app ()) @@ fun on_others ->
    on_others.read (char 'r');
    Thumper.run "munin" ~config ~argv
      ~budgets:
        [
          Thumper.Budget.no_slower_than 0.05;
          Thumper.Budget.no_more_alloc_than 0.01;
        ]
      (suite ~run ~store ~on_run ~on_runs ~on_five ~on_others)
  in
  exit code
