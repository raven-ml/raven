(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* What a user calls on a store: folding a long run, following its last records,
   listing many runs cold of their logs and again with a previous listing, and
   logging a record.

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

let suite () =
  let run = long_run () in
  let limit = before_tail run in
  let store = many_runs () in
  let listing = Store.runs store in
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
  exit
    (Thumper.run "munin" ~config ~argv
       ~budgets:
         [
           Thumper.Budget.no_slower_than 0.05;
           Thumper.Budget.no_more_alloc_than 0.01;
         ]
       (suite ()))
