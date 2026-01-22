(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let clear_screen () =
  (* ANSI clear + cursor home *)
  print_string "\027[2J\027[H"

let render ~run_id (store : Metric_store.t) =
  clear_screen ();
  Printf.printf "Kaun Console\n";
  Printf.printf "════════════\n\n";
  Printf.printf "Run: %s\n" run_id;

  (match Metric_store.latest_epoch store with
   | None -> ()
   | Some e -> Printf.printf "Epoch: %d\n" e);

  Printf.printf "\n";

  let latest = Metric_store.latest_metrics store in
  if latest = [] then
    Printf.printf "  Waiting for metrics...\n"
  else (
    Printf.printf "Metrics:\n";
    List.iter
      (fun (tag, (m : Metric_store.metric)) ->
        let epoch_str =
          match m.epoch with
          | None -> ""
          | Some e -> Printf.sprintf ", epoch %d" e
        in
        Printf.printf "  %-30s %8.4f  (step %d%s)\n"
          tag m.value m.step epoch_str)
      latest
  );

  Printf.printf "\n(Press Ctrl-C to quit)\n%!";
  flush stdout

let run ?(base_dir = "./runs") ?experiment:_ ?tags:_ ?runs () =
  match runs with
  | Some [ run_id ] ->
      let run_dir = Filename.concat base_dir run_id in
      let events_path = Filename.concat run_dir "events.jsonl" in

      let reader = Event_reader.create ~file_path:events_path in
      let store = Metric_store.create () in

      (* Allow graceful cleanup on Ctrl-C *)
      Sys.catch_break true;

      (try
         while true do
           let new_events = Event_reader.read_new reader in
           Metric_store.update store new_events;
           render ~run_id store;
           Unix.sleepf 0.5
         done
       with
       | Sys.Break ->
           Event_reader.close reader;
           Printf.printf "\nExiting.\n%!"
       | exn ->
           (* Best-effort cleanup on unexpected errors. *)
           Event_reader.close reader;
           raise exn)
  | _ ->
      (* Multi-run view and filtering not yet implemented *)
      Printf.printf "kaun-console: please specify a single run\n%!"
