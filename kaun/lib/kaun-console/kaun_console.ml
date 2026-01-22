(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ───── Data Aggregation ───── *)

let latest_values events =
  let table = Hashtbl.create 16 in
  List.iter
    (function
      | Event.Scalar s ->
          let update =
            match Hashtbl.find_opt table s.tag with
            | Some (prev_step, _, _) -> s.step > prev_step
            | None -> true
          in
          if update then Hashtbl.replace table s.tag (s.step, s.epoch, s.value)
      | Event.Unknown _ | Event.Malformed _ -> ())
    events;
  Hashtbl.fold
    (fun tag (step, epoch, value) acc -> (tag, step, epoch, value) :: acc)
    table []
  |> List.sort (fun (a, _, _, _) (b, _, _, _) -> compare a b)

(* Get the latest epoch from events *)
let latest_epoch events =
  List.fold_left
    (fun acc event ->
      match event with
      | Event.Scalar s -> (
          match s.epoch with
          | Some e -> max acc e
          | None -> acc)
      | Event.Unknown _ | Event.Malformed _ -> acc)
    0 events

(* ───── Terminal Helpers ───── *)

let clear_screen () = print_string "\027[2J\027[H"

let render ~run_id ~events =
  clear_screen ();
  Printf.printf "Kaun Console\n";
  Printf.printf "════════════\n\n";
  Printf.printf "Run: %s\n" run_id;

  let epoch = latest_epoch events in
  if epoch > 0 then Printf.printf "Epoch: %d\n" epoch;
  Printf.printf "\n";

  let latest = latest_values events in

  if latest = [] then Printf.printf "  Waiting for metrics...\n"
  else (
    Printf.printf "Metrics:\n";
    List.iter
      (fun (tag, step, epoch, value) ->
        let epoch_str =
          match epoch with
          | Some e when e > 0 -> Printf.sprintf ", epoch %d" e
          | _ -> ""
        in
        Printf.printf "  %-30s %8.4f  (step %d%s)\n" tag value step epoch_str)
      latest);

  Printf.printf "\n%!";
  flush stdout

(* ───── Public API ───── *)

let run ?(base_dir = "./runs") ?experiment:_ ?tags:_ ?runs () =
  match runs with
  | Some [ run_id ] ->
      let run_dir = Filename.concat base_dir run_id in
      let events_path = Filename.concat run_dir "events.jsonl" in

      (* Create event reader for incremental reading *)
      let reader = Event_reader.create ~file_path:events_path in
      let all_events = ref [] in

      (* Poll and re-render at 2Hz with incremental reading *)
      while true do
        (* Read only new events since last read *)
        let new_events = Event_reader.read_new reader in

        (* Accumulate events *)
        all_events := !all_events @ new_events;

        (* Render with accumulated events *)
        render ~run_id ~events:!all_events;
        Unix.sleepf 0.5
      done;

      (* Cleanup (unreachable but good practice) *)
      Event_reader.close reader
  | _ ->
      (* Multi-run view and filtering not yet implemented *)
      Printf.printf "kaun-console: please specify a single run\n%!"
