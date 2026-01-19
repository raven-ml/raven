(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ───── Event Types ───── *)

type scalar = { step : int; epoch : int; tag : string; value : float }
type event = Scalar of scalar | Unknown of Yojson.Basic.t

(* ───── JSON Parsing Helpers ───── *)

let get_string key fields =
  match List.assoc_opt key fields with
  | Some (`String s) -> Some s
  | _ -> None

let get_int key fields =
  match List.assoc_opt key fields with
  | Some (`Int i) -> Some i
  | _ -> None

let get_float key fields =
  match List.assoc_opt key fields with
  | Some (`Float f) -> Some f
  | Some (`Int i) -> Some (float_of_int i)
  | _ -> None

(* ───── Event Parsing ───── *)

let parse_event json =
  match json with
  | `Assoc fields -> (
      match get_string "type" fields with
      | Some "scalar" -> (
          match
            ( get_int "step" fields,
              get_string "tag" fields,
              get_float "value" fields )
          with
          | Some step, Some tag, Some value ->
              let epoch = Option.value ~default:0 (get_int "epoch" fields) in
              Scalar { step; epoch; tag; value }
          | _ -> Unknown json)
      | _ -> Unknown json)
  | _ -> Unknown json

let parse_event_line line =
  match Yojson.Basic.from_string line with
  | json -> Some (parse_event json)
  | exception Yojson.Json_error _ -> None

(* ───── Event Loading ───── *)

let load_events ~run_dir =
  let path = Filename.concat run_dir "events.jsonl" in
  if not (Sys.file_exists path) then []
  else
    match open_in path with
    | ic ->
        let rec read_lines acc =
          match input_line ic with
          | line ->
              let acc = Option.fold ~none:acc ~some:(fun e -> e :: acc) (parse_event_line line) in
              read_lines acc
          | exception End_of_file ->
              close_in ic;
              List.rev acc
        in
        read_lines []
    | exception _ -> []

(* ───── Data Aggregation ───── *)

let latest_values events =
  let table = Hashtbl.create 16 in
  List.iter
    (function
      | Scalar s ->
          let update =
            match Hashtbl.find_opt table s.tag with
            | Some (prev_step, _, _) -> s.step > prev_step
            | None -> true
          in
          if update then Hashtbl.replace table s.tag (s.step, s.epoch, s.value)
      | Unknown _ -> ())
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
      | Scalar s -> max acc s.epoch
      | Unknown _ -> acc)
    0 events

(* ───── Terminal Helpers ───── *)

let clear_screen () = print_string "\027[2J\027[H"

let render ~run_id ~run_dir =
  clear_screen ();
  Printf.printf "Kaun Console\n";
  Printf.printf "════════════\n\n";
  Printf.printf "Run: %s\n" run_id;

  let events = load_events ~run_dir in
  let epoch = latest_epoch events in
  if epoch > 0 then Printf.printf "Epoch: %d\n" epoch;
  Printf.printf "\n";

  let latest = latest_values events in

  if latest = [] then Printf.printf "  Waiting for metrics...\n"
  else (
    Printf.printf "Metrics:\n";
    List.iter
      (fun (tag, step, epoch, value) ->
        let epoch_str = if epoch > 0 then Printf.sprintf ", epoch %d" epoch else "" in
        Printf.printf "  %-30s %8.4f  (step %d%s)\n" tag value step epoch_str)
      latest);

  Printf.printf "\n%!";
  flush stdout

(* ───── Public API ───── *)

let run ?(base_dir = "./runs") ?experiment:_ ?tags:_ ?runs () =
  match runs with
  | Some [ run_id ] ->
      let run_dir = Filename.concat base_dir run_id in
      (* Poll and re-render at 2Hz. Future: use inotify/kqueue for efficiency. *)
      while true do
        render ~run_id ~run_dir;
        Unix.sleepf 0.5
      done
  | _ ->
      (* Multi-run view and filtering not yet implemented *)
      Printf.printf "kaun-console: please specify a single run\n%!"
