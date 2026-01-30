(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Command-line interface for kaun-console. *)

open Kaun_runlog

let () =
  (* Default values *)
  let base_dir = ref "./runs" in
  let runs = ref [] in
  let experiment = ref None in
  let tags = ref [] in

  (* Command-line argument specifications *)
  let specs =
    [
      ( "--base-dir",
        Arg.Set_string base_dir,
        "DIR Directory containing training runs (default: ./runs)" );
      ( "--runs",
        Arg.String (fun s -> runs := s :: !runs),
        "ID Specific run ID to monitor" );
      ( "--experiment",
        Arg.String (fun s -> experiment := Some s),
        "NAME Filter runs by experiment name" );
      ( "--tag",
        Arg.String (fun s -> tags := s :: !tags),
        "TAG Filter runs by tag (can be specified multiple times)" );
    ]
  in

  let usage_msg = "Usage: kaun-console [options]\n\nMonitor Kaun training runs." in

  (* Parse arguments *)
  Arg.parse specs
    (fun arg ->
      Printf.eprintf "Error: Unknown argument '%s'\n" arg;
      Arg.usage specs usage_msg;
      exit 1)
    usage_msg;

  (* Determine which run to monitor *)
  let run_id =
    match !runs with
    | explicit_id :: _ ->
        (* User provided explicit run ID *)
        explicit_id
    | [] ->
        (* Auto-discover latest run with optional filtering *)
        let all_runs = Kaun_runlog.discover ~base_dir:!base_dir () in
        let filtered =
          all_runs
          |> List.filter (fun run ->
                 match !experiment with
                 | None -> true
                 | Some exp -> Run.experiment_name run = Some exp)
          |> List.filter (fun run ->
                 match List.rev !tags with
                 | [] -> true
                 | required_tags ->
                     List.for_all
                       (fun t -> List.mem t (Run.tags run))
                       required_tags)
        in
        (match filtered with
        | run :: _ ->
            Printf.printf "Auto-discovered run: %s\n" (Run.run_id run);
            (match Run.experiment_name run with
            | Some exp -> Printf.printf "  Experiment: %s\n" exp
            | None -> ());
            if Run.tags run <> [] then
              Printf.printf "  Tags: %s\n" (String.concat ", " (Run.tags run));
            Printf.printf "\n%!";
            Run.run_id run
        | [] ->
            Printf.eprintf "Error: No runs found in %s\n" !base_dir;
            (match !experiment with
            | Some exp ->
                Printf.eprintf "  (filtered by experiment: %s)\n" exp
            | None -> ());
            if List.length !tags > 0 then
              Printf.eprintf "  (filtered by tags: %s)\n"
                (String.concat ", " (List.rev !tags));
            exit 1)
  in

  (* Launch console with discovered/explicit run_id *)
  try Kaun_console.run ~base_dir:!base_dir ~runs:[ run_id ] ()
  with
  | Failure msg ->
      Printf.eprintf "Error: %s\n" msg;
      exit 1
  | exn ->
      Printf.eprintf "Error: %s\n" (Printexc.to_string exn);
      exit 1
