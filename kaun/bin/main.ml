(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Command-line interface for Kaun. *)

open Kaun_runlog

(* ───── Console ───── *)

let discover_run ~base_dir ~experiment ~tags =
  let all_runs = Kaun_runlog.discover ~base_dir () in
  let filtered =
    all_runs
    |> List.filter (fun run ->
           match experiment with
           | None -> true
           | Some exp -> Run.experiment_name run = Some exp)
    |> List.filter (fun run ->
           match tags with
           | [] -> true
           | required_tags ->
               List.for_all (fun t -> List.mem t (Run.tags run)) required_tags)
  in
  match filtered with
  | run :: _ ->
      Printf.printf "Auto-discovered run: %s\n" (Run.run_id run);
      (match Run.experiment_name run with
      | Some exp -> Printf.printf "  Experiment: %s\n" exp
      | None -> ());
      if Run.tags run <> [] then
        Printf.printf "  Tags: %s\n" (String.concat ", " (Run.tags run));
      Printf.printf "\n%!";
      Some (Run.run_id run)
  | [] ->
      Printf.eprintf "Error: No runs found in %s\n" base_dir;
      (match experiment with
      | Some exp -> Printf.eprintf "  (filtered by experiment: %s)\n" exp
      | None -> ());
      if tags <> [] then
        Printf.eprintf "  (filtered by tags: %s)\n"
          (String.concat ", " tags);
      None

let console_cmd base_dir run_id experiment tags =
  let base_dir = Option.value base_dir ~default:(Kaun_runlog.base_dir ()) in
  let run_id =
    match run_id with
    | Some id -> id
    | None -> (
        match discover_run ~base_dir ~experiment ~tags with
        | Some id -> id
        | None -> exit 1)
  in
  try Kaun_console.run ~base_dir ~runs:[ run_id ] () with
  | Failure msg ->
      Printf.eprintf "Error: %s\n" msg;
      exit 1
  | exn ->
      Printf.eprintf "Error: %s\n" (Printexc.to_string exn);
      exit 1

(* ───── Runs ───── *)

let format_time unix_time =
  let tm = Unix.localtime unix_time in
  Printf.sprintf "%04d-%02d-%02d %02d:%02d:%02d" (tm.Unix.tm_year + 1900)
    (tm.Unix.tm_mon + 1) tm.Unix.tm_mday tm.Unix.tm_hour tm.Unix.tm_min
    tm.Unix.tm_sec

let runs_list_cmd base_dir experiment tags =
  let base_dir = Option.value base_dir ~default:(Kaun_runlog.base_dir ()) in
  let runs = Kaun_runlog.discover ~base_dir () in
  let runs =
    runs
    |> List.filter (fun run ->
           match experiment with
           | None -> true
           | Some exp -> Run.experiment_name run = Some exp)
    |> List.filter (fun run ->
           match tags with
           | [] -> true
           | required_tags ->
               List.for_all (fun t -> List.mem t (Run.tags run)) required_tags)
  in
  match runs with
  | [] -> Printf.printf "No runs found.\n"
  | runs ->
      List.iter
        (fun run ->
          Printf.printf "%s" (Run.run_id run);
          (match Run.experiment_name run with
          | Some exp -> Printf.printf "  experiment=%s" exp
          | None -> ());
          if Run.tags run <> [] then
            Printf.printf "  tags=[%s]" (String.concat ", " (Run.tags run));
          Printf.printf "  %s\n" (format_time (Run.created_at run)))
        runs

let runs_show_cmd base_dir run_id =
  let base_dir = Option.value base_dir ~default:(Kaun_runlog.base_dir ()) in
  let run_dir = Filename.concat base_dir run_id in
  match Run.load run_dir with
  | None ->
      Printf.eprintf "Error: Run not found: %s\n" run_id;
      exit 1
  | Some run ->
      Printf.printf "Run:        %s\n" (Run.run_id run);
      Printf.printf "Created:    %s\n" (format_time (Run.created_at run));
      (match Run.experiment_name run with
      | Some exp -> Printf.printf "Experiment: %s\n" exp
      | None -> ());
      if Run.tags run <> [] then
        Printf.printf "Tags:       %s\n" (String.concat ", " (Run.tags run));
      let config = Run.config run in
      if config <> [] then
        Printf.printf "Config:     %d entries\n" (List.length config);
      let events = Run.events run in
      Printf.printf "Events:     %d\n" (List.length events);
      Printf.printf "Directory:  %s\n" (Run.dir run)

(* ───── Cmdliner ───── *)

open Cmdliner

(* Shared arguments *)

let base_dir_arg =
  Arg.(
    value
    & opt (some string) None
    & info [ "base-dir" ] ~docv:"DIR"
        ~doc:"Directory containing training runs.")

let experiment_arg =
  Arg.(
    value
    & opt (some string) None
    & info [ "experiment" ] ~docv:"NAME"
        ~doc:"Filter runs by experiment name.")

let tags_arg =
  Arg.(
    value & opt_all string []
    & info [ "tag" ] ~docv:"TAG"
        ~doc:"Filter runs by tag (can be repeated).")

(* Console *)

let console_run_arg =
  Arg.(
    value
    & opt (some string) None
    & info [ "run" ] ~docv:"ID" ~doc:"Specific run ID to monitor.")

let console_term =
  Term.(const console_cmd $ base_dir_arg $ console_run_arg $ experiment_arg
        $ tags_arg)

let console_cmd_v =
  Cmd.v
    (Cmd.info "console" ~doc:"Launch the training dashboard.")
    console_term

(* Runs list *)

let runs_list_term =
  Term.(const runs_list_cmd $ base_dir_arg $ experiment_arg $ tags_arg)

let runs_list_cmd_v =
  Cmd.v (Cmd.info "list" ~doc:"List discovered training runs.") runs_list_term

(* Runs show *)

let runs_show_run_arg =
  Arg.(
    required
    & pos 0 (some string) None
    & info [] ~docv:"RUN_ID" ~doc:"Run ID to show details for.")

let runs_show_term =
  Term.(const runs_show_cmd $ base_dir_arg $ runs_show_run_arg)

let runs_show_cmd_v =
  Cmd.v (Cmd.info "show" ~doc:"Show details of a specific run.") runs_show_term

(* Runs group *)

let runs_group =
  Cmd.group ~default:runs_list_term
    (Cmd.info "runs" ~doc:"Manage training runs.")
    [ runs_list_cmd_v; runs_show_cmd_v ]

(* Top-level *)

let kaun_cmd =
  let doc = "Neural network toolkit for OCaml." in
  let info = Cmd.info "kaun" ~doc in
  Cmd.group ~default:console_term info [ console_cmd_v; runs_group ]

let () = exit (Cmd.eval kaun_cmd)
