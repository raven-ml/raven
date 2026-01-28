(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Command-line interface for kaun-console. *)

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
        (* Auto-discover latest run *)
        let tags_opt =
          match List.rev !tags with [] -> None | ts -> Some ts
        in
        (match
           Kaun_filesystem.Run_discovery.get_latest_run ~base_dir:!base_dir
             ?experiment:!experiment ?tags:tags_opt ()
         with
        | Some manifest ->
            Printf.printf "Auto-discovered run: %s\n" manifest.Kaun_filesystem.Manifest.run_id;
            (match manifest.Kaun_filesystem.Manifest.experiment with
            | Some exp -> Printf.printf "  Experiment: %s\n" exp
            | None -> ());
            if manifest.Kaun_filesystem.Manifest.tags <> [] then
              Printf.printf "  Tags: %s\n"
                (String.concat ", " manifest.Kaun_filesystem.Manifest.tags);
            Printf.printf "\n%!";
            manifest.Kaun_filesystem.Manifest.run_id
        | None ->
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
