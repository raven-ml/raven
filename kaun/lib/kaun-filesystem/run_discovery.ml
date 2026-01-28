(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Run discovery for training logs. *)

let discover_runs ~base_dir ?experiment ?tags () =
  if not (Sys.file_exists base_dir) then []
  else
    let entries = Sys.readdir base_dir in
    let runs =
      Array.fold_left
        (fun acc entry ->
          let run_dir = Filename.concat base_dir entry in
          if Sys.is_directory run_dir then
            match Manifest.read ~run_dir with
            | Some manifest -> manifest :: acc
            | None -> acc
          else acc)
        [] entries
    in

    (* Filter by experiment if specified *)
    let runs =
      match experiment with
      | None -> runs
      | Some exp ->
          List.filter
            (fun manifest ->
              match manifest.Manifest.experiment with
              | Some e -> e = exp
              | None -> false)
            runs
    in

    (* Filter by tags if specified *)
    let runs =
      match tags with
      | None | Some [] -> runs
      | Some filter_tags ->
          List.filter
            (fun manifest ->
              List.for_all
                (fun tag -> List.mem tag manifest.Manifest.tags)
                filter_tags)
            runs
    in

    (* Sort by creation time (newest first) *)
    List.sort
      (fun a b -> compare b.Manifest.created_at a.Manifest.created_at)
      runs

let get_latest_run ~base_dir ?experiment ?tags () =
  match discover_runs ~base_dir ?experiment ?tags () with
  | [] -> None
  | latest :: _ -> Some latest
