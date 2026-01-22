(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Run discovery for Kaun training logs. *)

type run_info = {
  run_id : string;
  created_at : float;
  experiment : string option;
  tags : string list;
}

(* ───── JSON Parsing Helpers ───── *)

let get_string key = function
  | `Assoc fields -> (
      match List.assoc_opt key fields with
      | Some (`String s) -> Some s
      | _ -> None)
  | _ -> None

let get_float key = function
  | `Assoc fields -> (
      match List.assoc_opt key fields with
      | Some (`Float f) -> Some f
      | Some (`Int i) -> Some (float_of_int i)
      | _ -> None)
  | _ -> None

let get_string_list key = function
  | `Assoc fields -> (
      match List.assoc_opt key fields with
      | Some (`List items) ->
          List.filter_map (function `String s -> Some s | _ -> None) items
      | _ -> [])
  | _ -> []

(* ───── Manifest Parsing ───── *)

let parse_manifest run_dir =
  let manifest_path = Filename.concat run_dir "run.json" in
  if not (Sys.file_exists manifest_path) then None
  else
    try
      let json = Yojson.Basic.from_file manifest_path in
      let run_id =
        match get_string "run_id" json with
        | Some s -> s
        | None -> Filename.basename run_dir
      in
      let created_at =
        match get_float "created_at" json with Some f -> f | None -> 0.0
      in
      let experiment = get_string "experiment" json in
      let tags = get_string_list "tags" json in
      Some { run_id; created_at; experiment; tags }
    with _ -> None

(* ───── Run Discovery ───── *)

let discover_runs ~base_dir ?experiment ?tags () =
  if not (Sys.file_exists base_dir) then []
  else
    let entries = Sys.readdir base_dir in
    let runs =
      Array.fold_left
        (fun acc entry ->
          let run_dir = Filename.concat base_dir entry in
          if Sys.is_directory run_dir then
            match parse_manifest run_dir with
            | Some info -> info :: acc
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
            (fun info ->
              match info.experiment with
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
            (fun info ->
              List.for_all (fun tag -> List.mem tag info.tags) filter_tags)
            runs
    in

    (* Sort by creation time (newest first) *)
    List.sort (fun a b -> compare b.created_at a.created_at) runs

let get_latest_run ~base_dir ?experiment ?tags () =
  match discover_runs ~base_dir ?experiment ?tags () with
  | [] -> None
  | latest :: _ -> Some latest
