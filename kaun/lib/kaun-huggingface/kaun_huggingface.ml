(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Rune

(* ───── Types ───── *)

type model_id = string
type revision = Latest | Tag of string | Commit of string
type cache_dir = string

type download_progress = {
  downloaded_bytes : int;
  total_bytes : int option;
  rate : float;
}

type 'a download_result = Cached of 'a | Downloaded of 'a * download_progress

(* ───── Configuration ───── *)

module Config = struct
  type t = {
    cache_dir : cache_dir;
    token : string option;
    offline_mode : bool;
    force_download : bool;
    show_progress : bool;
  }

  let default =
    {
      cache_dir =
        Filename.concat
          (try Sys.getenv "HOME" with Not_found -> "/tmp")
          ".cache/kaun/huggingface";
      token = None;
      offline_mode = false;
      force_download = false;
      show_progress = true;
    }

  let from_env () =
    let get_env_opt var = try Some (Sys.getenv var) with Not_found -> None in
    let get_env_bool var =
      match get_env_opt var with Some "true" | Some "1" -> true | _ -> false
    in
    {
      cache_dir =
        (match get_env_opt "KAUN_HF_CACHE_DIR" with
        | Some dir -> dir
        | None -> default.cache_dir);
      token = get_env_opt "KAUN_HF_TOKEN";
      offline_mode = get_env_bool "KAUN_HF_OFFLINE_MODE";
      force_download = get_env_bool "KAUN_HF_FORCE_DOWNLOAD";
      show_progress =
        (match get_env_opt "KAUN_HF_SHOW_PROGRESS" with
        | Some "false" | Some "0" -> false
        | _ -> true);
    }
end

(* ───── Model Registry ───── *)

module Registry = struct
  type ('params, 'a, 'dev) model_spec = {
    architecture : string;
    config_file : string;
    weight_files : string list;
    load_config : Yojson.Safe.t -> 'params;
    build_params : dtype:(float, 'a) dtype -> 'params -> Kaun.params;
  }

  let registry : (string, Obj.t) Hashtbl.t = Hashtbl.create 10
  let register name spec = Hashtbl.replace registry name (Obj.repr spec)

  let get name =
    try Some (Obj.obj (Hashtbl.find registry name)) with Not_found -> None
end

(* ───── Utilities ───── *)

let ensure_dir dir =
  let rec mkdir_p path =
    if not (Sys.file_exists path) then (
      mkdir_p (Filename.dirname path);
      try Unix.mkdir path 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ())
  in
  mkdir_p dir

let hub_url ~model_id ~filename ~revision =
  let revision_str =
    match revision with
    | Latest -> "main"
    | Tag tag -> tag
    | Commit commit -> commit
  in
  Printf.sprintf "https://huggingface.co/%s/resolve/%s/%s" model_id revision_str
    filename

let cache_path config ~model_id ~filename ~revision =
  let revision_str =
    match revision with
    | Latest -> "main"
    | Tag tag -> "tags/" ^ tag
    | Commit commit -> "commits/" ^ commit
  in
  let model_dir = String.map (fun c -> if c = '/' then '-' else c) model_id in
  Filename.concat config.Config.cache_dir
    (Filename.concat model_dir (Filename.concat revision_str filename))

(* ───── Core Loading Functions ───── *)

let download_with_progress ~url ~dest ~show_progress =
  ensure_dir (Filename.dirname dest);

  (* Use curl with progress bar if requested *)
  let progress_flag = if show_progress then "" else "-s" in
  let cmd = Printf.sprintf "curl -L %s -o %s '%s'" progress_flag dest url in

  if show_progress then Printf.printf "Downloading from %s...\n%!" url;

  let start_time = Unix.gettimeofday () in

  match Unix.system cmd with
  | Unix.WEXITED 0 ->
      let elapsed = Unix.gettimeofday () -. start_time in
      let stats = Unix.stat dest in
      let bytes = stats.Unix.st_size in
      let rate = float_of_int bytes /. elapsed in
      { downloaded_bytes = bytes; total_bytes = Some bytes; rate }
  | _ -> failwith (Printf.sprintf "Failed to download %s" url)

let download_file ?(config = Config.default) ?(revision = Latest) ~model_id
    ~filename () =
  let local_path = cache_path config ~model_id ~filename ~revision in

  (* Check if already cached *)
  if Sys.file_exists local_path && not config.force_download then
    Cached local_path
  else if config.offline_mode then
    failwith (Printf.sprintf "File not in cache (offline mode): %s" local_path)
  else
    (* Download the file *)
    let url = hub_url ~model_id ~filename ~revision in
    let progress =
      download_with_progress ~url ~dest:local_path
        ~show_progress:config.show_progress
    in
    Downloaded (local_path, progress)

let load_safetensors ?(config = Config.default) ?(revision = Latest) ~model_id
    () =
  let load_entries ?allowed_names path =
    let archive = Nx_io.load_safetensor path in
    match allowed_names with
    | None ->
        Hashtbl.fold
          (fun name (Nx_io.P nx_tensor) acc ->
            let rune_tensor = Rune.of_nx nx_tensor in
            (name, Kaun.Ptree.tensor rune_tensor) :: acc)
          archive []
    | Some names ->
        List.map
          (fun name ->
            match Hashtbl.find_opt archive name with
            | Some (Nx_io.P nx_tensor) ->
                let rune_tensor = Rune.of_nx nx_tensor in
                (name, Kaun.Ptree.tensor rune_tensor)
            | None ->
                failwith
                  (Printf.sprintf
                     "Shard for %s missing tensor '%s' while loading %s"
                     model_id name path))
          names
  in

  let params_from_entries entries = Kaun.Ptree.dict entries in

  let apply_progress progress_list params =
    let combine acc next =
      let accumulated_time =
        if acc.rate <= 0. then 0.
        else float_of_int acc.downloaded_bytes /. acc.rate
      in
      let next_time =
        if next.rate <= 0. then 0.
        else float_of_int next.downloaded_bytes /. next.rate
      in
      let total_bytes = acc.downloaded_bytes + next.downloaded_bytes in
      let total_time = accumulated_time +. next_time in
      let rate =
        if total_time <= 0. then acc.rate
        else float_of_int total_bytes /. total_time
      in
      let total_size =
        match (acc.total_bytes, next.total_bytes) with
        | Some a, Some b -> Some (a + b)
        | _ -> None
      in
      { downloaded_bytes = total_bytes; total_bytes = total_size; rate }
    in
    match progress_list with
    | [] -> Cached params
    | first :: rest ->
        let total = List.fold_left combine first rest in
        Downloaded (params, total)
  in

  let index_filenames =
    [ "model.safetensors.index.json"; "pytorch_model.bin.index.json" ]
  in

  let fallback_filenames =
    [
      "model.safetensors";
      "pytorch_model.safetensors";
      "model-00001-of-00001.safetensors";
    ]
  in

  let local_path_of_result = function
    | Cached path -> path
    | Downloaded (path, _) -> path
  in

  let progress_of_result acc = function
    | Cached _ -> acc
    | Downloaded (_, progress) -> progress :: acc
  in

  let attempt_index filename =
    try
      let result = download_file ~config ~revision ~model_id ~filename () in
      let index_path = local_path_of_result result in
      let progress_acc = progress_of_result [] result in

      let json = Yojson.Safe.from_file index_path in
      let weight_map =
        match Yojson.Safe.Util.member "weight_map" json with
        | `Assoc entries ->
            List.map
              (fun (tensor_name, shard_json) ->
                match shard_json with
                | `String shard -> (tensor_name, shard)
                | _ -> failwith "Invalid shard entry in weight_map")
              entries
        | _ -> failwith "Missing weight_map in index file"
      in

      if weight_map = [] then failwith "Empty weight_map in index file";

      let shards_by_file = Hashtbl.create 8 in
      let file_order = ref [] in
      List.iter
        (fun (tensor_name, shard_filename) ->
          let existing = Hashtbl.find_opt shards_by_file shard_filename in
          match existing with
          | Some tensors ->
              Hashtbl.replace shards_by_file shard_filename
                (tensor_name :: tensors)
          | None ->
              Hashtbl.add shards_by_file shard_filename [ tensor_name ];
              file_order := shard_filename :: !file_order)
        weight_map;

      let file_order = List.rev !file_order in
      let seen_tensors = Hashtbl.create (List.length weight_map) in

      let progress_list, entries_rev =
        List.fold_left
          (fun (progresses, acc_entries_rev) shard_filename ->
            let shard_result =
              download_file ~config ~revision ~model_id ~filename:shard_filename
                ()
            in
            let shard_path = local_path_of_result shard_result in
            let progresses = progress_of_result progresses shard_result in
            let tensors =
              match Hashtbl.find_opt shards_by_file shard_filename with
              | Some names -> List.rev names
              | None ->
                  failwith
                    (Printf.sprintf "Shard mapping missing for file '%s' in %s"
                       shard_filename filename)
            in
            let new_entries = load_entries ~allowed_names:tensors shard_path in
            List.iter
              (fun (tensor_name, _) ->
                if Hashtbl.mem seen_tensors tensor_name then
                  failwith
                    (Printf.sprintf
                       "Tensor '%s' defined multiple times across shards"
                       tensor_name);
                Hashtbl.add seen_tensors tensor_name ())
              new_entries;
            let acc_entries_rev =
              List.fold_left
                (fun acc entry -> entry :: acc)
                acc_entries_rev (List.rev new_entries)
            in
            (progresses, acc_entries_rev))
          (progress_acc, []) file_order
      in

      if Hashtbl.length seen_tensors <> List.length weight_map then
        failwith
          "Incomplete shard loading: not all tensors listed in the weight_map \
           were found";

      let entries = List.rev entries_rev in
      let params = params_from_entries entries in
      Some (apply_progress progress_list params)
    with
    | Failure msg
      when String.starts_with ~prefix:"Failed to download" msg
           || String.starts_with ~prefix:"No such file" msg
           || String.starts_with ~prefix:"File not in cache (offline mode)" msg
      ->
        None
    | Yojson.Json_error _ -> None
    | Sys_error _ -> None
  in

  let rec try_indexes = function
    | [] -> None
    | filename :: rest -> (
        match attempt_index filename with
        | Some result -> Some result
        | None -> try_indexes rest)
  in

  match try_indexes index_filenames with
  | Some result -> result
  | None ->
      let rec try_files = function
        | [] ->
            failwith
              (Printf.sprintf "No safetensors file found for %s" model_id)
        | filename :: rest -> (
            try
              let result =
                download_file ~config ~revision ~model_id ~filename ()
              in
              let local_path = local_path_of_result result in
              let entries = load_entries local_path in
              let params = params_from_entries entries in
              match result with
              | Cached _ -> Cached params
              | Downloaded (_, progress) -> Downloaded (params, progress)
            with
            | Failure msg
              when String.starts_with ~prefix:"Failed to download" msg
                   || String.starts_with ~prefix:"No such file" msg ->
                try_files rest
            | Failure _ -> try_files rest
            | _ -> try_files rest)
      in
      try_files fallback_filenames

let load_config ?(config = Config.default) ?(revision = Latest) ~model_id () =
  let result =
    download_file ~config ~revision ~model_id ~filename:"config.json" ()
  in
  let local_path =
    match result with Cached path -> path | Downloaded (path, _) -> path
  in

  let json = Yojson.Safe.from_file local_path in

  match result with
  | Cached _ -> Cached json
  | Downloaded (_, progress) -> Downloaded (json, progress)

(* ───── High-level Model Loading ───── *)

let from_pretrained ?(config = Config.default) ?(revision = Latest) ~model_id ()
    =
  (* Load safetensors weights *)
  match load_safetensors ~config ~revision ~model_id () with
  | Cached params -> params
  | Downloaded (params, _) -> params

(* ───── Utilities ───── *)

let list_cached_models ?(config = Config.default) () =
  if not (Sys.file_exists config.cache_dir) then []
  else
    let entries = Sys.readdir config.cache_dir in
    Array.to_list entries
    |> List.filter (fun e ->
        Sys.is_directory (Filename.concat config.cache_dir e))
    |> List.map (fun e -> String.map (fun c -> if c = '-' then '/' else c) e)

let clear_cache ?(config = Config.default) ?model_id () =
  let rec rm_rf path =
    if Sys.is_directory path then (
      let entries = Sys.readdir path in
      Array.iter (fun entry -> rm_rf (Filename.concat path entry)) entries;
      Unix.rmdir path)
    else Sys.remove path
  in

  match model_id with
  | Some id ->
      let model_dir = String.map (fun c -> if c = '/' then '-' else c) id in
      let path = Filename.concat config.cache_dir model_dir in
      if Sys.file_exists path then rm_rf path
  | None -> if Sys.file_exists config.cache_dir then rm_rf config.cache_dir

let get_model_info model_id =
  let url = Printf.sprintf "https://huggingface.co/api/models/%s" model_id in
  let cmd = Printf.sprintf "curl -s '%s'" url in

  let ic = Unix.open_process_in cmd in
  let rec read_all acc =
    try
      let line = input_line ic in
      read_all (acc ^ line ^ "\n")
    with End_of_file -> acc
  in
  let output = read_all "" in
  let status = Unix.close_process_in ic in

  match status with
  | Unix.WEXITED 0 -> (
      try Ok (Yojson.Safe.from_string output)
      with _ -> Error "Failed to parse JSON response")
  | _ -> Error "Failed to fetch model info"
