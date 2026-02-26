(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Types *)

type revision = Main | Rev of string

(* Error messages *)

let err_no_curl = "curl not found on PATH"
let err_download url = Printf.sprintf "Failed to download %s" url

let err_offline model_id filename =
  Printf.sprintf "Not cached (offline): %s/%s" model_id filename

let err_no_safetensors model_id =
  Printf.sprintf "No safetensors found for %s" model_id

let err_missing_tensor model_id name path =
  Printf.sprintf "%s: tensor %S missing in shard %s" model_id name path

let err_empty_weight_map = "Empty weight_map in index file"
let err_missing_weight_map = "Missing weight_map in index file"

let err_incomplete_shards =
  "Incomplete shard loading: not all weight_map tensors were found"

(* Cache directory *)

let default_cache_dir () =
  match Sys.getenv_opt "RAVEN_CACHE_ROOT" with
  | Some d when d <> "" -> Filename.concat d "huggingface"
  | _ ->
      let xdg =
        match Sys.getenv_opt "XDG_CACHE_HOME" with
        | Some d when d <> "" -> d
        | _ -> Filename.concat (Sys.getenv "HOME") ".cache"
      in
      Filename.concat (Filename.concat xdg "raven") "huggingface"

(* Filesystem *)

let rec mkdir_p path =
  if path = "" || path = "." || path = Filename.dir_sep then ()
  else if not (Sys.file_exists path) then begin
    mkdir_p (Filename.dirname path);
    try Unix.mkdir path 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ()
  end

let rec rm_rf path =
  if Sys.is_directory path then begin
    let entries = Sys.readdir path in
    Array.iter (fun e -> rm_rf (Filename.concat path e)) entries;
    Unix.rmdir path
  end
  else Sys.remove path

(* HTTP via curl *)

let curl_available =
  lazy (Unix.system "command -v curl >/dev/null 2>&1" = Unix.WEXITED 0)

let check_curl () = if not (Lazy.force curl_available) then failwith err_no_curl

let header_flags headers =
  List.map
    (fun (k, v) -> Printf.sprintf "-H %s" (Filename.quote (k ^ ": " ^ v)))
    headers
  |> String.concat " "

let curl_download ~headers ~url ~dest () =
  check_curl ();
  mkdir_p (Filename.dirname dest);
  let hdr = header_flags headers in
  let cmd =
    Printf.sprintf "curl -L --fail -s %s -o %s %s" hdr (Filename.quote dest)
      (Filename.quote url)
  in
  match Unix.system cmd with
  | Unix.WEXITED 0 -> ()
  | _ ->
      (try Sys.remove dest with Sys_error _ -> ());
      failwith (err_download url)

(* Hub URL and cache paths *)

let revision_string = function Main -> "main" | Rev r -> r

let hub_url ~model_id ~revision ~filename =
  Printf.sprintf "https://huggingface.co/%s/resolve/%s/%s" model_id
    (revision_string revision) filename

let sanitize_model_id model_id =
  String.map (fun c -> if c = '/' then '-' else c) model_id

let cache_path ~cache_dir ~model_id ~revision ~filename =
  let rev = revision_string revision in
  let model_dir = sanitize_model_id model_id in
  Filename.concat cache_dir
    (Filename.concat model_dir (Filename.concat rev filename))

let auth_headers = function
  | Some t -> [ ("Authorization", "Bearer " ^ t) ]
  | None -> []

(* Downloading *)

let download_file ?token ?cache_dir ?(offline = false) ?(revision = Main)
    ~model_id ~filename () =
  let token =
    match token with Some _ as t -> t | None -> Sys.getenv_opt "HF_TOKEN"
  in
  let cache_dir = Option.value cache_dir ~default:(default_cache_dir ()) in
  let local = cache_path ~cache_dir ~model_id ~revision ~filename in
  if Sys.file_exists local then local
  else if offline then failwith (err_offline model_id filename)
  else begin
    let url = hub_url ~model_id ~revision ~filename in
    curl_download ~headers:(auth_headers token) ~url ~dest:local ();
    local
  end

(* JSON helpers *)

let read_json_file path =
  let ic = open_in path in
  let s =
    Fun.protect
      ~finally:(fun () -> close_in ic)
      (fun () -> really_input_string ic (in_channel_length ic))
  in
  match Jsont_bytesrw.decode_string Jsont.json s with
  | Ok v -> v
  | Error e -> failwith e

let json_mem name = function
  | Jsont.Object (mems, _) -> (
      match Jsont.Json.find_mem name mems with
      | Some (_, v) -> v
      | None -> Jsont.Null ((), Jsont.Meta.none))
  | _ -> Jsont.Null ((), Jsont.Meta.none)

(* Tensor conversion *)

let to_ptree_tensor (Nx_io.P nx) = Kaun.Ptree.P (Rune.of_nx nx)

(* Loading *)

let load_entries ?allowed_names path =
  let archive = Nx_io.load_safetensors path in
  match allowed_names with
  | None ->
      Hashtbl.fold
        (fun name packed acc -> (name, to_ptree_tensor packed) :: acc)
        archive []
  | Some names ->
      List.map
        (fun name ->
          match Hashtbl.find_opt archive name with
          | Some packed -> (name, to_ptree_tensor packed)
          | None -> failwith (err_missing_tensor "" name path))
        names

let try_download f =
  try Some (f ()) with Failure _ -> None | Sys_error _ -> None

let load_sharded ~download index_filename =
  match try_download (fun () -> download index_filename) with
  | None -> None
  | Some index_path ->
      let json = read_json_file index_path in
      let weight_map =
        match json_mem "weight_map" json with
        | Jsont.Object (entries, _) ->
            List.map
              (fun ((tensor_name, _), shard_json) ->
                match shard_json with
                | Jsont.String (shard, _) -> (tensor_name, shard)
                | _ -> failwith err_missing_weight_map)
              entries
        | _ -> failwith err_missing_weight_map
      in
      if weight_map = [] then failwith err_empty_weight_map;
      (* Group tensors by shard filename, preserving file order *)
      let shards_by_file = Hashtbl.create 8 in
      let file_order = ref [] in
      List.iter
        (fun (tensor_name, shard_filename) ->
          match Hashtbl.find_opt shards_by_file shard_filename with
          | Some tensors ->
              Hashtbl.replace shards_by_file shard_filename
                (tensor_name :: tensors)
          | None ->
              Hashtbl.add shards_by_file shard_filename [ tensor_name ];
              file_order := shard_filename :: !file_order)
        weight_map;
      let file_order = List.rev !file_order in
      let seen = Hashtbl.create (List.length weight_map) in
      let entries =
        List.fold_left
          (fun acc shard_filename ->
            let shard_path = download shard_filename in
            let tensors =
              match Hashtbl.find_opt shards_by_file shard_filename with
              | Some names -> List.rev names
              | None -> []
            in
            let new_entries = load_entries ~allowed_names:tensors shard_path in
            List.iter
              (fun (name, _) -> Hashtbl.replace seen name ())
              new_entries;
            List.rev_append new_entries acc)
          [] file_order
      in
      if Hashtbl.length seen <> List.length weight_map then
        failwith err_incomplete_shards;
      Some (List.rev entries)

let load_single ~download filename =
  match try_download (fun () -> download filename) with
  | None -> None
  | Some path -> Some (load_entries path)

let load_config ?token ?cache_dir ?offline ?revision ~model_id () =
  let path =
    download_file ?token ?cache_dir ?offline ?revision ~model_id
      ~filename:"config.json" ()
  in
  read_json_file path

let load_weights ?token ?cache_dir ?offline ?revision ~model_id () =
  let download filename =
    download_file ?token ?cache_dir ?offline ?revision ~model_id ~filename ()
  in
  match load_sharded ~download "model.safetensors.index.json" with
  | Some entries -> entries
  | None -> (
      match load_single ~download "model.safetensors" with
      | Some entries -> entries
      | None -> failwith (err_no_safetensors model_id))

(* Cache management *)

let clear_cache ?cache_dir ?model_id () =
  let cache_dir = Option.value cache_dir ~default:(default_cache_dir ()) in
  match model_id with
  | Some id ->
      let path = Filename.concat cache_dir (sanitize_model_id id) in
      if Sys.file_exists path then rm_rf path
  | None -> if Sys.file_exists cache_dir then rm_rf cache_dir
