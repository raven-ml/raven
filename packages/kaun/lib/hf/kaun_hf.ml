(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Error messages *)

let err_offline repo_id file =
  Printf.sprintf "Not cached (offline): %s/%s" repo_id file

let err_no_safetensors repo_id =
  Printf.sprintf "No safetensors found for %s" repo_id

let err_missing_tensor repo_id name shard =
  Printf.sprintf "%s: tensor %S missing in shard %s" repo_id name shard

let err_bad_weight_map path =
  Printf.sprintf "%s: missing or malformed weight_map" path

(* Cache directory *)

(* The user's home: [HOME] on Unix, [USERPROFILE] on Windows. *)
let home () =
  match (Sys.getenv_opt "HOME", Sys.getenv_opt "USERPROFILE") with
  | Some d, _ when d <> "" -> d
  | _, Some d when d <> "" -> d
  | _ -> failwith "no home directory: set HOME, or RAVEN_CACHE_ROOT"

(* [dir] under raven's cache root. *)
let raven_cache dir =
  match Sys.getenv_opt "RAVEN_CACHE_ROOT" with
  | Some d when d <> "" -> Filename.concat d dir
  | _ ->
      let xdg =
        match Sys.getenv_opt "XDG_CACHE_HOME" with
        | Some d when d <> "" -> d
        | _ -> Filename.concat (home ()) ".cache"
      in
      Filename.concat (Filename.concat xdg "raven") dir

let default_cache_dir () = raven_cache "huggingface"

let sanitize_repo_id repo_id =
  String.map (fun c -> if c = '/' then '-' else c) repo_id

let cache_path ?cache_dir ?(revision = "main") ~file repo_id =
  let cache_dir =
    match cache_dir with Some d -> d | None -> default_cache_dir ()
  in
  let repo_dir = sanitize_repo_id repo_id in
  Filename.concat cache_dir
    (Filename.concat repo_dir (Filename.concat revision file))

(* Filesystem *)

let rec rm_rf path =
  if Sys.is_directory path then begin
    Array.iter (fun e -> rm_rf (Filename.concat path e)) (Sys.readdir path);
    Unix.rmdir path
  end
  else Sys.remove path

let clear_cache ?cache_dir ?repo_id () =
  let cache_dir =
    match cache_dir with Some d -> d | None -> default_cache_dir ()
  in
  let path =
    match repo_id with
    | Some id -> Filename.concat cache_dir (sanitize_repo_id id)
    | None -> cache_dir
  in
  if Sys.file_exists path then (
    try rm_rf path
    with Sys_error _ | Unix.Unix_error _ ->
      Gc.full_major ();
      if Sys.file_exists path then rm_rf path)

(* Downloading *)

(* The longest token whose header line, ["Authorization: Bearer "], the token
   and a newline, fits the headers Kaun_download.file sends. *)
let max_token_bytes =
  Kaun_download.max_header_bytes - String.length "Authorization: Bearer \n"

let download_file ?token ?cache_dir ?(offline = false) ?(revision = "main")
    ~file repo_id =
  let local = cache_path ?cache_dir ~revision ~file repo_id in
  if Sys.file_exists local then local
  else if offline then failwith (err_offline repo_id file)
  else begin
    let token =
      match token with Some _ as t -> t | None -> Sys.getenv_opt "HF_TOKEN"
    in
    let headers =
      match token with
      | Some t
        when String.exists (fun c -> c = '\r' || c = '\n' || c = '\000') t ->
          invalid_arg
            "Kaun_hf.download_file: the token holds a CR, an LF or a NUL"
      | Some t when String.length t > max_token_bytes ->
          Printf.ksprintf invalid_arg
            "Kaun_hf.download_file: the token is longer than %d bytes"
            max_token_bytes
      | Some t -> [ ("Authorization", "Bearer " ^ t) ]
      | None -> []
    in
    let url =
      Printf.sprintf "https://huggingface.co/%s/resolve/%s/%s" repo_id revision
        file
    in
    Kaun_download.file ~headers ~url local;
    local
  end

(* Downloading by digest *)

let is_sha256 s =
  String.length s = 64
  && String.for_all
       (function '0' .. '9' | 'a' .. 'f' | 'A' .. 'F' -> true | _ -> false)
       s

(* The last segment of [url]'s path, before any query or fragment. *)
let url_name url =
  let rest =
    match String.index_from_opt url 0 ':' with
    | Some i
      when i + 2 < String.length url && url.[i + 1] = '/' && url.[i + 2] = '/'
      ->
        String.sub url (i + 3) (String.length url - i - 3)
    | _ -> url
  in
  let rest =
    match String.index_opt rest '?', String.index_opt rest '#' with
    | Some i, Some j -> String.sub rest 0 (min i j)
    | Some i, None | None, Some i -> String.sub rest 0 i
    | None, None -> rest
  in
  match String.rindex_opt rest '/' with
  | Some i ->
      let name = String.sub rest (i + 1) (String.length rest - i - 1) in
      if name = "" || name = "." || name = ".." || String.contains name '\\'
      then None
      else Some name
  | None -> None

let download_url ?cache_dir ?(offline = false) ~sha256 url =
  if not (is_sha256 sha256) then
    Printf.ksprintf invalid_arg
      "Kaun_hf.download_url: %S is not 64 hexadecimal digits" sha256;
  let name =
    match url_name url with
    | Some name -> name
    | None ->
        Printf.ksprintf invalid_arg "Kaun_hf.download_url: %s names no file"
          url
  in
  let sha256 = String.lowercase_ascii sha256 in
  let cache_dir =
    match cache_dir with Some d -> d | None -> raven_cache "sha256"
  in
  let local = Filename.concat (Filename.concat cache_dir sha256) name in
  if Sys.file_exists local then local
  else if offline then
    failwith (Printf.sprintf "Not cached (offline): %s" url)
  else begin
    Kaun_download.file ~sha256 ~url local;
    local
  end

(* JSON *)

let read_json_file path =
  let ic = open_in_bin path in
  let s =
    Fun.protect
      ~finally:(fun () -> close_in ic)
      (fun () -> really_input_string ic (in_channel_length ic))
  in
  match Jsont_bytesrw.decode_string Jsont.json s with
  | Ok v -> v
  | Error e -> failwith e

let load_config ?token ?cache_dir ?offline ?revision repo_id =
  read_json_file
    (download_file ?token ?cache_dir ?offline ?revision ~file:"config.json"
       repo_id)

(* Loading weights *)

(* The entries the index [index_path] names, each from its shard. *)
let load_sharded ~download repo_id index_path =
  let json = read_json_file index_path in
  let weight_map =
    match json with
    | Jsont.Object (mems, _) -> (
        match Jsont.Json.find_mem "weight_map" mems with
        | Some (_, Jsont.Object (entries, _)) ->
            List.map
              (fun ((tensor_name, _), shard_json) ->
                match shard_json with
                | Jsont.String (shard, _) -> (tensor_name, shard)
                | _ -> failwith (err_bad_weight_map index_path))
              entries
        | _ -> failwith (err_bad_weight_map index_path))
    | _ -> failwith (err_bad_weight_map index_path)
  in
  if weight_map = [] then failwith (err_bad_weight_map index_path);
  let shards = Hashtbl.create 8 in
  let shard file =
    match Hashtbl.find_opt shards file with
    | Some archive -> archive
    | None ->
        let archive = Nx_io.load_safetensors (download file) in
        Hashtbl.add shards file archive;
        archive
  in
  let names = Hashtbl.create (List.length weight_map) in
  let entry (name, file) =
    if Hashtbl.mem names name then failwith (err_bad_weight_map index_path);
    Hashtbl.add names name ();
    match Nx_io.Archive.find name (shard file) with
    | Some x -> (name, x)
    | None -> failwith (err_missing_tensor repo_id name file)
  in
  Nx_io.Archive.of_list (List.map entry weight_map)

let load_safetensors ?token ?cache_dir ?offline ?revision repo_id =
  let download file =
    download_file ?token ?cache_dir ?offline ?revision ~file repo_id
  in
  let try_download file =
    try Some (download file) with Failure _ | Sys_error _ -> None
  in
  let index = "model.safetensors.index.json" and single = "model.safetensors" in
  let cached file =
    Sys.file_exists (cache_path ?cache_dir ?revision ~file repo_id)
  in
  (* A cached single file means an earlier call found no index: asking the Hub
     for one again would put a request on every start. *)
  let index_path =
    if cached single && not (cached index) then None else try_download index
  in
  match index_path with
  | Some index_path -> load_sharded ~download repo_id index_path
  | None -> (
      match try_download single with
      | Some path -> Nx_io.load_safetensors path
      | None -> failwith (err_no_safetensors repo_id))
