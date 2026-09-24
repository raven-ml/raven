(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

module Checkpoint = Kaun.Checkpoint

let invalid_argf fmt = Printf.ksprintf invalid_arg fmt

(* Error messages *)

let err_no_curl = "curl not found on PATH"
let err_download url = Printf.sprintf "Failed to download %s" url

let err_offline repo_id file =
  Printf.sprintf "Not cached (offline): %s/%s" repo_id file

let err_no_safetensors repo_id =
  Printf.sprintf "No safetensors found for %s" repo_id

let err_missing_tensor repo_id name shard =
  Printf.sprintf "%s: tensor %S missing in shard %s" repo_id name shard

let err_bad_weight_map path =
  Printf.sprintf "%s: missing or malformed weight_map" path

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

let rec mkdir_p path =
  if path = "" || path = "." || path = Filename.dir_sep then ()
  else if not (Sys.file_exists path) then begin
    mkdir_p (Filename.dirname path);
    try Unix.mkdir path 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ()
  end

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

(* HTTP via curl *)

(* [curl] runs without a shell, so no argument needs quoting and the lookup on
   [PATH] is the same on every system. A missing program is exit code 127 from
   the forked child on Unix and [ENOENT] from process creation on Windows. *)
let curl args =
  let rec wait pid =
    try snd (Unix.waitpid [] pid)
    with Unix.Unix_error (Unix.EINTR, _, _) -> wait pid
  in
  match
    wait
      (Unix.create_process "curl"
         (Array.of_list ("curl" :: args))
         Unix.stdin Unix.stdout Unix.stderr)
  with
  | Unix.WEXITED 0 -> true
  | Unix.WEXITED 127 -> failwith err_no_curl
  | Unix.WEXITED _ | Unix.WSIGNALED _ | Unix.WSTOPPED _ -> false
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> failwith err_no_curl

let curl_download ~headers ~url ~dest () =
  mkdir_p (Filename.dirname dest);
  let headers =
    List.concat_map (fun (k, v) -> [ "-H"; k ^ ": " ^ v ]) headers
  in
  let temp =
    Filename.temp_file ~temp_dir:(Filename.dirname dest)
      (Filename.basename dest ^ ".")
      ".part"
  in
  let remove_temp () = try Sys.remove temp with Sys_error _ -> () in
  match curl ([ "-L"; "--fail"; "-s" ] @ headers @ [ "-o"; temp; url ]) with
  | true -> (
      Unix.chmod temp 0o644;
      try Unix.rename temp dest
      with Unix.Unix_error _ when Sys.file_exists dest -> remove_temp ())
  | false ->
      remove_temp ();
      failwith (err_download url)
  | exception e ->
      remove_temp ();
      raise e

(* Downloading *)

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
      | Some t -> [ ("Authorization", "Bearer " ^ t) ]
      | None -> []
    in
    let url =
      Printf.sprintf "https://huggingface.co/%s/resolve/%s/%s" repo_id revision
        file
    in
    curl_download ~headers ~url ~dest:local ();
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

(* Loading checkpoints *)

let load_sharded ~download index_path =
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
    | Some ckpt -> ckpt
    | None ->
        let ckpt = Checkpoint.load (download file) in
        Hashtbl.add shards file ckpt;
        ckpt
  in
  List.fold_left
    (fun acc (name, file) ->
      match Checkpoint.find name (shard file) with
      | Some (Nx.P x) -> Checkpoint.concat [ acc; Checkpoint.of_tensor name x ]
      | None -> failwith (err_missing_tensor "" name file))
    Checkpoint.empty weight_map

let load_checkpoint ?token ?cache_dir ?offline ?revision repo_id =
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
  | Some index_path -> load_sharded ~download index_path
  | None -> (
      match try_download single with
      | Some path -> Checkpoint.load path
      | None -> failwith (err_no_safetensors repo_id))
