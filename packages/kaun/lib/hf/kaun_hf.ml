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
  if Sys.file_exists path then rm_rf path

(* HTTP via curl *)

let curl_available =
  lazy (Unix.system "command -v curl >/dev/null 2>&1" = Unix.WEXITED 0)

let check_curl () = if not (Lazy.force curl_available) then failwith err_no_curl

let curl_download ~headers ~url ~dest () =
  check_curl ();
  mkdir_p (Filename.dirname dest);
  let hdr =
    List.map
      (fun (k, v) -> Printf.sprintf "-H %s" (Filename.quote (k ^ ": " ^ v)))
      headers
    |> String.concat " "
  in
  let cmd =
    Printf.sprintf "curl -L --fail -s %s -o %s %s" hdr (Filename.quote dest)
      (Filename.quote url)
  in
  match Unix.system cmd with
  | Unix.WEXITED 0 -> ()
  | _ ->
      (try Sys.remove dest with Sys_error _ -> ());
      failwith (err_download url)

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

(* [of_entries ~op entries] is the checkpoint holding [entries], validating name
   distinctness and non-emptiness with [op]-labelled errors before
   [Checkpoint.concat] can produce its own, less specific ones. *)
let of_entries ~op entries =
  let seen = Hashtbl.create (List.length entries) in
  List.iter
    (fun (name, _) ->
      if name = "" then invalid_argf "Kaun_hf.%s: empty entry name" op;
      if Hashtbl.mem seen name then
        invalid_argf "Kaun_hf.%s: duplicate name %S" op name;
      Hashtbl.add seen name ())
    entries;
  Checkpoint.concat
    (List.map
       (fun (name, Rune.Ptree.P x) -> Checkpoint.of_tensor name x)
       entries)

let entries t = List.map (fun n -> (n, Checkpoint.get n t)) (Checkpoint.names t)

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
      | Some (Rune.Ptree.P x) ->
          Checkpoint.concat [ acc; Checkpoint.of_tensor name x ]
      | None -> failwith (err_missing_tensor "" name file))
    Checkpoint.empty weight_map

let load_checkpoint ?token ?cache_dir ?offline ?revision repo_id =
  let download file =
    download_file ?token ?cache_dir ?offline ?revision ~file repo_id
  in
  let try_download file =
    try Some (download file) with Failure _ | Sys_error _ -> None
  in
  match try_download "model.safetensors.index.json" with
  | Some index_path -> load_sharded ~download index_path
  | None -> (
      match try_download "model.safetensors" with
      | Some path -> Checkpoint.load path
      | None -> failwith (err_no_safetensors repo_id))

(* Adapting foreign checkpoints *)

let rename f t =
  of_entries ~op:"rename" (List.map (fun (n, x) -> (f n, x)) (entries t))

let transpose name t =
  match Checkpoint.find name t with
  | None -> invalid_argf "Kaun_hf.transpose: no entry named %S" name
  | Some (Rune.Ptree.P x) ->
      let nd = Array.length (Nx.shape x) in
      if nd < 2 then
        invalid_argf "Kaun_hf.transpose: entry %S has %d axes, needs at least 2"
          name nd;
      let x = Nx.swapaxes (nd - 2) (nd - 1) x in
      of_entries ~op:"transpose"
        (List.map
           (fun (n, e) -> if n = name then (n, Rune.Ptree.P x) else (n, e))
           (entries t))

let split ?(axis = -1) name ~into t =
  match Checkpoint.find name t with
  | None -> invalid_argf "Kaun_hf.split: no entry named %S" name
  | Some (Rune.Ptree.P x) ->
      let parts = List.length into in
      if parts = 0 then invalid_arg "Kaun_hf.split: empty name list";
      let shape = Nx.shape x in
      let nd = Array.length shape in
      let axis = if axis < 0 then axis + nd else axis in
      if axis < 0 || axis >= nd then
        invalid_argf "Kaun_hf.split: axis out of bounds for entry %S" name;
      if shape.(axis) mod parts <> 0 then
        invalid_argf
          "Kaun_hf.split: axis %d of entry %S has size %d, not a multiple of %d"
          axis name shape.(axis) parts;
      let sections =
        List.map2 (fun n x -> (n, Rune.Ptree.P x)) into (Nx.split ~axis parts x)
      in
      let rest = List.filter (fun (n, _) -> n <> name) (entries t) in
      of_entries ~op:"split" (rest @ sections)
