(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* Error messages *)

let err_no_curl = "curl not found on PATH"
let err_download url = Printf.sprintf "Failed to download %s" url

let err_spawn url e =
  Printf.sprintf "%s: curl could not run: %s" url (Unix.error_message e)

let err_nul url =
  Printf.sprintf "%s: holds a NUL, which no command line can carry" url

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
   the forked child on Unix and [ENOENT] from process creation on Windows.
   Any other failure to make the process, such as a program on [PATH] that
   cannot be executed, fails the request for [url]. *)
let spawn ~url ~stdin args out =
  try
    Unix.create_process "curl"
      (Array.of_list ("curl" :: args))
      stdin out Unix.stderr
  with
  | Unix.Unix_error (Unix.ENOENT, _, _) -> failwith err_no_curl
  | Unix.Unix_error (e, _, _) -> failwith (err_spawn url e)

let rec wait pid =
  try snd (Unix.waitpid [] pid)
  with Unix.Unix_error (Unix.EINTR, _, _) -> wait pid

let succeeded = function
  | Unix.WEXITED 0 -> true
  | Unix.WEXITED 127 -> failwith err_no_curl
  | Unix.WEXITED _ | Unix.WSIGNALED _ | Unix.WSTOPPED _ -> false

let curl ~url ~stdin args ~out = succeeded (wait (spawn ~url ~stdin args out))

(* [curl args] with its output read through a pipe, each chunk given to [f]
   as it arrives. *)
let curl_read ~url ~stdin args ~f =
  let r, w = Unix.pipe ~cloexec:true () in
  let pid =
    Fun.protect
      ~finally:(fun () -> Unix.close w)
      (fun () ->
        try spawn ~url ~stdin args w
        with e ->
          Unix.close r;
          raise e)
  in
  let buf = Bytes.create 65536 in
  let rec copy () =
    match Unix.read r buf 0 (Bytes.length buf) with
    | 0 -> ()
    | n ->
        f buf n;
        copy ()
    | exception Unix.Unix_error (Unix.EINTR, _, _) -> copy ()
  in
  match copy () with
  | () ->
      Unix.close r;
      succeeded (wait pid)
  | exception e ->
      (* curl, writing to a closed pipe, ends. *)
      Unix.close r;
      ignore (wait pid);
      raise e

(* The headers curl reads on its standard input ([-H @-]), so that none, a
   token's above all, appears on its command line, which every user of the
   machine can read. They are written to the pipe before curl starts: at most
   1024 bytes, well under the 4096 a pipe is made with on Windows and the page a
   Linux pipe keeps at least, so the write cannot block with no reader, and it
   cannot raise SIGPIPE while this process holds the read end. *)
let max_header_bytes = 1024

let header_input headers =
  let lines = List.map (fun (k, v) -> k ^ ": " ^ v ^ "\n") headers in
  let text = String.concat "" lines in
  if String.length text > max_header_bytes then
    invalid_arg
      (Printf.sprintf "Kaun_hf: the headers %s exceed %d bytes"
         (String.concat ", " (List.map fst headers))
         max_header_bytes);
  let r, w = Unix.pipe ~cloexec:true () in
  Fun.protect
    ~finally:(fun () -> Unix.close w)
    (fun () ->
      try ignore (Unix.write_substring w text 0 (String.length text))
      with e ->
        Unix.close r;
        raise e);
  r

let with_headers headers f =
  match headers with
  | [] -> f ~stdin:Unix.stdin []
  | _ ->
      let r = header_input headers in
      Fun.protect
        ~finally:(fun () -> Unix.close r)
        (fun () -> f ~stdin:r [ "-H"; "@-" ])

(* With [sha256], the download is hashed as it is written and refused unless
   its digest is [sha256]. *)
let curl_download ?sha256 ~headers ~url ~dest () =
  (* A NUL ends a command line, and a path where the URL's parts name the
     cache path: refused before either is made. *)
  if String.contains url '\000' then failwith (err_nul url);
  mkdir_p (Filename.dirname dest);
  (* The download is written to the descriptor that created the file: a name
     reopened would follow a link put there meanwhile. *)
  let temp, oc =
    Filename.open_temp_file ~mode:[ Open_binary ] ~perms:0o644
      ~temp_dir:(Filename.dirname dest)
      (Filename.basename dest ^ ".")
      ".part"
  in
  let remove_temp () = try Sys.remove temp with Sys_error _ -> () in
  let fetch () =
    with_headers headers @@ fun ~stdin header_args ->
    (* [-q] first: curl reads no ~/.curlrc, whose options, such as
       [--location-trusted], would act on the download. *)
    let args = [ "-q"; "-L"; "--fail"; "-s" ] @ header_args @ [ url ] in
    match sha256 with
    | None -> curl ~url ~stdin ~out:(Unix.descr_of_out_channel oc) args
    | Some want ->
        let h = Sha256.init () in
        let f b n =
          Sha256.feed h b 0 n;
          output oc b 0 n
        in
        let ok = curl_read ~url ~stdin args ~f in
        flush oc;
        let got = Sha256.hex h in
        if ok && got <> want then
          failwith (Printf.sprintf "%s: SHA-256 %s, expected %s" url got want);
        ok
  in
  match Fun.protect ~finally:(fun () -> close_out_noerr oc) fetch with
  | true -> (
      try Unix.rename temp dest with
      | Unix.Unix_error _ when Sys.file_exists dest -> remove_temp ()
      | Unix.Unix_error (e, _, _) ->
          remove_temp ();
          failwith
            (Printf.sprintf "%s: cannot rename the download to %s: %s" url
               dest (Unix.error_message e)))
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
      | Some t
        when String.exists (fun c -> c = '\r' || c = '\n' || c = '\000') t ->
          invalid_arg
            "Kaun_hf.download_file: the token holds a CR, an LF or a NUL"
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
    curl_download ~sha256 ~headers:[] ~url ~dest:local ();
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
