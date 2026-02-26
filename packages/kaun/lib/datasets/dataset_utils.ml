(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let src = Logs.Src.create "kaun.datasets" ~doc:"Kaun datasets"

module Log = (val Logs.src_log src : Logs.LOG)

let mkdir_p path =
  if path = "" || path = "." || path = Filename.dir_sep then ()
  else
    let components =
      String.split_on_char Filename.dir_sep.[0] path |> List.filter (( <> ) "")
    in
    let is_absolute = path <> "" && path.[0] = Filename.dir_sep.[0] in
    let initial_prefix = if is_absolute then Filename.dir_sep else "." in
    ignore
      (List.fold_left
         (fun prefix comp ->
           let next =
             if prefix = Filename.dir_sep then Filename.dir_sep ^ comp
             else Filename.concat prefix comp
           in
           (if Sys.file_exists next then (
              if not (Sys.is_directory next) then
                failwith
                  (Printf.sprintf "mkdir_p: '%s' exists but is not a directory"
                     next))
            else
              try Unix.mkdir next 0o755
              with Unix.Unix_error (Unix.EEXIST, _, _) ->
                if not (Sys.is_directory next) then
                  failwith
                    (Printf.sprintf
                       "mkdir_p: '%s' appeared as non-directory after EEXIST"
                       next));
           next)
         initial_prefix components)

let get_cache_dir ?(getenv = Sys.getenv_opt) dataset_name =
  let root =
    match getenv "RAVEN_CACHE_ROOT" with
    | Some dir when dir <> "" -> dir
    | _ ->
        let xdg =
          match getenv "XDG_CACHE_HOME" with
          | Some d when d <> "" -> d
          | _ -> Filename.concat (Sys.getenv "HOME") ".cache"
        in
        Filename.concat xdg "raven"
  in
  let path =
    List.fold_left Filename.concat root ("datasets" :: [ dataset_name ])
  in
  let sep = Filename.dir_sep.[0] in
  if path <> "" && path.[String.length path - 1] = sep then path
  else path ^ Filename.dir_sep

let curl_download ~url ~dest () =
  let check =
    lazy (Unix.system "command -v curl >/dev/null 2>&1" = Unix.WEXITED 0)
  in
  if not (Lazy.force check) then failwith "curl not found on PATH";
  mkdir_p (Filename.dirname dest);
  let cmd =
    Printf.sprintf "curl -L --fail -s -o %s %s" (Filename.quote dest)
      (Filename.quote url)
  in
  match Unix.system cmd with
  | Unix.WEXITED 0 -> ()
  | _ ->
      (try Sys.remove dest with Sys_error _ -> ());
      failwith (Printf.sprintf "Failed to download %s" url)

let download_file url dest_path =
  Log.info (fun m -> m "Downloading %s to %s" (Filename.basename url) dest_path);
  curl_download ~url ~dest:dest_path ();
  Log.info (fun m -> m "Downloaded %s" (Filename.basename dest_path))

let ensure_file url dest_path =
  if not (Sys.file_exists dest_path) then download_file url dest_path
  else Log.debug (fun m -> m "Found %s" dest_path)

let ensure_decompressed_gz ~gz_path ~target_path =
  if Sys.file_exists target_path then (
    Log.debug (fun m -> m "Found %s" target_path);
    true)
  else if Sys.file_exists gz_path then (
    Log.info (fun m -> m "Decompressing %s..." gz_path);
    let ic = Gzip.open_in gz_path in
    let oc = open_out_bin target_path in
    Fun.protect
      ~finally:(fun () ->
        Gzip.close_in ic;
        close_out oc)
      (fun () ->
        let buf = Bytes.create 4096 in
        let rec loop () =
          let n = Gzip.input ic buf 0 4096 in
          if n > 0 then (
            output oc buf 0 n;
            loop ())
        in
        loop ());
    Log.info (fun m -> m "Decompressed to %s" target_path);
    true)
  else (
    Log.warn (fun m -> m "Compressed file %s not found" gz_path);
    false)

let ensure_extracted_tar_gz ~tar_gz_path ~target_dir ~check_file =
  if Sys.file_exists check_file then (
    Log.debug (fun m -> m "Found %s" check_file);
    true)
  else if Sys.file_exists tar_gz_path then (
    Log.info (fun m -> m "Extracting %s..." tar_gz_path);
    mkdir_p target_dir;
    let cmd =
      Printf.sprintf "tar -xzf %s -C %s"
        (Filename.quote tar_gz_path)
        (Filename.quote target_dir)
    in
    match Unix.system cmd with
    | Unix.WEXITED 0 ->
        Log.info (fun m -> m "Extracted to %s" target_dir);
        true
    | _ ->
        Log.warn (fun m -> m "Failed to extract %s" tar_gz_path);
        false)
  else (
    Log.warn (fun m -> m "Archive %s not found" tar_gz_path);
    false)
