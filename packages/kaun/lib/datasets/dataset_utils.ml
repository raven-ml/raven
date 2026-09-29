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
          | _ ->
              (* The user's home: [HOME] on Unix, [USERPROFILE] on Windows. *)
              let home =
                match (getenv "HOME", getenv "USERPROFILE") with
                | Some d, _ when d <> "" -> d
                | _, Some d when d <> "" -> d
                | _ ->
                    failwith "no home directory: set HOME, or RAVEN_CACHE_ROOT"
              in
              Filename.concat home ".cache"
        in
        Filename.concat xdg "raven"
  in
  let path =
    List.fold_left Filename.concat root ("datasets" :: [ dataset_name ])
  in
  let sep = Filename.dir_sep.[0] in
  if path <> "" && path.[String.length path - 1] = sep then path
  else path ^ Filename.dir_sep

(* [run prog args] runs [prog] without a shell, so no argument needs quoting and
   the lookup on [PATH] is the same on every system. It is [None] when [prog] is
   missing: exit code 127 from the forked child on Unix, [ENOENT] from process
   creation on Windows. *)
let run prog args =
  let rec wait pid =
    try snd (Unix.waitpid [] pid)
    with Unix.Unix_error (Unix.EINTR, _, _) -> wait pid
  in
  match
    wait
      (Unix.create_process prog
         (Array.of_list (prog :: args))
         Unix.stdin Unix.stdout Unix.stderr)
  with
  | Unix.WEXITED 127 -> None
  | status -> Some status
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> None

let curl_download ~url ~dest () =
  mkdir_p (Filename.dirname dest);
  match run "curl" [ "-L"; "--fail"; "-s"; "-o"; dest; url ] with
  | None -> failwith "curl not found on PATH"
  | Some (Unix.WEXITED 0) -> ()
  | Some _ ->
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
    Nx_io.gunzip ~src:gz_path ~dst:target_path;
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
    match run "tar" [ "-xzf"; tar_gz_path; "-C"; target_dir ] with
    | Some (Unix.WEXITED 0) ->
        Log.info (fun m -> m "Extracted to %s" target_dir);
        true
    | None ->
        Log.warn (fun m -> m "tar not found on PATH");
        false
    | Some _ ->
        Log.warn (fun m -> m "Failed to extract %s" tar_gz_path);
        false)
  else (
    Log.warn (fun m -> m "Archive %s not found" tar_gz_path);
    false)
