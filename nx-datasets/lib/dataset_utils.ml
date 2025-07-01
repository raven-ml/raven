(* dataset_utils.ml *)

(* Set up logging *)
let src = Logs.Src.create "nx.datasets" ~doc:"Nx datasets module"

module Log = (val Logs.src_log src : Logs.LOG)

let () = Curl.global_init Curl.CURLINIT_GLOBALALL

let mkdir_p path perm =
  if path = "" || path = "." || path = Filename.dir_sep then ()
  else
    let components =
      String.split_on_char Filename.dir_sep.[0] path |> List.filter (( <> ) "")
    in
    let is_absolute = path <> "" && path.[0] = Filename.dir_sep.[0] in
    let initial_prefix = if is_absolute then Filename.dir_sep else "." in

    ignore
      (List.fold_left
         (fun current_prefix comp ->
           let next_path =
             if current_prefix = Filename.dir_sep then Filename.dir_sep ^ comp
             else Filename.concat current_prefix comp
           in
           (if Sys.file_exists next_path then (
              if not (Sys.is_directory next_path) then
                failwith
                  (Printf.sprintf "mkdir_p: '%s' exists but is not a directory"
                     next_path))
            else
              try Unix.mkdir next_path perm with
              | Unix.Unix_error (Unix.EEXIST, _, _) ->
                  if not (Sys.is_directory next_path) then
                    failwith
                      (Printf.sprintf
                         "mkdir_p: '%s' appeared as non-directory file after \
                          EEXIST"
                         next_path)
              | Unix.Unix_error (e, fn, arg) ->
                  failwith
                    (Printf.sprintf
                       "mkdir_p: Cannot create directory '%s': %s (%s %s)"
                       next_path (Unix.error_message e) fn arg)
              | ex ->
                  failwith
                    (Printf.sprintf
                       "mkdir_p: Unexpected error creating directory '%s': %s"
                       next_path (Printexc.to_string ex)));
           next_path)
         initial_prefix components);
    ()

module Xdg = struct
  let home =
    try Sys.getenv "HOME"
    with Not_found -> failwith "HOME environment variable not set."

  let cache_base = home ^ "/.cache/ocaml-nx/datasets/"
end

let get_cache_dir dataset_name = Xdg.cache_base ^ dataset_name ^ "/"

let mkdir_p dir =
  try mkdir_p dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ()

let download_file url dest_path =
  let dest_dir = Filename.dirname dest_path in
  mkdir_p dest_dir;
  Log.info (fun m ->
      m "Attempting to download %s to %s" (Filename.basename url) dest_path);
  let h = new Curl.handle in
  h#set_url url;
  (* Follow redirects *)
  h#set_followlocation true;
  (* Set a reasonable timeout *)
  h#set_timeout 300;
  (* 5 minutes *)
  (* Provide a user agent *)
  h#set_useragent "ocaml-nx-datasets/1.0.0";

  let oc = open_out_bin dest_path in
  let result =
    try
      h#set_writefunction (fun s ->
          output_string oc s;
          String.length s);
      h#perform;
      let code = h#get_responsecode in
      if code >= 200 && code < 300 then Ok ()
      else Error (Printf.sprintf "HTTP Error: %d" code)
    with
    | Curl.CurlException (_code, _, msg) ->
        Error (Printf.sprintf "Curl error: %s" msg)
    | exn ->
        Error (Printf.sprintf "Download exception: %s" (Printexc.to_string exn))
  in
  close_out oc;
  h#cleanup;
  match result with
  | Ok () ->
      Log.info (fun m ->
          m "Downloaded %s successfully" (Filename.basename dest_path))
  | Error msg ->
      (* Clean up potentially incomplete file *)
      (try Sys.remove dest_path with Sys_error _ -> ());
      failwith (Printf.sprintf "Failed to download %s: %s" url msg)

let ensure_file url dest_path =
  if not (Sys.file_exists dest_path) then download_file url dest_path
  else Log.debug (fun m -> m "Found file %s" dest_path)

let ensure_extracted_archive ~url ~archive_path ~extract_dir ~check_file =
  let check_file_full_path = Filename.concat extract_dir check_file in
  if not (Sys.file_exists check_file_full_path) then (
    Log.debug (fun m -> m "Extracted file %s not found" check_file_full_path);
    ensure_file url archive_path;

    mkdir_p extract_dir;
    Log.info (fun m -> m "Extracting %s to %s..." archive_path extract_dir);
    (* Basic support for tar.gz *)
    if Filename.check_suffix archive_path ".tar.gz" then (
      let command =
        Printf.sprintf "tar xzf %s -C %s"
          (Filename.quote archive_path)
          (Filename.quote extract_dir)
      in
      Log.debug (fun m -> m "Executing: %s" command);
      let exit_code = Unix.system command in
      if exit_code <> Unix.WEXITED 0 then
        failwith
          (Printf.sprintf "Archive extraction command failed: '%s'" command)
      else Log.info (fun m -> m "Extracted archive successfully")
      (* Verify extraction *))
    else
      failwith
        (Printf.sprintf "Unsupported archive type for %s (only .tar.gz)"
           archive_path);

    if not (Sys.file_exists check_file_full_path) then
      failwith
        (Printf.sprintf "Extraction failed, %s not found after extraction."
           check_file_full_path))
  else Log.debug (fun m -> m "Found extracted file %s" check_file_full_path)

let ensure_decompressed_gz ~gz_path ~target_path =
  if Sys.file_exists target_path then (
    Log.debug (fun m -> m "Found decompressed file %s" target_path);
    true)
  else if Sys.file_exists gz_path then (
    Log.info (fun m -> m "Decompressing %s..." gz_path);
    try
      let ic = Gzip.open_in gz_path in
      let oc = open_out_bin target_path in
      let buf = Bytes.create 4096 in
      let rec loop () =
        let n = Gzip.input ic buf 0 4096 in
        if n > 0 then (
          output oc buf 0 n;
          loop ())
      in
      loop ();
      Gzip.close_in ic;
      close_out oc;
      Log.info (fun m -> m "Decompressed to %s" target_path);
      true
    with Gzip.Error msg ->
      failwith (Printf.sprintf "Gzip error for %s: %s" gz_path msg))
  else (
    Log.warn (fun m -> m "Compressed file %s not found" gz_path);
    false)

let parse_float_cell ~context s =
  try float_of_string s
  with Failure _ | Invalid_argument _ ->
    failwith (Printf.sprintf "Failed to parse float '%s' (%s)" s (context ()))

let parse_int_cell ~context s =
  try int_of_string s
  with Failure _ | Invalid_argument _ ->
    failwith (Printf.sprintf "Failed to parse int '%s' (%s)" s (context ()))
