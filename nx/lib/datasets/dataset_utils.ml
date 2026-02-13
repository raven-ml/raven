(* dataset_utils.ml *)

(* Set up logging *)
let src = Logs.Src.create "nx.datasets" ~doc:"Nx datasets module"

module Log = (val Logs.src_log src : Logs.LOG)

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

let get_cache_dir ?(getenv = Sys.getenv_opt) dataset_name =
  Nx_io.Cache_dir.get_path_in_cache ~getenv ~scope:[ "datasets" ] dataset_name

let mkdir_p dir =
  try mkdir_p dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ()

let download_file url dest_path =
  Log.info (fun m ->
      m "Attempting to download %s to %s" (Filename.basename url) dest_path);
  Nx_io.Http.download ~url ~dest:dest_path ();
  Log.info (fun m ->
      m "Downloaded %s successfully" (Filename.basename dest_path))

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

let load_csv ?(separator = ',') ?(has_header = false) path =
  let ic = open_in path in
  let content =
    Fun.protect ~finally:(fun () -> close_in ic) (fun () ->
        really_input_string ic (in_channel_length ic))
  in
  let lines = String.split_on_char '\n' content in
  let lines =
    List.map
      (fun line ->
        if line <> "" && line.[String.length line - 1] = '\r' then
          String.sub line 0 (String.length line - 1)
        else line)
      lines
  in
  let lines = List.filter (fun l -> l <> "") lines in
  let parse_line line =
    let len = String.length line in
    let fields = ref [] in
    let buf = Buffer.create 64 in
    let i = ref 0 in
    while !i < len do
      if line.[!i] = '"' then (
        incr i;
        let in_quotes = ref true in
        while !i < len && !in_quotes do
          if line.[!i] = '"' then
            if !i + 1 < len && line.[!i + 1] = '"' then (
              Buffer.add_char buf '"';
              i := !i + 2)
            else (
              in_quotes := false;
              incr i)
          else (
            Buffer.add_char buf line.[!i];
            incr i)
        done;
        if !i < len && line.[!i] = separator then incr i;
        fields := Buffer.contents buf :: !fields;
        Buffer.clear buf)
      else if line.[!i] = separator then (
        fields := Buffer.contents buf :: !fields;
        Buffer.clear buf;
        incr i)
      else (
        Buffer.add_char buf line.[!i];
        incr i)
    done;
    fields := Buffer.contents buf :: !fields;
    List.rev !fields
  in
  let rows = List.map parse_line lines in
  if has_header then
    match rows with
    | [] -> ([], [])
    | header :: data -> (header, data)
  else ([], rows)
