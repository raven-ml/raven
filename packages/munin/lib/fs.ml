let is_directory path =
  try (Unix.stat path).Unix.st_kind = Unix.S_DIR
  with Unix.Unix_error _ -> false

let ensure_dir path =
  let rec loop current =
    if current = "" || current = Filename.dir_sep then ()
    else if Sys.file_exists current then ()
    else (
      loop (Filename.dirname current);
      Unix.mkdir current 0o755)
  in
  loop path

let list_entries path =
  if Sys.file_exists path && is_directory path then
    Sys.readdir path |> Array.to_list |> List.sort String.compare
  else []

let list_dirs path =
  List.filter
    (fun name -> is_directory (Filename.concat path name))
    (list_entries path)

let read_file path =
  let ic = open_in_bin path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

let write_file path text =
  ensure_dir (Filename.dirname path);
  let oc = open_out_bin path in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc text)

let append_line path line =
  ensure_dir (Filename.dirname path);
  let oc = open_out_gen [ Open_creat; Open_append; Open_binary ] 0o644 path in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () ->
      output_string oc line;
      output_char oc '\n')

let copy_file src dst =
  ensure_dir (Filename.dirname dst);
  let ic = open_in_bin src in
  let oc = open_out_bin dst in
  let buffer = Bytes.create 65536 in
  Fun.protect
    ~finally:(fun () ->
      close_in ic;
      close_out oc)
    (fun () ->
      let rec loop () =
        let count = input ic buffer 0 (Bytes.length buffer) in
        if count > 0 then (
          output oc buffer 0 count;
          loop ())
      in
      loop ())

let rec copy_tree src dst =
  if is_directory src then (
    ensure_dir dst;
    List.iter
      (fun name ->
        copy_tree (Filename.concat src name) (Filename.concat dst name))
      (list_entries src))
  else copy_file src dst

let rec remove_tree path =
  if is_directory path then (
    List.iter
      (fun name -> remove_tree (Filename.concat path name))
      (list_entries path);
    Unix.rmdir path)
  else Sys.remove path

let rec iter_tree ?(rel = "") root f =
  let path = if rel = "" then root else Filename.concat root rel in
  if is_directory path then (
    f rel `Dir;
    List.iter
      (fun name ->
        let child = if rel = "" then name else Filename.concat rel name in
        iter_tree ~rel:child root f)
      (list_entries path))
  else f rel `File

let sha256_file path = Sha256.to_hex (Sha256.file_fast path)

let sha256_path path =
  if is_directory path then (
    let ctx = Sha256.init () in
    iter_tree path (fun rel kind ->
        if rel <> "" then
          match kind with
          | `Dir ->
              Sha256.update_string ctx "dir:";
              Sha256.update_string ctx rel;
              Sha256.update_string ctx "\n"
          | `File ->
              Sha256.update_string ctx "file:";
              Sha256.update_string ctx rel;
              Sha256.update_string ctx ":";
              Sha256.update_string ctx (sha256_file (Filename.concat path rel));
              Sha256.update_string ctx "\n");
    Sha256.to_hex (Sha256.finalize ctx))
  else sha256_file path

let file_size path =
  try (Unix.stat path).Unix.st_size with Unix.Unix_error _ -> 0

let rec path_size path =
  if is_directory path then
    list_entries path
    |> List.fold_left
         (fun acc name -> acc + path_size (Filename.concat path name))
         0
  else file_size path

let command_output command =
  let ic = Unix.open_process_in (command ^ " 2>/dev/null") in
  let output =
    Fun.protect
      ~finally:(fun () -> ignore (Unix.close_process_in ic))
      (fun () ->
        let rec loop acc =
          match input_line ic with
          | line -> loop (line :: acc)
          | exception End_of_file -> List.rev acc
        in
        String.concat "\n" (loop []))
  in
  if output = "" then None else Some output
