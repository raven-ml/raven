(* let populate_embedded_libraries libs = let dir = Filename.temp_dir "quill_"
   "" in let write_library (name, cma_contents, cmi_contents) = let cma_path =
   Filename.concat dir (name ^ ".cma") in let cmi_path = Filename.concat dir
   (name ^ ".cmi") in Utils.write_text_to_file cma_path cma_contents;
   Utils.write_text_to_file cmi_path cmi_contents; cma_path in List.map
   write_library libs

   let libraries = populate_embedded_libraries [ ( "ndarray", Option.get
   (Lib.read "ndarray.cma"), Option.get (Lib.read "ndarray.cmi") ); ( "hugin",
   Option.get (Lib.read "hugin.cma"), Option.get (Lib.read "hugin.cmi") ); ] *)

let start path = Quill_server.start path

open Cmdliner

let path_arg =
  Arg.(
    required
    & pos 0 (some string) None
    & info [] ~docv:"PATH"
        ~doc:"Path to a Markdown (.md) file or a directory containing them.")

let file_arg =
  Arg.(
    required
    & pos 0 (some file) None
    & info [] ~docv:"FILE" ~doc:"Markdown file to execute.")

let watch_flag =
  Arg.(value & flag & info [ "w"; "watch" ] ~doc:"Enable watch mode.")

let watch_exec file =
  let id = "exec" in
  Quill_top.initialize_toplevel id;
  let _ = Quill_top.eval ~id {|
#require "ndarray";;
#require "hugin";;
|} in
  let process_and_write () =
    try
      let original_md = Utils.read_text_from_file file in
      let new_md = Exec.process_md id original_md in
      if new_md <> original_md then (
        Utils.write_text_to_file file new_md;
        Printf.printf "Execution completed for '%s'.\n" file)
      else Printf.printf "No changes for '%s'.\n" file;
      Some (Unix.stat file).Unix.st_mtime
    with
    | Sys_error msg ->
        Printf.eprintf "Error: %s\n" msg;
        None
    | Unix.Unix_error (err, _, _) ->
        Printf.eprintf "Unix error: %s\n" (Unix.error_message err);
        None
  in
  let rec loop last_mtime =
    Unix.sleepf 0.1;
    try
      let current_mtime = (Unix.stat file).Unix.st_mtime in
      if current_mtime > last_mtime then
        match process_and_write () with
        | Some new_mtime -> loop new_mtime
        | None -> ()
      else loop last_mtime
    with Unix.Unix_error (err, _, _) ->
      Printf.eprintf "Unix error: %s\n" (Unix.error_message err)
  in
  Printf.printf "Starting watch mode for '%s'.\n" file;
  match process_and_write () with
  | Some initial_mtime -> loop initial_mtime
  | None -> ()

let exec_term =
  let exec watch file =
    if not (Sys.file_exists file) then (
      Printf.eprintf "Error: File '%s' does not exist.\n" file;
      exit 1);
    if not (String.ends_with ~suffix:".md" file) then (
      Printf.eprintf "Error: File '%s' must be a Markdown (.md) file.\n" file;
      exit 1);
    if watch then watch_exec file else Exec.exec file
  in
  Term.(const exec $ watch_flag $ file_arg)

let start_cmd =
  let doc = "Start the Quill web server." in
  Cmd.v (Cmd.info "start" ~doc) Term.(const start $ path_arg)

let exec_cmd =
  let doc =
    "Execute code blocks in a Markdown file and update it with outputs."
  in
  Cmd.v (Cmd.info "exec" ~doc) exec_term

let quill_cmd =
  let doc = "Serve or execute Quill documents." in
  let info = Cmd.info "quill" ~version:"0.1.0" ~doc in
  Cmd.group info [ start_cmd; exec_cmd ]

let () = exit (Cmd.eval quill_cmd)
