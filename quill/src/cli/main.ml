let serve path = Quill_server.start path

let get_file_mtime path =
  try Some (Unix.stat path).Unix.st_mtime with Unix.Unix_error _ -> None

let rec watch_file path last_mtime eval_fn =
  Unix.sleep 1;
  (* Check every second *)
  match get_file_mtime path with
  | None ->
      Printf.eprintf "File %s no longer exists\n" path;
      exit 1
  | Some mtime when mtime > last_mtime ->
      Printf.printf "\n[%s] File changed, re-evaluating...\n"
        ( Unix.gettimeofday () |> Unix.localtime |> fun tm ->
          Printf.sprintf "%02d:%02d:%02d" tm.Unix.tm_hour tm.Unix.tm_min
            tm.Unix.tm_sec );
      eval_fn ();
      (* Get the new mtime after evaluation (important for --inplace mode) *)
      let new_mtime =
        match get_file_mtime path with
        | Some t -> t
        | None -> mtime (* Shouldn't happen but fallback to old mtime *)
      in
      watch_file path new_mtime eval_fn
  | Some _ -> watch_file path last_mtime eval_fn

let eval_cmd inplace watch path_opt =
  match path_opt with
  | None -> (
      if inplace then (
        Printf.eprintf "Error: --inplace requires a file path\n";
        exit 1);
      if watch then (
        Printf.eprintf "Error: --watch requires a file path\n";
        exit 1);
      match Eval.eval_stdin () with
      | Ok result -> print_string result
      | Error err ->
          Printf.eprintf "Error: %s\n" err;
          exit 1)
  | Some path -> (
      let eval_fn () =
        match Eval.eval_file path with
        | Ok result ->
            if inplace then (
              (* Write result back to the file *)
              try
                let oc = open_out path in
                output_string oc result;
                close_out oc;
                Printf.printf "Updated %s\n" path
              with Sys_error msg ->
                Printf.eprintf "Error writing file: %s\n" msg;
                exit 1)
            else print_string result
        | Error err ->
            Printf.eprintf "Error: %s\n" err;
            if not watch then exit 1
      in

      (* Initial evaluation *)
      eval_fn ();

      (* Start watching if requested *)
      if watch then
        match get_file_mtime path with
        | None ->
            Printf.eprintf "Error: Cannot watch %s\n" path;
            exit 1
        | Some mtime ->
            Printf.printf
              "\nWatching %s for changes... (Press Ctrl+C to stop)\n" path;
            watch_file path mtime eval_fn)

open Cmdliner

let path_arg =
  Arg.(
    required
    & pos 0 (some string) None
    & info [] ~docv:"PATH"
        ~doc:"Path to a Markdown (.md) file or a directory containing them.")

let eval_path_arg =
  Arg.(
    value
    & pos 0 (some string) None
    & info [] ~docv:"FILE"
        ~doc:
          "Path to a Markdown file to evaluate. If not provided, reads from \
           stdin.")

let serve_cmd =
  let doc = "Start the Quill web server." in
  Cmd.v (Cmd.info "serve" ~doc) Term.(const serve $ path_arg)

let inplace_flag =
  Arg.(
    value & flag
    & info [ "inplace"; "i" ]
        ~doc:"Replace the original file with the evaluated output.")

let watch_flag =
  Arg.(
    value & flag
    & info [ "watch"; "w" ]
        ~doc:"Watch the file for changes and re-evaluate automatically.")

let eval_cmd =
  let doc = "Evaluate code blocks in a Markdown file." in
  Cmd.v (Cmd.info "eval" ~doc)
    Term.(const eval_cmd $ inplace_flag $ watch_flag $ eval_path_arg)

let quill_cmd =
  let doc = "Serve or execute Quill documents." in
  let info = Cmd.info "quill" ~version:"1.0.0" ~doc in
  Cmd.group info [ serve_cmd; eval_cmd ]

let () = exit (Cmd.eval quill_cmd)
