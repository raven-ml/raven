let serve path = Quill_server.start path

let eval_cmd path_opt =
  match path_opt with
  | None -> (
      match Eval.eval_stdin () with
      | Ok result -> print_string result
      | Error err ->
          Printf.eprintf "Error: %s\n" err;
          exit 1)
  | Some path -> (
      match Eval.eval_file path with
      | Ok result -> print_string result
      | Error err ->
          Printf.eprintf "Error: %s\n" err;
          exit 1)

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

let eval_cmd =
  let doc = "Evaluate code blocks in a Markdown file." in
  Cmd.v (Cmd.info "eval" ~doc) Term.(const eval_cmd $ eval_path_arg)

let quill_cmd =
  let doc = "Serve or execute Quill documents." in
  let info = Cmd.info "quill" ~version:"1.0.0" ~doc in
  Cmd.group info [ serve_cmd; eval_cmd ]

let () = exit (Cmd.eval quill_cmd)
