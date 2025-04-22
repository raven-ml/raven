let start path = Quill_server.start path
let exec path = Exec.exec path

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

let start_cmd =
  let doc = "Start the Quill web server." in
  Cmd.v (Cmd.info "start" ~doc) Term.(const start $ path_arg)

let exec_cmd =
  let doc =
    "Execute code blocks in a Markdown file and update it with outputs."
  in
  Cmd.v (Cmd.info "exec" ~doc) Term.(const exec $ file_arg)

let quill_cmd =
  let doc = "Serve or execute Quill documents." in
  let info = Cmd.info "quill" ~version:"0.1.0" ~doc in
  Cmd.group info [ start_cmd; exec_cmd ]

let () = exit (Cmd.eval quill_cmd)
