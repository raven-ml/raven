let serve path = Quill_server.start path

open Cmdliner

let path_arg =
  Arg.(
    required
    & pos 0 (some string) None
    & info [] ~docv:"PATH"
        ~doc:"Path to a Markdown (.md) file or a directory containing them.")

let serve_cmd =
  let doc = "Start the Quill web server." in
  Cmd.v (Cmd.info "serve" ~doc) Term.(const serve $ path_arg)

let quill_cmd =
  let doc = "Serve or execute Quill documents." in
  let info = Cmd.info "quill" ~version:"0.1.0" ~doc in
  Cmd.group info [ serve_cmd ]

let () = exit (Cmd.eval quill_cmd)
