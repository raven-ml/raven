(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let () =
  let open Cmdliner in
  let base_dir =
    Arg.(value & opt (some string) None & info [ "base-dir" ] ~docv:"DIR")
  in
  let run_id =
    Arg.(value & pos 0 (some string) None & info [] ~docv:"RUN_ID")
  in
  let cmd =
    Cmd.v (Cmd.info "kaun-board" ~doc:"Training dashboard TUI")
    @@ Term.(
         const (fun base_dir run ->
             let runs = Option.map (fun r -> [ r ]) run in
             Kaun_board_tui.run ?base_dir ?runs ())
         $ base_dir $ run_id)
  in
  exit (Cmd.eval cmd)
