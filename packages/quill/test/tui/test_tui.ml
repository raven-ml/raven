(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Quill
module T = Quill_tui_internal.Quill_tui

let kernel =
  {
    Kernel.execute = (fun ~cell_id:_ ~code:_ -> ());
    interrupt = ignore;
    complete = (fun ~code:_ ~pos:_ -> []);
    type_at = None;
    diagnostics = None;
    is_complete = None;
    status = (fun () -> Kernel.Idle);
    shutdown = ignore;
  }

(* A notebook of one code cell, running or idle, in [mode]. *)
let model ~running mode =
  let path = Filename.temp_file "quill_tui" ".md" in
  let m, _ = T.init ~create_kernel:(fun ~on_event:_ -> kernel) ~path () in
  Sys.remove path;
  let c = Cell.code "1" in
  let session = Session.create (Doc.of_cells [ c ]) in
  let session =
    if running then Session.mark_running (Cell.id c) session else session
  in
  { m with T.session; mode }

let footer m =
  List.map (fun (a : T.footer_action) -> (a.key, a.label)) (T.footer_actions m)

let actions = list (pair string string)

let footer_tests =
  [
    test "an idle notebook offers to run the focused cell" (fun () ->
        equal actions
          [
            ("Enter", "Edit"); ("x", "Run"); ("j/k", "Navigate"); ("?", "Help");
          ]
          (footer (model ~running:false T.Normal)));
    test "a running cell puts Ctrl-C Interrupt in place of Run" (fun () ->
        equal actions
          [
            ("Enter", "Edit");
            ("Ctrl-C", "Interrupt");
            ("j/k", "Navigate");
            ("?", "Help");
          ]
          (footer (model ~running:true T.Normal)));
    test "so it does while a cell is edited" (fun () ->
        equal actions
          [
            ("Ctrl-C", "Interrupt");
            ("Tab", "Complete");
            ("Esc", "Exit");
            ("?", "Help");
          ]
          (footer (model ~running:true T.Editing)));
  ]

let () = exit (run "Quill_tui" [ group "footer" footer_tests ])
