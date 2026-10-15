(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Quill

(* A notebook of one code cell, running or idle. *)
let session ~running =
  let c = Cell.code "1" in
  let s = Session.create (Doc.of_cells [ c ]) in
  if running then Session.mark_running (Cell.id c) s else s

let footer mode ~running =
  List.map
    (fun (a : Quill_tui.footer_action) -> (a.key, a.label))
    (Quill_tui.footer_actions mode (session ~running))

let actions = list (pair string string)

let footer_tests =
  [
    test "an idle notebook offers to run the focused cell" (fun () ->
        equal actions
          [
            ("Enter", "Edit"); ("x", "Run"); ("j/k", "Navigate"); ("?", "Help");
          ]
          (footer Quill_tui.Normal ~running:false));
    test "a running cell puts Ctrl-C Interrupt in place of Run" (fun () ->
        equal actions
          [
            ("Enter", "Edit");
            ("Ctrl-C", "Interrupt");
            ("j/k", "Navigate");
            ("?", "Help");
          ]
          (footer Quill_tui.Normal ~running:true));
    test "so it does while a cell is edited" (fun () ->
        equal actions
          [
            ("Ctrl-C", "Interrupt");
            ("Tab", "Complete");
            ("Esc", "Exit");
            ("?", "Help");
          ]
          (footer Quill_tui.Editing ~running:true));
  ]

let () = exit (run "Quill_tui" [ group "footer" footer_tests ])
