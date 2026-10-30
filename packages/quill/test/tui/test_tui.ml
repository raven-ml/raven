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
            ("F1", "Help");
          ]
          (footer Quill_tui.Editing ~running:true));
  ]

(* Saving *)

let write path s = Out_channel.with_open_bin path (fun oc -> output_string oc s)
let read path = In_channel.with_open_bin path In_channel.input_all
let doc = Doc.of_cells [ Cell.code ~id:"c_x" "1" ]

(* A notebook saved by the terminal notebook, then replaced on disk by [edit],
   which the terminal notebook has not reloaded. *)
let changed edit =
  let path = Filename.concat (temp_dir ()) "nb.md" in
  let stamp = require_ok (Quill_markdown.save Inline path doc) in
  edit path;
  (path, Some stamp)

let save_tests =
  [
    test "a save follows the save before it" (fun () ->
        let path, stamp = changed ignore in
        let stamp =
          require_ok (Quill_tui.save ~stamp Quill_markdown.Inline path doc)
        in
        equal (result unit string) (Ok ())
          (Result.map ignore
             (Quill_tui.save ~stamp:(Some stamp) Quill_markdown.Inline path doc)));
    cases "a save refuses a file changed on disk" ~id:fst
      [
        ( "by a copy that kept an older time",
          fun path ->
            let t = (Unix.stat path).st_mtime in
            write path "# Theirs\n";
            Unix.utimes path (t -. 60.) (t -. 60.) );
        ( "at the same time",
          fun path ->
            let t = (Unix.stat path).st_mtime in
            write path "# Theirs\n";
            Unix.utimes path t t );
      ]
      (fun (_, edit) ->
        let path, stamp = changed edit in
        equal (result unit string)
          (Error "nb.md changed on disk and is not reloaded yet")
          (Result.map ignore
             (Quill_tui.save ~stamp Quill_markdown.Inline path doc));
        equal string "# Theirs\n" (read path));
  ]

let () =
  exit
    (run "Quill_tui" [ group "footer" footer_tests; group "save" save_tests ])
