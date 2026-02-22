(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Windtrap
open Quill

let basic_tests =
  [
    test "create session" (fun () ->
        let doc = Doc.of_cells [ Cell.text "hello" ] in
        let s = Session.create doc in
        equal int 1 (Doc.length (Session.doc s)));
    test "update source" (fun () ->
        let c = Cell.text "old" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.update_source (Cell.id c) "new" s in
        match Doc.find (Cell.id c) (Session.doc s) with
        | Some c -> equal string "new" (Cell.source c)
        | None -> fail "cell not found");
    test "insert cell" (fun () ->
        let doc = Doc.of_cells [ Cell.text "a" ] in
        let s = Session.create doc in
        let new_cell = Cell.text "b" in
        let s = Session.insert_cell ~pos:1 new_cell s in
        equal int 2 (Doc.length (Session.doc s)));
    test "remove cell" (fun () ->
        let c1 = Cell.text "a" in
        let c2 = Cell.text "b" in
        let doc = Doc.of_cells [ c1; c2 ] in
        let s = Session.create doc in
        let s = Session.remove_cell (Cell.id c1) s in
        equal int 1 (Doc.length (Session.doc s)));
    test "move cell" (fun () ->
        let c1 = Cell.text "a" in
        let c2 = Cell.text "b" in
        let c3 = Cell.text "c" in
        let doc = Doc.of_cells [ c1; c2; c3 ] in
        let s = Session.create doc in
        let s = Session.move_cell (Cell.id c3) ~pos:0 s in
        match Doc.nth 0 (Session.doc s) with
        | Some c -> equal string "c" (Cell.source c)
        | None -> fail "expected Some");
    test "set cell kind" (fun () ->
        let c = Cell.text "code here" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.set_cell_kind (Cell.id c) `Code s in
        match Doc.nth 0 (Session.doc s) with
        | Some (Cell.Code _) -> ()
        | _ -> fail "expected Code cell");
    test "clear outputs" (fun () ->
        let c = Cell.code "x" |> Cell.set_outputs [ Cell.Stdout "out" ] in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.clear_outputs (Cell.id c) s in
        match Doc.find (Cell.id c) (Session.doc s) with
        | Some (Cell.Code { outputs; _ }) -> equal int 0 (List.length outputs)
        | _ -> fail "expected Code cell");
    test "clear all outputs" (fun () ->
        let c1 = Cell.code "x" |> Cell.set_outputs [ Cell.Stdout "out1" ] in
        let c2 = Cell.code "y" |> Cell.set_outputs [ Cell.Stdout "out2" ] in
        let doc = Doc.of_cells [ c1; c2 ] in
        let s = Session.create doc in
        let s = Session.clear_all_outputs s in
        List.iter
          (fun cell ->
            match cell with
            | Cell.Code { outputs; _ } -> equal int 0 (List.length outputs)
            | _ -> ())
          (Doc.cells (Session.doc s)));
  ]

let execution_state_tests =
  [
    test "mark running" (fun () ->
        let c = Cell.code "let x = 1" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.mark_running (Cell.id c) s in
        match Session.cell_status (Cell.id c) s with
        | Session.Running -> ()
        | _ -> fail "expected Running");
    test "mark queued" (fun () ->
        let c = Cell.code "let x = 1" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.mark_queued (Cell.id c) s in
        match Session.cell_status (Cell.id c) s with
        | Session.Queued -> ()
        | _ -> fail "expected Queued");
    test "apply output and finish" (fun () ->
        let c = Cell.code "let x = 1" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.mark_running (Cell.id c) s in
        let s = Session.apply_output (Cell.id c) (Cell.Stdout "val x = 1") s in
        let s =
          Session.apply_output (Cell.id c) (Cell.Stdout "more output") s
        in
        let s = Session.finish_execution (Cell.id c) ~success:true s in
        (match Session.cell_status (Cell.id c) s with
        | Session.Idle -> ()
        | _ -> fail "expected Idle after finish");
        match Doc.find (Cell.id c) (Session.doc s) with
        | Some (Cell.Code { outputs; execution_count; _ }) ->
            equal int 2 (List.length outputs);
            equal int 1 execution_count
        | _ -> fail "expected Code cell with outputs");
    test "default status is idle" (fun () ->
        let c = Cell.code "x" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        match Session.cell_status (Cell.id c) s with
        | Session.Idle -> ()
        | _ -> fail "expected Idle");
  ]

let undo_redo_tests =
  [
    test "update_source does not push history" (fun () ->
        let c = Cell.text "original" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.update_source (Cell.id c) "changed" s in
        is_false ~msg:"no undo without checkpoint" (Session.can_undo s));
    test "checkpoint enables undo" (fun () ->
        let c = Cell.text "original" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        is_false ~msg:"no undo initially" (Session.can_undo s);
        let s = Session.update_source (Cell.id c) "changed" s in
        let s = Session.checkpoint s in
        is_true ~msg:"can undo after checkpoint" (Session.can_undo s);
        let s = Session.undo s in
        (match Doc.find (Cell.id c) (Session.doc s) with
        | Some c -> equal string "original" (Cell.source c)
        | None -> fail "cell not found");
        is_true ~msg:"can redo" (Session.can_redo s));
    test "redo after undo" (fun () ->
        let c = Cell.text "original" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.update_source (Cell.id c) "changed" s in
        let s = Session.checkpoint s in
        let s = Session.undo s in
        let s = Session.redo s in
        match Doc.find (Cell.id c) (Session.doc s) with
        | Some c -> equal string "changed" (Cell.source c)
        | None -> fail "cell not found");
    test "structural ops auto-checkpoint" (fun () ->
        let c = Cell.text "a" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        is_false ~msg:"no undo initially" (Session.can_undo s);
        let s = Session.insert_cell ~pos:1 (Cell.text "b") s in
        is_true ~msg:"can undo after insert" (Session.can_undo s);
        let s = Session.undo s in
        equal int 1 (Doc.length (Session.doc s)));
    test "checkpoint is noop when unchanged" (fun () ->
        let doc = Doc.of_cells [ Cell.text "a" ] in
        let s = Session.create doc in
        let s = Session.checkpoint s in
        is_false ~msg:"no undo after noop checkpoint" (Session.can_undo s));
    test "undo on empty history is noop" (fun () ->
        let doc = Doc.of_cells [ Cell.text "a" ] in
        let s = Session.create doc in
        let s2 = Session.undo s in
        equal int (Doc.length (Session.doc s)) (Doc.length (Session.doc s2)));
    test "reload clears history" (fun () ->
        let c = Cell.text "original" in
        let doc = Doc.of_cells [ c ] in
        let s = Session.create doc in
        let s = Session.update_source (Cell.id c) "changed" s in
        let s = Session.checkpoint s in
        is_true ~msg:"can undo before reload" (Session.can_undo s);
        let new_doc = Doc.of_cells [ Cell.text "reloaded" ] in
        let s = Session.reload new_doc s in
        is_false ~msg:"no undo after reload" (Session.can_undo s);
        equal int 1 (Doc.length (Session.doc s)));
  ]

let () =
  run "Session"
    [
      group "Basic" basic_tests;
      group "Execution state" execution_state_tests;
      group "Undo/Redo" undo_redo_tests;
    ]
