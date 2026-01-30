(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Quill_editor
module MD = Quill_editor.Document

let effect_pp fmt = function
  | Effect.Execute_code { block_id; code } ->
      Format.fprintf fmt "Execute_code(%d,%S)" block_id code
  | Effect.Load_document { path } -> Format.fprintf fmt "Load_document(%S)" path
  | Effect.Save_document { path; content = _ } ->
      Format.fprintf fmt "Save_document(%S)" path
  | Effect.Copy_to_clipboard { text } ->
      Format.fprintf fmt "Copy_to_clipboard(%S)" text
  | Effect.Cut_to_clipboard { text } ->
      Format.fprintf fmt "Cut_to_clipboard(%S)" text
  | Effect.Request_clipboard_paste ->
      Format.pp_print_string fmt "Request_clipboard_paste"
  | Effect.Notify { level; message } ->
      Format.fprintf fmt "Notify(%s,%S)"
        (match level with
        | `Info -> "info"
        | `Warning -> "warning"
        | `Error -> "error")
        message

let effect_equal a b =
  match (a, b) with
  | Effect.Execute_code a, Effect.Execute_code b ->
      a.block_id = b.block_id && String.equal a.code b.code
  | Effect.Load_document a, Effect.Load_document b -> String.equal a.path b.path
  | Effect.Save_document a, Effect.Save_document b ->
      String.equal a.path b.path && String.equal a.content b.content
  | Effect.Copy_to_clipboard a, Effect.Copy_to_clipboard b ->
      String.equal a.text b.text
  | Effect.Cut_to_clipboard a, Effect.Cut_to_clipboard b ->
      String.equal a.text b.text
  | Effect.Request_clipboard_paste, Effect.Request_clipboard_paste -> true
  | Effect.Notify a, Effect.Notify b ->
      a.level = b.level && String.equal a.message b.message
  | _ -> false

let effect_testable = Alcotest.testable effect_pp effect_equal

let doc_of_markdown md =
  MD.reset_ids ();
  Document.of_markdown md

let first_block doc =
  match doc with
  | blk :: _ -> blk
  | [] -> Alcotest.fail "expected at least one block"

let second_block doc =
  match doc with
  | _ :: blk :: _ -> blk
  | _ -> Alcotest.fail "expected at least two blocks"

let third_block doc =
  match doc with
  | _ :: _ :: blk :: _ -> blk
  | _ -> Alcotest.fail "expected at least three blocks"

let paragraph_inline block =
  match block.MD.content with
  | MD.Paragraph inline -> inline
  | _ -> Alcotest.fail "expected paragraph block"

let code_block block =
  match block.MD.content with
  | MD.Codeblock data -> data
  | _ -> Alcotest.fail "expected code block"

let caret block_id = { State.block_id; inline_id = None; offset = 0 }

let test_focus_inline_command () =
  let doc = doc_of_markdown "Text" in
  let block = first_block doc in
  let inline = paragraph_inline block in
  let state = State.with_document doc in
  let state', effects =
    Reducer.apply_command state (Command.Focus_inline inline.MD.id)
  in
  Alcotest.(check int) "no effects" 0 (List.length effects);
  let updated_block = first_block state'.State.document in
  let updated_inline = paragraph_inline updated_block in
  Alcotest.(check bool) "block focused" true updated_block.MD.focused;
  Alcotest.(check bool) "inline focused" true updated_inline.MD.focused;
  match state'.State.selection with
  | State.Caret c -> Alcotest.(check int) "caret block" block.MD.id c.block_id
  | _ -> Alcotest.fail "expected caret selection"

let test_request_code_execution_command () =
  let doc = doc_of_markdown "```ocaml\nprint_endline \"hi\"\n```" in
  let block = first_block doc in
  let code = (code_block block).code in
  let block_id = block.MD.id in
  let state = State.with_document doc in
  let state', effects =
    Reducer.apply_command state
      (Command.Request_code_execution { block_id; code })
  in
  Alcotest.(check bool)
    "block running" true
    (State.is_block_running state' block_id);
  Alcotest.(check (list effect_testable))
    "effect"
    [ Effect.Execute_code { block_id; code } ]
    effects

let test_undo_redo () =
  let doc = doc_of_markdown "Paragraph" in
  let block = first_block doc in
  let block_id = block.MD.id in
  let state = State.with_document doc in
  let state', _ =
    Reducer.apply_command state (Command.Replace_block_with_codeblock block_id)
  in
  (match (first_block state'.State.document).MD.content with
  | MD.Codeblock _ -> ()
  | _ -> Alcotest.fail "expected codeblock after replace");
  let state'', _ = Reducer.apply_command state' Command.Undo in
  (match (first_block state''.State.document).MD.content with
  | MD.Paragraph _ -> ()
  | _ -> Alcotest.fail "expected paragraph after undo");
  let state''', _ = Reducer.apply_command state'' Command.Redo in
  match (first_block state'''.State.document).MD.content with
  | MD.Codeblock _ -> ()
  | _ -> Alcotest.fail "expected codeblock after redo"

let test_request_copy_selection () =
  let doc = doc_of_markdown "First\n\nSecond" in
  let first_block_id = (first_block doc).MD.id in
  let second_block_id = (second_block doc).MD.id in
  let state = State.with_document doc in
  let selection =
    State.Range { anchor = caret first_block_id; focus = caret second_block_id }
  in
  let state = State.set_selection state selection in
  let expected =
    Document.slice_between doc ~start_id:first_block_id ~end_id:second_block_id
    |> Document.to_markdown
  in
  let _, effects = Reducer.apply_command state Command.Request_copy_selection in
  Alcotest.(check (list effect_testable))
    "copy effect"
    [ Effect.Copy_to_clipboard { text = expected } ]
    effects

let code_of_first_block state =
  match (first_block state.State.document).MD.content with
  | MD.Codeblock { code; _ } -> code
  | _ -> Alcotest.fail "expected codeblock in first position"

let test_history_capacity () =
  let doc = doc_of_markdown "Paragraph" in
  let block = first_block doc in
  let block_id = block.MD.id in
  let config = { State.history_limit = 2; auto_normalize = false } in
  let state = State.create ~config ~document:doc () in
  let state, _ =
    Reducer.apply_command state (Command.Replace_block_with_codeblock block_id)
  in
  let state, _ =
    Reducer.apply_command state
      (Command.Update_codeblock { block_id; code = "foo" })
  in
  let state, _ =
    Reducer.apply_command state
      (Command.Update_codeblock { block_id; code = "bar" })
  in
  Alcotest.(check string) "latest code" "bar" (code_of_first_block state);
  let state, _ = Reducer.apply_command state Command.Undo in
  Alcotest.(check string) "undo back to foo" "foo" (code_of_first_block state);
  let state, _ = Reducer.apply_command state Command.Undo in
  Alcotest.(check string) "undo back to empty" "" (code_of_first_block state);
  let state_final, _ = Reducer.apply_command state Command.Undo in
  Alcotest.(check bool) "history exhausted" false (State.has_undo state_final);
  Alcotest.(check string) "remains empty" "" (code_of_first_block state_final)

let test_selection_restored_after_undo () =
  let doc = doc_of_markdown "Paragraph" in
  let block = first_block doc in
  let block_id = block.MD.id in
  let caret = caret block_id in
  let state =
    let state = State.with_document doc in
    State.set_selection state (State.Caret caret)
  in
  let state, _ =
    Reducer.apply_command state (Command.Replace_block_with_codeblock block_id)
  in
  let state, _ = Reducer.apply_command state Command.Undo in
  match state.State.selection with
  | State.Caret c -> Alcotest.(check int) "caret restored" block_id c.block_id
  | _ -> Alcotest.fail "expected caret selection after undo"

let test_selection_blocks_reverse () =
  let doc = doc_of_markdown "One\n\nTwo\n\nThree" in
  let b1 = first_block doc in
  let b2 = second_block doc in
  let b3 = third_block doc in
  let selection =
    State.Range { anchor = caret b3.MD.id; focus = caret b1.MD.id }
  in
  let state =
    let state = State.with_document doc in
    State.set_selection state selection
  in
  let ids = State.selection_blocks state in
  Alcotest.(check (list int))
    "selection spans ascending blocks"
    [ b1.MD.id; b2.MD.id; b3.MD.id ]
    ids

let tests =
  [
    Alcotest.test_case "focus_inline command" `Quick test_focus_inline_command;
    Alcotest.test_case "request_code_execution" `Quick
      test_request_code_execution_command;
    Alcotest.test_case "undo/redo" `Quick test_undo_redo;
    Alcotest.test_case "request_copy_selection" `Quick
      test_request_copy_selection;
    Alcotest.test_case "history capacity" `Quick test_history_capacity;
    Alcotest.test_case "selection restored after undo" `Quick
      test_selection_restored_after_undo;
    Alcotest.test_case "selection blocks reverse order" `Quick
      test_selection_blocks_reverse;
  ]

let () = Alcotest.run "quill.editor.reducer" [ ("reducer", tests) ]
