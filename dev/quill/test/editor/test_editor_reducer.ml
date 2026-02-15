(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Quill_editor
open Windtrap
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

let effect_testable =
  Testable.make ~pp:effect_pp ~equal:effect_equal ()

let doc_of_markdown md =
  MD.reset_ids ();
  Document.of_markdown md

let first_block doc =
  match doc with
  | blk :: _ -> blk
  | [] -> fail "expected at least one block"

let second_block doc =
  match doc with
  | _ :: blk :: _ -> blk
  | _ -> fail "expected at least two blocks"

let third_block doc =
  match doc with
  | _ :: _ :: blk :: _ -> blk
  | _ -> fail "expected at least three blocks"

let paragraph_inline block =
  match block.MD.content with
  | MD.Paragraph inline -> inline
  | _ -> fail "expected paragraph block"

let code_block block =
  match block.MD.content with
  | MD.Codeblock data -> data
  | _ -> fail "expected code block"

let caret block_id = { State.block_id; inline_id = None; offset = 0 }

let test_focus_inline_command () =
  let doc = doc_of_markdown "Text" in
  let block = first_block doc in
  let inline = paragraph_inline block in
  let state = State.with_document doc in
  let state', effects =
    Reducer.apply_command state (Command.Focus_inline inline.MD.id)
  in
  equal ~msg:"no effects" int 0 (List.length effects);
  let updated_block = first_block state'.State.document in
  let updated_inline = paragraph_inline updated_block in
  equal ~msg:"block focused" bool true updated_block.MD.focused;
  equal ~msg:"inline focused" bool true updated_inline.MD.focused;
  match state'.State.selection with
  | State.Caret c -> equal ~msg:"caret block" int block.MD.id c.block_id
  | _ -> fail "expected caret selection"

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
  equal ~msg:"block running" bool true
    (State.is_block_running state' block_id);
  equal ~msg:"effect"
    (list effect_testable)
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
  | _ -> fail "expected codeblock after replace");
  let state'', _ = Reducer.apply_command state' Command.Undo in
  (match (first_block state''.State.document).MD.content with
  | MD.Paragraph _ -> ()
  | _ -> fail "expected paragraph after undo");
  let state''', _ = Reducer.apply_command state'' Command.Redo in
  match (first_block state'''.State.document).MD.content with
  | MD.Codeblock _ -> ()
  | _ -> fail "expected codeblock after redo"

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
  equal ~msg:"copy effect"
    (list effect_testable)
    [ Effect.Copy_to_clipboard { text = expected } ]
    effects

let code_of_first_block state =
  match (first_block state.State.document).MD.content with
  | MD.Codeblock { code; _ } -> code
  | _ -> fail "expected codeblock in first position"

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
  equal ~msg:"latest code" string "bar" (code_of_first_block state);
  let state, _ = Reducer.apply_command state Command.Undo in
  equal ~msg:"undo back to foo" string "foo" (code_of_first_block state);
  let state, _ = Reducer.apply_command state Command.Undo in
  equal ~msg:"undo back to empty" string "" (code_of_first_block state);
  let state_final, _ = Reducer.apply_command state Command.Undo in
  equal ~msg:"history exhausted" bool false (State.has_undo state_final);
  equal ~msg:"remains empty" string "" (code_of_first_block state_final)

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
  | State.Caret c -> equal ~msg:"caret restored" int block_id c.block_id
  | _ -> fail "expected caret selection after undo"

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
  equal ~msg:"selection spans ascending blocks"
    (list int)
    [ b1.MD.id; b2.MD.id; b3.MD.id ]
    ids

let tests =
  [
    test "focus_inline command" test_focus_inline_command;
    test "request_code_execution" test_request_code_execution_command;
    test "undo/redo" test_undo_redo;
    test "request_copy_selection" test_request_copy_selection;
    test "history capacity" test_history_capacity;
    test "selection restored after undo" test_selection_restored_after_undo;
    test "selection blocks reverse order" test_selection_blocks_reverse;
  ]

let () = run "quill.editor.reducer" [ group "reducer" tests ]
