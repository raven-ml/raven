(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let caret_for_inline document inline_id =
  match Document.find_block_of_inline document ~inline_id with
  | Some (block, _) ->
      let block_id = block.id in
      let open State in
      Some { block_id; inline_id = Some inline_id; offset = 0 }
  | None -> None

let caret_for_block block_id =
  let open State in
  { block_id; inline_id = None; offset = 0 }

let markdown_for_selection state =
  let open State in
  match state.selection with
  | No_selection -> ""
  | Caret _ -> ""
  | Range { anchor; focus } ->
      let slice =
        Document.slice_between state.document ~start_id:anchor.block_id
          ~end_id:focus.block_id
      in
      Document.to_markdown slice

let apply_command (state : State.t) (command : Command.t) =
  match command with
  | Command.Focus_inline inline_id ->
      let document =
        Document.focus_inline_by_id State.(state.document) inline_id
      in
      let state = State.set_document state document in
      let selection =
        match caret_for_inline document inline_id with
        | Some caret -> State.Caret caret
        | None -> State.(state.selection)
      in
      let state = State.set_selection state selection in
      (state, [])
  | Command.Focus_block block_id ->
      let document =
        Document.focus_block_by_id State.(state.document) block_id
      in
      let state = State.set_document state document in
      let selection = State.Caret (caret_for_block block_id) in
      let state = State.set_selection state selection in
      (state, [])
  | Command.Set_document document ->
      let state = State.record_document_change state document in
      (state, [])
  | Command.Replace_block_with_codeblock block_id ->
      let document =
        Document.replace_block_with_codeblock ~block_id State.(state.document)
      in
      let state = State.record_document_change state document in
      (state, [])
  | Command.Update_codeblock { block_id; code } ->
      let document =
        Document.update_codeblock State.(state.document) ~block_id ~code
      in
      let state = State.record_document_change state document in
      (state, [])
  | Command.Split_block { block_id; inline_id; offset } ->
      let document =
        Document.split_block_at_inline
          State.(state.document)
          ~block_id ~inline_id ~offset
      in
      let state = State.record_document_change state document in
      (state, [])
  | Command.Request_code_execution { block_id; code } ->
      let state = State.mark_block_running state block_id in
      (state, [ Effect.Execute_code { block_id; code } ])
  | Command.Load_document { path } ->
      let state = State.set_load_state state (State.Loading { path }) in
      (state, [ Effect.Load_document { path } ])
  | Command.Set_selection selection ->
      let state = State.set_selection state selection in
      (state, [])
  | Command.Clear_selection ->
      let state = State.clear_selection state in
      (state, [])
  | Command.Normalize_document ->
      let normalized = Document.normalize_blanklines State.(state.document) in
      let state = State.record_document_change state normalized in
      (state, [])
  | Command.Undo -> (
      match State.undo state with
      | Some state' -> (state', [])
      | None -> (state, []))
  | Command.Redo -> (
      match State.redo state with
      | Some state' -> (state', [])
      | None -> (state, []))
  | Command.Save_document { path } ->
      let content = Document.to_markdown State.(state.document) in
      (state, [ Effect.Save_document { path; content } ])
  | Command.Request_copy_selection ->
      let text = markdown_for_selection state in
      if String.equal text "" then (state, [])
      else (state, [ Effect.Copy_to_clipboard { text } ])
  | Command.Request_cut_selection ->
      let text = markdown_for_selection state in
      if String.equal text "" then (state, [])
      else (state, [ Effect.Cut_to_clipboard { text } ])
  | Command.Request_paste_clipboard ->
      (state, [ Effect.Request_clipboard_paste ])

let apply_event (state : State.t) (event : Event.t) =
  match event with
  | Event.Document_loaded { path = _; document } ->
      let config = State.(state.config) in
      let state = State.create ~config ~document () in
      (state, [])
  | Event.Document_load_failed { path; error } ->
      let load_state = State.Load_failed { path; error } in
      let state = State.set_load_state state load_state in
      (state, [])
  | Event.Document_saved { path } ->
      let message = Printf.sprintf "Document saved to %s" path in
      (state, [ Effect.Notify { level = `Info; message } ])
  | Event.Document_save_failed { path; error } ->
      let message = Printf.sprintf "Failed to save %s: %s" path error in
      (state, [ Effect.Notify { level = `Error; message } ])
  | Event.Code_execution_completed { block_id; result } ->
      let output_text =
        match (result.error, result.status) with
        | Some err, `Error -> "Error: " ^ err
        | None, `Error -> "Unknown error"
        | _ -> result.output
      in
      let output_block = Document.block_of_md output_text in
      let document =
        Document.set_codeblock_output
          State.(state.document)
          ~block_id output_block
      in
      let state = State.record_document_change state document in
      let state = State.mark_block_idle state block_id in
      (state, [])
  | Event.Clipboard_content_received { text = _ } -> (state, [])
  | Event.Clipboard_operation_failed { error } ->
      (state, [ Effect.Notify { level = `Error; message = error } ])
