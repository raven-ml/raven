type state = {
  document : Document.t;
  execution : Execution.t;
  view : View.t;
  history : state list;
  redo_stack : state list;
  next_block_id : int;
  next_inline_id : int;
}

let empty =
  {
    document = Document.empty;
    execution = Execution.empty;
    view = View.empty;
    history = [];
    redo_stack = [];
    next_block_id = 0;
    next_inline_id = 0;
  }

let make document =
  {
    document;
    execution = Execution.empty;
    view = View.empty;
    history = [];
    redo_stack = [];
    next_block_id = 0;
    next_inline_id = 0;
  }

let get_document state = state.document
let get_execution state = state.execution
let get_view state = state.view
let can_undo state = state.history <> []
let can_redo state = state.redo_stack <> []

let next_block_id state =
  let id = state.next_block_id in
  let state = { state with next_block_id = state.next_block_id + 1 } in
  (id, state)

let next_inline_id state =
  let id = state.next_inline_id in
  let state = { state with next_inline_id = state.next_inline_id + 1 } in
  (id, state)

let block_can_execute state block_id =
  Execution.can_execute state.execution state.document block_id

let get_block_result state block_id =
  Execution.get_result state.execution block_id

let get_focused_block state =
  match state.view.focused_block with
  | None -> None
  | Some id -> Document.find_block state.document id

let get_selection state = state.view.selection

let save_history state =
  (* Create a snapshot of the current state for history *)
  let snapshot =
    {
      state with
      history = [];
      (* Don't include history in the snapshot *)
      redo_stack = [];
      (* Don't include redo stack in the snapshot *)
    }
  in
  { state with history = snapshot :: state.history; redo_stack = [] }

let apply_text_edit state block_id f =
  match Document.find_block state.document block_id with
  | None -> (state, [])
  | Some block -> (
      match block.content with
      | Document.Paragraph inline ->
          let text = Document.inline_to_text inline in
          let new_text = f text in
          let inline_id, state = next_inline_id state in
          let new_inline = Document.run ~id:inline_id new_text in
          let new_content = Document.Paragraph new_inline in
          let document =
            Document.update_block_content state.document block_id new_content
          in
          let execution =
            Execution.mark_stale_from state.execution document block_id
          in
          ({ state with document; execution }, [])
      | Document.Codeblock { code; language; output } ->
          let new_code = f code in
          let new_content =
            Document.Codeblock { code = new_code; language; output }
          in
          let document =
            Document.update_block_content state.document block_id new_content
          in
          let execution =
            Execution.mark_stale_from state.execution document block_id
          in
          ({ state with document; execution }, [])
      | _ -> (state, []))

let execute state command =
  match command with
  | Command.Insert_text (block_id, offset, text) ->
      let state = save_history state in
      apply_text_edit state block_id (fun t -> Text.insert_at t offset text)
  | Command.Delete_range (block_id, start_offset, end_offset) ->
      let state = save_history state in
      apply_text_edit state block_id (fun t ->
          Text.delete_range t start_offset end_offset)
  | Command.Split_block (block_id, offset) -> (
      let state = save_history state in
      match Document.find_block state.document block_id with
      | None -> (state, [])
      | Some block -> (
          match block.content with
          | Document.Paragraph inline ->
              let text = Document.inline_to_text inline in
              let before, after = Text.split_at text offset in
              let inline1_id, state = next_inline_id state in
              let inline1 = Document.run ~id:inline1_id before in
              let inline2_id, state = next_inline_id state in
              let inline2 = Document.run ~id:inline2_id after in
              let block1_id, state = next_block_id state in
              let block1 = Document.paragraph ~id:block1_id inline1 in
              let block2_id, state = next_block_id state in
              let block2 = Document.paragraph ~id:block2_id inline2 in
              let document =
                state.document |> fun d ->
                Document.update_block_content d block_id block1.content
                |> fun d -> Document.insert_after d block_id block2
              in
              let execution =
                Execution.mark_stale_from state.execution document block_id
              in
              ({ state with document; execution }, [])
          | Document.Codeblock { code; language; output } ->
              let before, after = Text.split_at code offset in
              let block1_id, state = next_block_id state in
              let block1 =
                Document.codeblock ?language ?output ~id:block1_id before
              in
              let block2_id, state = next_block_id state in
              let block2 = Document.codeblock ?language ~id:block2_id after in
              let document =
                state.document |> fun d ->
                Document.update_block_content d block_id block1.content
                |> fun d -> Document.insert_after d block_id block2
              in
              let execution =
                Execution.mark_stale_from state.execution document block_id
              in
              ({ state with document; execution }, [])
          | _ -> (state, [])))
  | Command.Merge_blocks (block1_id, block2_id) -> (
      let state = save_history state in
      match
        ( Document.find_block state.document block1_id,
          Document.find_block state.document block2_id )
      with
      | Some b1, Some b2 -> (
          match (b1.content, b2.content) with
          | Document.Paragraph i1, Document.Paragraph i2 ->
              let text1 = Document.inline_to_text i1 in
              let text2 = Document.inline_to_text i2 in
              let merged_text = text1 ^ text2 in
              let new_inline_id, state = next_inline_id state in
              let new_inline = Document.run ~id:new_inline_id merged_text in
              let new_content = Document.Paragraph new_inline in
              let document =
                state.document |> fun d ->
                Document.update_block_content d block1_id new_content
                |> fun d -> Document.remove_block d block2_id
              in
              let execution =
                Execution.mark_stale_from state.execution document block1_id
              in
              ({ state with document; execution }, [])
          | Document.Codeblock c1, Document.Codeblock c2
            when c1.language = c2.language ->
              let merged_code = c1.code ^ "\n" ^ c2.code in
              let new_content =
                Document.Codeblock
                  {
                    code = merged_code;
                    language = c1.language;
                    output = c1.output;
                  }
              in
              let document =
                state.document |> fun d ->
                Document.update_block_content d block1_id new_content
                |> fun d -> Document.remove_block d block2_id
              in
              let execution =
                Execution.mark_stale_from state.execution document block1_id
              in
              ({ state with document; execution }, [])
          | _ -> (state, []))
      | _ -> (state, []))
  | Command.Insert_block content ->
      let state = save_history state in
      let block_id, state = next_block_id state in
      let block = Document.make_block ~id:block_id content in
      let document = Document.add_block state.document block in
      ({ state with document }, [])
  | Command.Insert_after (after_id, content) ->
      let state = save_history state in
      let block_id, state = next_block_id state in
      let block = Document.make_block ~id:block_id content in
      let document = Document.insert_after state.document after_id block in
      let execution =
        Execution.mark_stale_from state.execution document block.Document.id
      in
      ({ state with document; execution }, [])
  | Command.Remove_block block_id ->
      let state = save_history state in
      let document = Document.remove_block state.document block_id in
      let execution =
        Execution.mark_stale_from state.execution document block_id
      in
      ({ state with document; execution }, [])
  | Command.Change_block_type (block_id, content) ->
      let state = save_history state in
      let document =
        Document.update_block_content state.document block_id content
      in
      let execution =
        Execution.mark_stale_from state.execution document block_id
      in
      ({ state with document; execution }, [])
  | Command.Indent block_id -> (
      (* Simple implementation: convert paragraph to block quote *)
      let state = save_history state in
      match Document.find_block state.document block_id with
      | None -> (state, [])
      | Some block -> (
          match block.content with
          | Document.Paragraph _ | Document.Heading _ ->
              let new_content = Document.Block_quote [ block ] in
              let document =
                Document.update_block_content state.document block_id
                  new_content
              in
              let execution =
                Execution.mark_stale_from state.execution document block_id
              in
              ({ state with document; execution }, [])
          | _ -> (state, [])))
  | Command.Outdent block_id -> (
      (* Simple implementation: extract content from block quote *)
      let state = save_history state in
      match Document.find_block state.document block_id with
      | None -> (state, [])
      | Some block -> (
          match block.content with
          | Document.Block_quote [ inner_block ] ->
              let document =
                Document.update_block_content state.document block_id
                  inner_block.content
              in
              let execution =
                Execution.mark_stale_from state.execution document block_id
              in
              ({ state with document; execution }, [])
          | _ -> (state, [])))
  | Command.Set_selection selection ->
      let view = View.set_selection state.view selection in
      ({ state with view }, [])
  | Command.Move_cursor direction -> (
      (* Get current cursor position from selection *)
      match state.view.selection with
      | None -> (
          (* No selection - position based on direction *)
          match direction with
          | `Start -> (
              match Document.get_blocks state.document with
              | [] -> (state, [])
              | block :: _ ->
                  let pos = { View.block_id = block.Document.id; offset = 0 } in
                  let selection = View.collapsed_at pos in
                  let view = View.set_selection state.view selection in
                  ({ state with view }, []))
          | `End -> (
              match List.rev (Document.get_blocks state.document) with
              | [] -> (state, [])
              | last_block :: _ ->
                  let text = Document.block_to_text last_block in
                  let text_len = String.length text in
                  let offset =
                    if text_len > 0 && text.[text_len - 1] = '\n' then
                      text_len - 1
                    else text_len
                  in
                  let pos =
                    { View.block_id = last_block.Document.id; offset }
                  in
                  let selection = View.collapsed_at pos in
                  let view = View.set_selection state.view selection in
                  ({ state with view }, []))
          | `Left | `Right | `Up | `Down -> (
              (* For directional movement, start at beginning of document *)
              match Document.get_blocks state.document with
              | [] -> (state, [])
              | block :: _ ->
                  let pos = { View.block_id = block.Document.id; offset = 0 } in
                  let selection = View.collapsed_at pos in
                  let view = View.set_selection state.view selection in
                  ({ state with view }, [])))
      | Some sel -> (
          (* Handle special cases for Start and End *)
          let new_pos_opt =
            match direction with
            | `Start -> (
                (* Move to start of document *)
                match Document.get_blocks state.document with
                | [] -> None
                | block :: _ ->
                    Some { View.block_id = block.Document.id; offset = 0 })
            | `End -> (
                (* Move to end of document *)
                match List.rev (Document.get_blocks state.document) with
                | [] -> None
                | last_block :: _ ->
                    let text = Document.block_to_text last_block in
                    (* Remove trailing newline from block_to_text *)
                    let text_len = String.length text in
                    let offset =
                      if text_len > 0 && text.[text_len - 1] = '\n' then
                        text_len - 1
                      else text_len
                    in
                    Some { View.block_id = last_block.Document.id; offset })
            | (`Left | `Right | `Up | `Down) as dir ->
                (* Use cursor module for directional movement *)
                Cursor.move_cursor state.document sel.focus dir
          in
          match new_pos_opt with
          | None -> (state, []) (* Can't move in that direction *)
          | Some new_pos ->
              (* Create a collapsed selection at the new position *)
              let selection = View.collapsed_at new_pos in
              let view = View.set_selection state.view selection in
              ({ state with view }, [])))
  | Command.Focus_block block_id ->
      let view = View.set_focus_block state.view (Some block_id) in
      ({ state with view }, [])
  | Command.Focus_inline inline_id ->
      let view = View.set_focus_inline state.view (Some inline_id) in
      ({ state with view }, [])
  | Command.Clear_focus ->
      let view = View.clear_focus state.view in
      ({ state with view }, [])
  | Command.Execute_block block_id ->
      if Execution.can_execute state.execution state.document block_id then
        match Document.find_block state.document block_id with
        | Some block -> (
            match block.content with
            | Document.Codeblock { code; language; output = _ } ->
                let callback result =
                  Command.Set_execution_result (block_id, result)
                in
                ( state,
                  [ Effect.Execute_code { block_id; code; language; callback } ]
                )
            | _ -> (state, []))
        | None -> (state, [])
      else (state, [])
  | Command.Execute_all ->
      let codeblocks = Document.get_codeblocks state.document in
      let effects =
        List.filter_map
          (fun (block_id, code, language) ->
            if Execution.can_execute state.execution state.document block_id
            then
              let callback result =
                Command.Set_execution_result (block_id, result)
              in
              Some (Effect.Execute_code { block_id; code; language; callback })
            else None)
          codeblocks
      in
      (state, effects)
  | Command.Set_execution_result (block_id, result) ->
      let execution = Execution.mark_executed state.execution block_id result in
      ({ state with execution }, [])
  | Command.Clear_results ->
      let execution = Execution.clear_results state.execution in
      ({ state with execution }, [])
  | Command.Undo -> (
      match state.history with
      | [] -> (state, [])
      | prev :: rest ->
          let redo_state =
            {
              state with
              history = rest;
              redo_stack = state :: state.redo_stack;
            }
          in
          ( {
              prev with
              history = rest;
              redo_stack = redo_state :: prev.redo_stack;
            },
            [] ))
  | Command.Redo -> (
      match state.redo_stack with
      | [] -> (state, [])
      | next :: rest ->
          let undo_state =
            { state with history = state :: state.history; redo_stack = rest }
          in
          ( { next with history = undo_state :: next.history; redo_stack = rest },
            [] ))
  | Command.Toggle_inline_style style -> (
      (* For now, implement a simple version that toggles style on entire
         paragraph *)
      match state.view.selection with
      | None -> (state, [])
      | Some sel -> (
          match Document.find_block state.document sel.focus.block_id with
          | None -> (state, [])
          | Some block -> (
              match block.content with
              | Document.Paragraph inline ->
                  let state = save_history state in
                  (* Check if the inline already has this style *)
                  let has_style = Document.has_style style inline in
                  let new_inline, new_next_id =
                    if has_style then
                      Document.remove_style ~next_id:state.next_inline_id style
                        inline
                    else
                      Document.apply_style ~next_id:state.next_inline_id style
                        inline
                  in
                  let new_content = Document.Paragraph new_inline in
                  let document =
                    Document.update_block_content state.document
                      sel.focus.block_id new_content
                  in
                  let execution =
                    Execution.mark_stale_from state.execution document
                      sel.focus.block_id
                  in
                  ( {
                      state with
                      document;
                      execution;
                      next_inline_id = new_next_id;
                    },
                    [] )
              | _ -> (state, []))))
  | Command.Set_link href_opt -> (
      (* Simple implementation: apply/remove link to entire paragraph *)
      match state.view.selection with
      | None -> (state, [])
      | Some sel -> (
          match Document.find_block state.document sel.focus.block_id with
          | None -> (state, [])
          | Some block -> (
              match block.content with
              | Document.Paragraph inline ->
                  let state = save_history state in
                  let new_inline_id, state = next_inline_id state in
                  let new_inline =
                    match href_opt with
                    | Some href ->
                        (* Apply link *)
                        Document.link ~id:new_inline_id ~href inline
                    | None -> (
                        (* Remove link - extract text from any existing link *)
                        match inline.content with
                        | Document.Link { text; _ } -> text
                        | _ -> inline)
                  in
                  let new_content = Document.Paragraph new_inline in
                  let document =
                    Document.update_block_content state.document
                      sel.focus.block_id new_content
                  in
                  let execution =
                    Execution.mark_stale_from state.execution document
                      sel.focus.block_id
                  in
                  ({ state with document; execution }, [])
              | _ -> (state, []))))
