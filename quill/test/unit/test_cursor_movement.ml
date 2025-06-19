open Alcotest

let test_move_cursor_basic () =
  (* Create a document with some text *)
  let state = Quill.Engine.empty in
  let text =
    Quill.Document.run ~id:(Quill.Document.inline_id_of_int 0) "Hello, World!"
  in
  let block =
    Quill.Document.paragraph ~id:(Quill.Document.block_id_of_int 0) text
  in
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Insert_block block.Quill.Document.content)
  in

  (* Get the block ID *)
  let block_id =
    match Quill.Document.get_blocks (Quill.Engine.get_document state) with
    | [ b ] -> b.Quill.Document.id
    | _ -> fail "Expected one block"
  in

  (* Set initial cursor position at start *)
  let initial_pos = { Quill.View.block_id; offset = 0 } in
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Set_selection (Quill.View.collapsed_at initial_pos))
  in

  (* Move right *)
  let state, _ =
    Quill.Engine.execute state (Quill.Command.Move_cursor `Right)
  in
  match (Quill.Engine.get_view state).Quill.View.selection with
  | None -> fail "Expected selection after move"
  | Some sel -> check int "cursor moved right" 1 sel.focus.offset

let test_move_cursor_boundaries () =
  (* Create a document with two blocks *)
  let state = Quill.Engine.empty in
  let text1 =
    Quill.Document.run ~id:(Quill.Document.inline_id_of_int 0) "First"
  in
  let block1 =
    Quill.Document.paragraph ~id:(Quill.Document.block_id_of_int 0) text1
  in
  let text2 =
    Quill.Document.run ~id:(Quill.Document.inline_id_of_int 1) "Second"
  in
  let block2 =
    Quill.Document.paragraph ~id:(Quill.Document.block_id_of_int 1) text2
  in

  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Insert_block block1.Quill.Document.content)
  in
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Insert_block block2.Quill.Document.content)
  in

  (* Move to end of document *)
  let state, _ = Quill.Engine.execute state (Quill.Command.Move_cursor `End) in
  match (Quill.Engine.get_view state).Quill.View.selection with
  | None -> fail "Expected selection after move"
  | Some sel -> (
      let blocks =
        Quill.Document.get_blocks (Quill.Engine.get_document state)
      in
      match List.rev blocks with
      | [] -> fail "No blocks"
      | last_block :: _ ->
          check bool "at last block" true
            (sel.focus.block_id = last_block.Quill.Document.id);
          check int "at end of block" 6 sel.focus.offset)

let test_move_cursor_start_end () =
  let state = Quill.Engine.empty in
  let text =
    Quill.Document.run ~id:(Quill.Document.inline_id_of_int 0) "Test"
  in
  let block =
    Quill.Document.paragraph ~id:(Quill.Document.block_id_of_int 0) text
  in
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Insert_block block.Quill.Document.content)
  in

  (* Move to end *)
  let state, _ = Quill.Engine.execute state (Quill.Command.Move_cursor `End) in
  let end_sel = (Quill.Engine.get_view state).Quill.View.selection in

  (* Move to start *)
  let state, _ =
    Quill.Engine.execute state (Quill.Command.Move_cursor `Start)
  in
  let start_sel = (Quill.Engine.get_view state).Quill.View.selection in

  match (end_sel, start_sel) with
  | Some e, Some s ->
      check int "end position" 4 e.focus.offset;
      check int "start position" 0 s.focus.offset
  | _ -> fail "Expected selections"

let () =
  run "Cursor Movement Tests"
    [
      ( "basic",
        [
          test_case "Move right" `Quick test_move_cursor_basic;
          test_case "Move boundaries" `Quick test_move_cursor_boundaries;
          test_case "Move start/end" `Quick test_move_cursor_start_end;
        ] );
    ]
