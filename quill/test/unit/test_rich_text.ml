open Alcotest

let test_toggle_bold () =
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

  (* Set selection on the paragraph *)
  let block_id =
    match Quill.Document.get_blocks (Quill.Engine.get_document state) with
    | [ b ] -> b.Quill.Document.id
    | _ -> fail "Expected one block"
  in
  let pos = { Quill.View.block_id; offset = 0 } in
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Set_selection (Quill.View.collapsed_at pos))
  in

  (* Toggle bold *)
  let state, _ =
    Quill.Engine.execute state (Quill.Command.Toggle_inline_style `Bold)
  in

  (* Check that the text is now bold *)
  match
    Quill.Document.find_block (Quill.Engine.get_document state) block_id
  with
  | None -> fail "Block not found"
  | Some block -> (
      match block.content with
      | Quill.Document.Paragraph inline ->
          check bool "is bold" true (Quill.Document.has_style `Bold inline)
      | _ -> fail "Expected paragraph")

let test_toggle_italic () =
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

  (* Set selection on the paragraph *)
  let block_id =
    match Quill.Document.get_blocks (Quill.Engine.get_document state) with
    | [ b ] -> b.Quill.Document.id
    | _ -> fail "Expected one block"
  in
  let pos = { Quill.View.block_id; offset = 0 } in
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Set_selection (Quill.View.collapsed_at pos))
  in

  (* Toggle italic *)
  let state, _ =
    Quill.Engine.execute state (Quill.Command.Toggle_inline_style `Italic)
  in

  (* Check that the text is now italic *)
  match
    Quill.Document.find_block (Quill.Engine.get_document state) block_id
  with
  | None -> fail "Block not found"
  | Some block -> (
      match block.content with
      | Quill.Document.Paragraph inline ->
          check bool "is italic" true (Quill.Document.has_style `Italic inline)
      | _ -> fail "Expected paragraph")

let test_toggle_code () =
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

  (* Set selection on the paragraph *)
  let block_id =
    match Quill.Document.get_blocks (Quill.Engine.get_document state) with
    | [ b ] -> b.Quill.Document.id
    | _ -> fail "Expected one block"
  in
  let pos = { Quill.View.block_id; offset = 0 } in
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Set_selection (Quill.View.collapsed_at pos))
  in

  (* Toggle code *)
  let state, _ =
    Quill.Engine.execute state (Quill.Command.Toggle_inline_style `Code)
  in

  (* Check that the text is now code *)
  match
    Quill.Document.find_block (Quill.Engine.get_document state) block_id
  with
  | None -> fail "Block not found"
  | Some block -> (
      match block.content with
      | Quill.Document.Paragraph inline ->
          check bool "is code" true (Quill.Document.has_style `Code inline)
      | _ -> fail "Expected paragraph")

let test_toggle_removes_style () =
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

  (* Set selection on the paragraph *)
  let block_id =
    match Quill.Document.get_blocks (Quill.Engine.get_document state) with
    | [ b ] -> b.Quill.Document.id
    | _ -> fail "Expected one block"
  in
  let pos = { Quill.View.block_id; offset = 0 } in
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Set_selection (Quill.View.collapsed_at pos))
  in

  (* Toggle bold on *)
  let state, _ =
    Quill.Engine.execute state (Quill.Command.Toggle_inline_style `Bold)
  in

  (* Toggle bold off *)
  let state, _ =
    Quill.Engine.execute state (Quill.Command.Toggle_inline_style `Bold)
  in

  (* Check that the text is no longer bold *)
  match
    Quill.Document.find_block (Quill.Engine.get_document state) block_id
  with
  | None -> fail "Block not found"
  | Some block -> (
      match block.content with
      | Quill.Document.Paragraph inline ->
          check bool "is not bold" false (Quill.Document.has_style `Bold inline)
      | _ -> fail "Expected paragraph")

let test_set_link () =
  let state = Quill.Engine.empty in
  let text =
    Quill.Document.run ~id:(Quill.Document.inline_id_of_int 0) "Click here"
  in
  let block =
    Quill.Document.paragraph ~id:(Quill.Document.block_id_of_int 0) text
  in
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Insert_block block.Quill.Document.content)
  in

  (* Set selection on the paragraph *)
  let block_id =
    match Quill.Document.get_blocks (Quill.Engine.get_document state) with
    | [ b ] -> b.Quill.Document.id
    | _ -> fail "Expected one block"
  in
  let pos = { Quill.View.block_id; offset = 0 } in
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Set_selection (Quill.View.collapsed_at pos))
  in

  (* Set link *)
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Set_link (Some "https://example.com"))
  in

  (* Check that the text is now a link *)
  match
    Quill.Document.find_block (Quill.Engine.get_document state) block_id
  with
  | None -> fail "Block not found"
  | Some block -> (
      match block.content with
      | Quill.Document.Paragraph inline -> (
          match inline.content with
          | Quill.Document.Link { href; _ } ->
              check string "link href" "https://example.com" href
          | _ -> fail "Expected link")
      | _ -> fail "Expected paragraph")

let test_remove_link () =
  let state = Quill.Engine.empty in
  let text =
    Quill.Document.run ~id:(Quill.Document.inline_id_of_int 0) "Click here"
  in
  let link =
    Quill.Document.link
      ~id:(Quill.Document.inline_id_of_int 1)
      ~href:"https://example.com" text
  in
  let block =
    Quill.Document.paragraph ~id:(Quill.Document.block_id_of_int 0) link
  in
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Insert_block block.Quill.Document.content)
  in

  (* Set selection on the paragraph *)
  let block_id =
    match Quill.Document.get_blocks (Quill.Engine.get_document state) with
    | [ b ] -> b.Quill.Document.id
    | _ -> fail "Expected one block"
  in
  let pos = { Quill.View.block_id; offset = 0 } in
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Set_selection (Quill.View.collapsed_at pos))
  in

  (* Remove link *)
  let state, _ = Quill.Engine.execute state (Quill.Command.Set_link None) in

  (* Check that the link is removed *)
  match
    Quill.Document.find_block (Quill.Engine.get_document state) block_id
  with
  | None -> fail "Block not found"
  | Some block -> (
      match block.content with
      | Quill.Document.Paragraph inline -> (
          match inline.content with
          | Quill.Document.Link _ -> fail "Link should be removed"
          | _ -> ())
      | _ -> fail "Expected paragraph")

let () =
  run "Rich Text Tests"
    [
      ( "inline styles",
        [
          test_case "Toggle bold" `Quick test_toggle_bold;
          test_case "Toggle italic" `Quick test_toggle_italic;
          test_case "Toggle code" `Quick test_toggle_code;
          test_case "Toggle removes style" `Quick test_toggle_removes_style;
        ] );
      ( "links",
        [
          test_case "Set link" `Quick test_set_link;
          test_case "Remove link" `Quick test_remove_link;
        ] );
    ]
