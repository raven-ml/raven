open Alcotest

let test_indent_paragraph () =
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

  (* Indent the paragraph *)
  let state, _ = Quill.Engine.execute state (Quill.Command.Indent block_id) in

  (* Check that the paragraph is now in a block quote *)
  match
    Quill.Document.find_block (Quill.Engine.get_document state) block_id
  with
  | None -> fail "Block not found"
  | Some block -> (
      match block.content with
      | Quill.Document.Block_quote _ -> ()
      | _ -> fail "Expected block quote")

let test_outdent_block_quote () =
  let state = Quill.Engine.empty in
  let text =
    Quill.Document.run ~id:(Quill.Document.inline_id_of_int 0) "Hello, World!"
  in
  let inner_block =
    Quill.Document.paragraph ~id:(Quill.Document.block_id_of_int 1) text
  in
  let block_quote =
    Quill.Document.make_block
      ~id:(Quill.Document.block_id_of_int 0)
      (Quill.Document.Block_quote [ inner_block ])
  in
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Insert_block block_quote.Quill.Document.content)
  in

  (* Get the block ID *)
  let block_id =
    match Quill.Document.get_blocks (Quill.Engine.get_document state) with
    | [ b ] -> b.Quill.Document.id
    | _ -> fail "Expected one block"
  in

  (* Outdent the block quote *)
  let state, _ = Quill.Engine.execute state (Quill.Command.Outdent block_id) in

  (* Check that the block quote is now a paragraph *)
  match
    Quill.Document.find_block (Quill.Engine.get_document state) block_id
  with
  | None -> fail "Block not found"
  | Some block -> (
      match block.content with
      | Quill.Document.Paragraph _ -> ()
      | _ -> fail "Expected paragraph")

let test_change_block_type () =
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

  (* Change to heading *)
  let new_content = Quill.Document.Heading (2, text) in
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Change_block_type (block_id, new_content))
  in

  (* Check that the block is now a heading *)
  match
    Quill.Document.find_block (Quill.Engine.get_document state) block_id
  with
  | None -> fail "Block not found"
  | Some block -> (
      match block.content with
      | Quill.Document.Heading (level, _) -> check int "heading level" 2 level
      | _ -> fail "Expected heading")

let test_change_to_codeblock () =
  let state = Quill.Engine.empty in
  let text =
    Quill.Document.run ~id:(Quill.Document.inline_id_of_int 0) "print('Hello')"
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

  (* Change to code block *)
  let new_content =
    Quill.Document.Codeblock
      { code = "print('Hello')"; language = Some "python"; output = None }
  in
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Change_block_type (block_id, new_content))
  in

  (* Check that the block is now a code block *)
  match
    Quill.Document.find_block (Quill.Engine.get_document state) block_id
  with
  | None -> fail "Block not found"
  | Some block -> (
      match block.content with
      | Quill.Document.Codeblock { code; language; _ } ->
          check string "code" "print('Hello')" code;
          check (option string) "language" (Some "python") language
      | _ -> fail "Expected codeblock")

let () =
  run "Block Command Tests"
    [
      ( "indent/outdent",
        [
          test_case "Indent paragraph" `Quick test_indent_paragraph;
          test_case "Outdent block quote" `Quick test_outdent_block_quote;
        ] );
      ( "change type",
        [
          test_case "Change to heading" `Quick test_change_block_type;
          test_case "Change to codeblock" `Quick test_change_to_codeblock;
        ] );
    ]
