open Alcotest

(* Simple test helpers to work with the new API *)
module SimpleDoc = struct
  let next_id = ref 0

  let get_id () =
    let id = !next_id in
    incr next_id;
    id

  let reset_ids () = next_id := 0
  let next_inline_id () = Quill.Document.inline_id_of_int (get_id ())
  let next_block_id () = Quill.Document.block_id_of_int (get_id ())
end

let test_document_empty () =
  let doc = Quill.Document.empty in
  check int "empty document has no blocks" 0 (Quill.Document.block_count doc);
  check (list pass) "empty document blocks list" []
    (Quill.Document.get_blocks doc)

let test_document_add_block () =
  SimpleDoc.reset_ids ();
  let doc = Quill.Document.empty in
  let text =
    Quill.Document.run ~id:(SimpleDoc.next_inline_id ()) "Hello, world!"
  in
  let block = Quill.Document.paragraph ~id:(SimpleDoc.next_block_id ()) text in
  let doc = Quill.Document.add_block doc block in

  check int "document has one block" 1 (Quill.Document.block_count doc);

  match Quill.Document.find_block doc block.id with
  | None -> fail "Block not found"
  | Some found_block -> check bool "block found" true (block.id = found_block.id)

let test_document_insert_after () =
  SimpleDoc.reset_ids ();
  let doc = Quill.Document.empty in
  let text1 =
    Quill.Document.run ~id:(SimpleDoc.next_inline_id ()) "First paragraph"
  in
  let block1 =
    Quill.Document.paragraph ~id:(SimpleDoc.next_block_id ()) text1
  in
  let doc = Quill.Document.add_block doc block1 in

  let text2 =
    Quill.Document.run ~id:(SimpleDoc.next_inline_id ()) "Second paragraph"
  in
  let block2 =
    Quill.Document.paragraph ~id:(SimpleDoc.next_block_id ()) text2
  in
  let doc = Quill.Document.insert_after doc block1.id block2 in

  check int "document has two blocks" 2 (Quill.Document.block_count doc);

  let blocks = Quill.Document.get_blocks doc in
  check bool "first block is block1" true
    (block1.id = (List.nth blocks 0).Quill.Document.id);
  check bool "second block is block2" true
    (block2.id = (List.nth blocks 1).Quill.Document.id)

let test_engine_empty () =
  let state = Quill.Engine.empty in
  let doc = Quill.Engine.get_document state in
  check int "empty engine has empty document" 0 (Quill.Document.block_count doc)

let test_engine_insert_text () =
  let state = Quill.Engine.empty in

  (* First create a block *)
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Insert_block
         (Quill.Document.Paragraph
            (Quill.Document.run ~id:(Quill.Document.inline_id_of_int 0) "")))
  in

  (* Get the block ID *)
  let doc = Quill.Engine.get_document state in
  let block_id =
    match Quill.Document.get_blocks doc with
    | [ block ] -> block.Quill.Document.id
    | _ -> fail "Expected one block"
  in

  (* Insert text into it *)
  let state, _ =
    Quill.Engine.execute state
      (Quill.Command.Insert_text (block_id, 0, "Hello"))
  in

  let doc = Quill.Engine.get_document state in
  match Quill.Document.find_block doc block_id with
  | Some block -> (
      match block.Quill.Document.content with
      | Quill.Document.Paragraph inline ->
          check string "text inserted" "Hello"
            (Quill.Document.inline_to_text inline)
      | _ -> fail "Wrong block type")
  | None -> fail "Block not found"

let () =
  run "Quill Document Tests"
    [
      ( "document",
        [
          test_case "Empty document" `Quick test_document_empty;
          test_case "Add block" `Quick test_document_add_block;
          test_case "Insert after" `Quick test_document_insert_after;
        ] );
      ( "engine",
        [
          test_case "Empty engine" `Quick test_engine_empty;
          test_case "Insert text" `Quick test_engine_insert_text;
        ] );
    ]
