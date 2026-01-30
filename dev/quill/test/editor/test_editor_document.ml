(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Quill_editor
module MD = Quill_editor.Document

let doc_of_markdown md =
  MD.reset_ids ();
  Document.of_markdown md

let first_block doc =
  match doc with
  | blk :: _ -> blk
  | [] -> Alcotest.fail "expected at least one block"

let block_text block =
  match block.MD.content with
  | MD.Paragraph inline -> MD.inline_to_plain inline
  | MD.Heading (_, inline) -> MD.inline_to_plain inline
  | MD.Codeblock { code; _ } -> code
  | _ -> Alcotest.fail "unexpected block kind for text extraction"

let test_focus_inline_by_id () =
  let doc = doc_of_markdown "Hello" in
  let block = first_block doc in
  let inline =
    match block.MD.content with
    | MD.Paragraph inline -> inline
    | _ -> Alcotest.fail "expected paragraph block"
  in
  let inline_id = inline.MD.id in
  let updated = Document.focus_inline_by_id doc inline_id in
  let updated_block = first_block updated in
  let updated_inline =
    match updated_block.MD.content with
    | MD.Paragraph inline -> inline
    | _ -> Alcotest.fail "expected paragraph block"
  in
  Alcotest.(check bool) "inline focused" true updated_inline.MD.focused;
  Alcotest.(check bool) "block focused" true updated_block.MD.focused

let test_split_block_at_inline () =
  let doc = doc_of_markdown "Hello" in
  let block = first_block doc in
  let inline =
    match block.MD.content with
    | MD.Paragraph inline -> inline
    | _ -> Alcotest.fail "expected paragraph block"
  in
  let doc' =
    Document.split_block_at_inline doc ~block_id:block.MD.id
      ~inline_id:inline.MD.id ~offset:3
  in
  Alcotest.(check int) "two blocks produced" 2 (List.length doc');
  let parts = List.map block_text doc' in
  Alcotest.(check (list string)) "split content preserved" [ "Hel"; "lo" ] parts

let doc_of_paragraphs texts =
  MD.reset_ids ();
  List.map (fun text -> MD.paragraph (MD.run text)) texts

let nth_block doc n =
  try List.nth doc n with Failure _ -> Alcotest.fail "insufficient blocks"

let test_block_ids_between () =
  let doc = doc_of_paragraphs [ "First"; "Second"; "Third" ] in
  let b1 = nth_block doc 0 in
  let b2 = nth_block doc 1 in
  let b3 = nth_block doc 2 in
  let ids =
    Document.block_ids_between doc ~start_id:b1.MD.id ~end_id:b3.MD.id
  in
  Alcotest.(check (list int))
    "inclusive ids"
    [ b1.MD.id; b2.MD.id; b3.MD.id ]
    ids

let test_slice_between () =
  let doc = doc_of_paragraphs [ "First"; "Second"; "Third" ] in
  let b1 = nth_block doc 0 in
  let b3 = nth_block doc 2 in
  let slice = Document.slice_between doc ~start_id:b1.MD.id ~end_id:b3.MD.id in
  let expected = doc_of_paragraphs [ "First"; "Second"; "Third" ] in
  Alcotest.(check (list string))
    "slice includes bounds"
    (List.map block_text expected)
    (List.map block_text slice)

let test_block_ids_between_reverse () =
  let doc = doc_of_paragraphs [ "One"; "Two"; "Three" ] in
  let b1 = nth_block doc 0 in
  let b3 = nth_block doc 2 in
  let ids =
    Document.block_ids_between doc ~start_id:b3.MD.id ~end_id:b1.MD.id
  in
  Alcotest.(check (list int))
    "reverse preserves order"
    [ b1.MD.id; (nth_block doc 1).MD.id; b3.MD.id ]
    ids

let test_split_block_at_inline_in_list () =
  let doc = doc_of_markdown "- Item" in
  let list_block = first_block doc in
  let inline =
    match list_block.MD.content with
    | MD.List (_, _, (item_block :: _) :: _) -> (
        match item_block.MD.content with
        | MD.Paragraph inline -> inline
        | _ -> Alcotest.fail "expected paragraph item")
    | _ -> Alcotest.fail "expected list block"
  in
  let doc' =
    Document.split_block_at_inline doc ~block_id:list_block.MD.id
      ~inline_id:inline.MD.id ~offset:2
  in
  Alcotest.(check int) "split into two blocks" 2 (List.length doc');
  let texts =
    List.map
      (fun block ->
        match block.MD.content with
        | MD.List (_, _, (item_block :: _) :: _) -> block_text item_block
        | _ -> Alcotest.fail "expected list block")
      doc'
  in
  Alcotest.(check (list string))
    "list item halves preserved" [ "It"; "em" ] texts

let test_find_block_of_inline_nested () =
  let doc = doc_of_markdown "> Nested" in
  let block = first_block doc in
  let inline =
    match block.MD.content with
    | MD.Block_quote (inner :: _) -> (
        match inner.MD.content with
        | MD.Paragraph inline -> inline
        | _ -> Alcotest.fail "expected inner paragraph")
    | _ -> Alcotest.fail "expected block quote paragraph"
  in
  match Document.find_block_of_inline doc ~inline_id:inline.MD.id with
  | Some (found_block, index) ->
      Alcotest.(check int) "block id" block.MD.id found_block.MD.id;
      Alcotest.(check int) "block index" 0 index
  | None -> Alcotest.fail "expected to locate inline within block quote"

let test_codeblock_output_roundtrip () =
  let source = "```ocaml\n1 + 1\n```" in
  let doc = doc_of_markdown source in
  let block = first_block doc in
  let block_id = block.MD.id in
  (match block.MD.content with
  | MD.Codeblock { code; output; info } ->
      Alcotest.(check string) "code preserved" "1 + 1" code;
      Alcotest.(check bool) "no output yet" true (Option.is_none output);
      Alcotest.(check (option string)) "info preserved" (Some "ocaml") info
  | _ -> Alcotest.fail "expected code block");
  let output_block = MD.paragraph (MD.run "2") in
  let doc' = Document.set_codeblock_output doc ~block_id output_block in
  match first_block doc' with
  | { MD.content = MD.Codeblock { output = Some out; _ }; _ } ->
      Alcotest.(check string) "output paragraph rendered" "2" (block_text out);
      let regenerated = Document.to_markdown doc' |> String.trim in
      Alcotest.(check string)
        "markdown captures output markers"
        "```ocaml\n\
         1 + 1\n\
         ```\n\
         <!-- quill=output_start -->\n\
         2\n\
         <!-- quill=output_end -->"
        regenerated
  | _ -> Alcotest.fail "expected code block with output"

let test_html_block_preserved () =
  let html = "<div class=\"note\">Hi</div>" in
  let doc = doc_of_markdown (html ^ "\n") in
  match first_block doc with
  | { MD.content = MD.Html_block rendered; _ } ->
      Alcotest.(check string) "html captured" html rendered
  | _ -> Alcotest.fail "expected html block"

let tests =
  [
    Alcotest.test_case "focus_inline_by_id" `Quick test_focus_inline_by_id;
    Alcotest.test_case "split_block_at_inline" `Quick test_split_block_at_inline;
    Alcotest.test_case "block_ids_between" `Quick test_block_ids_between;
    Alcotest.test_case "slice_between" `Quick test_slice_between;
    Alcotest.test_case "block_ids_between_reverse" `Quick
      test_block_ids_between_reverse;
    Alcotest.test_case "split_block_at_inline_in_list" `Quick
      test_split_block_at_inline_in_list;
    Alcotest.test_case "find_block_of_inline_nested" `Quick
      test_find_block_of_inline_nested;
    Alcotest.test_case "codeblock_output_roundtrip" `Quick
      test_codeblock_output_roundtrip;
    Alcotest.test_case "html_block_preserved" `Quick test_html_block_preserved;
  ]

let () = Alcotest.run "quill.editor.document" [ ("document", tests) ]
