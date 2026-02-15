(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Quill_editor
open Windtrap
module MD = Quill_editor.Document

let doc_of_markdown md =
  MD.reset_ids ();
  Document.of_markdown md

let first_block doc =
  match doc with
  | blk :: _ -> blk
  | [] -> fail "expected at least one block"

let block_text block =
  match block.MD.content with
  | MD.Paragraph inline -> MD.inline_to_plain inline
  | MD.Heading (_, inline) -> MD.inline_to_plain inline
  | MD.Codeblock { code; _ } -> code
  | _ -> fail "unexpected block kind for text extraction"

let test_focus_inline_by_id () =
  let doc = doc_of_markdown "Hello" in
  let block = first_block doc in
  let inline =
    match block.MD.content with
    | MD.Paragraph inline -> inline
    | _ -> fail "expected paragraph block"
  in
  let inline_id = inline.MD.id in
  let updated = Document.focus_inline_by_id doc inline_id in
  let updated_block = first_block updated in
  let updated_inline =
    match updated_block.MD.content with
    | MD.Paragraph inline -> inline
    | _ -> fail "expected paragraph block"
  in
  equal ~msg:"inline focused" bool true updated_inline.MD.focused;
  equal ~msg:"block focused" bool true updated_block.MD.focused

let test_split_block_at_inline () =
  let doc = doc_of_markdown "Hello" in
  let block = first_block doc in
  let inline =
    match block.MD.content with
    | MD.Paragraph inline -> inline
    | _ -> fail "expected paragraph block"
  in
  let doc' =
    Document.split_block_at_inline doc ~block_id:block.MD.id
      ~inline_id:inline.MD.id ~offset:3
  in
  equal ~msg:"two blocks produced" int 2 (List.length doc');
  let parts = List.map block_text doc' in
  equal ~msg:"split content preserved" (list string) [ "Hel"; "lo" ] parts

let doc_of_paragraphs texts =
  MD.reset_ids ();
  List.map (fun text -> MD.paragraph (MD.run text)) texts

let nth_block doc n =
  try List.nth doc n with Failure _ -> fail "insufficient blocks"

let test_block_ids_between () =
  let doc = doc_of_paragraphs [ "First"; "Second"; "Third" ] in
  let b1 = nth_block doc 0 in
  let b2 = nth_block doc 1 in
  let b3 = nth_block doc 2 in
  let ids =
    Document.block_ids_between doc ~start_id:b1.MD.id ~end_id:b3.MD.id
  in
  equal ~msg:"inclusive ids" (list int) [ b1.MD.id; b2.MD.id; b3.MD.id ] ids

let test_slice_between () =
  let doc = doc_of_paragraphs [ "First"; "Second"; "Third" ] in
  let b1 = nth_block doc 0 in
  let b3 = nth_block doc 2 in
  let slice = Document.slice_between doc ~start_id:b1.MD.id ~end_id:b3.MD.id in
  let expected = doc_of_paragraphs [ "First"; "Second"; "Third" ] in
  equal ~msg:"slice includes bounds"
    (list string)
    (List.map block_text expected)
    (List.map block_text slice)

let test_block_ids_between_reverse () =
  let doc = doc_of_paragraphs [ "One"; "Two"; "Three" ] in
  let b1 = nth_block doc 0 in
  let b3 = nth_block doc 2 in
  let ids =
    Document.block_ids_between doc ~start_id:b3.MD.id ~end_id:b1.MD.id
  in
  equal ~msg:"reverse preserves order"
    (list int)
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
        | _ -> fail "expected paragraph item")
    | _ -> fail "expected list block"
  in
  let doc' =
    Document.split_block_at_inline doc ~block_id:list_block.MD.id
      ~inline_id:inline.MD.id ~offset:2
  in
  equal ~msg:"split into two blocks" int 2 (List.length doc');
  let texts =
    List.map
      (fun block ->
        match block.MD.content with
        | MD.List (_, _, (item_block :: _) :: _) -> block_text item_block
        | _ -> fail "expected list block")
      doc'
  in
  equal ~msg:"list item halves preserved" (list string) [ "It"; "em" ] texts

let test_find_block_of_inline_nested () =
  let doc = doc_of_markdown "> Nested" in
  let block = first_block doc in
  let inline =
    match block.MD.content with
    | MD.Block_quote (inner :: _) -> (
        match inner.MD.content with
        | MD.Paragraph inline -> inline
        | _ -> fail "expected inner paragraph")
    | _ -> fail "expected block quote paragraph"
  in
  match Document.find_block_of_inline doc ~inline_id:inline.MD.id with
  | Some (found_block, index) ->
      equal ~msg:"block id" int block.MD.id found_block.MD.id;
      equal ~msg:"block index" int 0 index
  | None -> fail "expected to locate inline within block quote"

let test_codeblock_output_roundtrip () =
  let source = "```ocaml\n1 + 1\n```" in
  let doc = doc_of_markdown source in
  let block = first_block doc in
  let block_id = block.MD.id in
  (match block.MD.content with
  | MD.Codeblock { code; output; info } ->
      equal ~msg:"code preserved" string "1 + 1" code;
      equal ~msg:"no output yet" bool true (Option.is_none output);
      equal ~msg:"info preserved" (option string) (Some "ocaml") info
  | _ -> fail "expected code block");
  let output_block = MD.paragraph (MD.run "2") in
  let doc' = Document.set_codeblock_output doc ~block_id output_block in
  match first_block doc' with
  | { MD.content = MD.Codeblock { output = Some out; _ }; _ } ->
      equal ~msg:"output paragraph rendered" string "2" (block_text out);
      let regenerated = Document.to_markdown doc' |> String.trim in
      equal ~msg:"markdown captures output markers" string
        "```ocaml\n\
         1 + 1\n\
         ```\n\
         <!-- quill=output_start -->\n\
         2\n\
         <!-- quill=output_end -->"
        regenerated
  | _ -> fail "expected code block with output"

let test_html_block_preserved () =
  let html = "<div class=\"note\">Hi</div>" in
  let doc = doc_of_markdown (html ^ "\n") in
  match first_block doc with
  | { MD.content = MD.Html_block rendered; _ } ->
      equal ~msg:"html captured" string html rendered
  | _ -> fail "expected html block"

let tests =
  [
    test "focus_inline_by_id" test_focus_inline_by_id;
    test "split_block_at_inline" test_split_block_at_inline;
    test "block_ids_between" test_block_ids_between;
    test "slice_between" test_slice_between;
    test "block_ids_between_reverse" test_block_ids_between_reverse;
    test "split_block_at_inline_in_list" test_split_block_at_inline_in_list;
    test "find_block_of_inline_nested" test_find_block_of_inline_nested;
    test "codeblock_output_roundtrip" test_codeblock_output_roundtrip;
    test "html_block_preserved" test_html_block_preserved;
  ]

let () = run "quill.editor.document" [ group "document" tests ]
