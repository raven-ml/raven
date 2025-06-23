open Quill

let make_test_doc () =
  let text1 = Document.run ~id:(Document.inline_id_of_int 0) "hello world" in
  let text2 = Document.run ~id:(Document.inline_id_of_int 1) "second paragraph" in
  let text3 = Document.run ~id:(Document.inline_id_of_int 2) "third block" in
  let para1 = Document.paragraph ~id:(Document.block_id_of_int 0) text1 in
  let para2 = Document.paragraph ~id:(Document.block_id_of_int 1) text2 in
  let para3 = Document.paragraph ~id:(Document.block_id_of_int 2) text3 in
  { Document.blocks = [para1; para2; para3] }

let test_basic_selection () =
  let open Alcotest in
  let _doc = make_test_doc () in
  
  (* Test position creation *)
  let pos1 = Selection.make_position (Document.block_id_of_int 0) 5 in
  check int "position block_id" 0 (pos1.block_id :> int);
  check int "position offset" 5 pos1.offset;
  
  (* Test selection creation *)
  let pos2 = Selection.make_position (Document.block_id_of_int 1) 10 in
  let sel = Selection.make pos1 pos2 in
  check int "anchor block_id" 0 (sel.anchor.block_id :> int);
  check int "anchor offset" 5 sel.anchor.offset;
  check int "focus block_id" 1 (sel.focus.block_id :> int);
  check int "focus offset" 10 sel.focus.offset;
  
  (* Test collapsed selection *)
  let collapsed = Selection.collapsed pos1 in
  check bool "is collapsed" true (Selection.is_collapsed collapsed);
  check bool "regular selection not collapsed" false (Selection.is_collapsed sel)

let test_position_comparison () =
  let open Alcotest in
  let doc = make_test_doc () in
  
  let pos1 = Selection.make_position (Document.block_id_of_int 0) 5 in
  let pos2 = Selection.make_position (Document.block_id_of_int 0) 10 in
  let pos3 = Selection.make_position (Document.block_id_of_int 1) 3 in
  
  (* Same block comparisons *)
  check int "same block, pos1 < pos2" (-1) 
    (Selection.compare_positions doc pos1 pos2);
  check int "same block, pos2 > pos1" 1 
    (Selection.compare_positions doc pos2 pos1);
  check int "same position" 0 
    (Selection.compare_positions doc pos1 pos1);
  
  (* Different block comparisons *)
  check int "pos1 < pos3 (different blocks)" (-1) 
    (Selection.compare_positions doc pos1 pos3);
  check int "pos3 > pos1 (different blocks)" 1 
    (Selection.compare_positions doc pos3 pos1)

let test_normalize_selection () =
  let open Alcotest in
  let doc = make_test_doc () in
  
  let pos1 = Selection.make_position (Document.block_id_of_int 0) 5 in
  let pos2 = Selection.make_position (Document.block_id_of_int 1) 10 in
  
  (* Already normalized *)
  let sel1 = Selection.make pos1 pos2 in
  let norm1 = Selection.normalize doc sel1 in
  check int "normalized anchor block" 0 (norm1.anchor.block_id :> int);
  check int "normalized anchor offset" 5 norm1.anchor.offset;
  check int "normalized focus block" 1 (norm1.focus.block_id :> int);
  check int "normalized focus offset" 10 norm1.focus.offset;
  
  (* Needs normalization *)
  let sel2 = Selection.make pos2 pos1 in
  let norm2 = Selection.normalize doc sel2 in
  check int "swapped anchor block" 0 (norm2.anchor.block_id :> int);
  check int "swapped anchor offset" 5 norm2.anchor.offset;
  check int "swapped focus block" 1 (norm2.focus.block_id :> int);
  check int "swapped focus offset" 10 norm2.focus.offset

let test_selection_contains () =
  let open Alcotest in
  let doc = make_test_doc () in
  
  let pos1 = Selection.make_position (Document.block_id_of_int 0) 5 in
  let pos2 = Selection.make_position (Document.block_id_of_int 1) 10 in
  let sel = Selection.make pos1 pos2 in
  
  (* Test positions inside selection *)
  let inside1 = Selection.make_position (Document.block_id_of_int 0) 8 in
  let inside2 = Selection.make_position (Document.block_id_of_int 1) 5 in
  check bool "contains position in first block" true 
    (Selection.contains_position doc sel inside1);
  check bool "contains position in second block" true 
    (Selection.contains_position doc sel inside2);
  
  (* Test positions outside selection *)
  let before = Selection.make_position (Document.block_id_of_int 0) 2 in
  let after = Selection.make_position (Document.block_id_of_int 2) 5 in
  check bool "doesn't contain position before" false 
    (Selection.contains_position doc sel before);
  check bool "doesn't contain position after" false 
    (Selection.contains_position doc sel after);
  
  (* Test boundary positions *)
  check bool "contains start position" true 
    (Selection.contains_position doc sel pos1);
  check bool "contains end position" true 
    (Selection.contains_position doc sel pos2)

let test_selection_intersection () =
  let open Alcotest in
  let doc = make_test_doc () in
  
  (* Create overlapping selections *)
  let sel1 = Selection.make 
    (Selection.make_position (Document.block_id_of_int 0) 5)
    (Selection.make_position (Document.block_id_of_int 1) 10) in
  let sel2 = Selection.make 
    (Selection.make_position (Document.block_id_of_int 0) 8)
    (Selection.make_position (Document.block_id_of_int 2) 3) in
  
  (* Test intersection *)
  check bool "selections intersect" true 
    (Selection.intersects doc sel1 sel2);
  
  match Selection.intersection doc sel1 sel2 with
  | None -> fail "Expected intersection"
  | Some inter ->
      check int "intersection start block" 0 (inter.anchor.block_id :> int);
      check int "intersection start offset" 8 inter.anchor.offset;
      check int "intersection end block" 1 (inter.focus.block_id :> int);
      check int "intersection end offset" 10 inter.focus.offset;
  
  (* Test non-overlapping selections *)
  let sel3 = Selection.make 
    (Selection.make_position (Document.block_id_of_int 0) 1)
    (Selection.make_position (Document.block_id_of_int 0) 4) in
  check bool "selections don't intersect" false 
    (Selection.intersects doc sel1 sel3);
  match Selection.intersection doc sel1 sel3 with
  | None -> () (* Good, no intersection *)
  | Some _ -> fail "Expected no intersection"

let test_selection_union () =
  let open Alcotest in
  let doc = make_test_doc () in
  
  let sel1 = Selection.make 
    (Selection.make_position (Document.block_id_of_int 0) 5)
    (Selection.make_position (Document.block_id_of_int 1) 10) in
  let sel2 = Selection.make 
    (Selection.make_position (Document.block_id_of_int 0) 8)
    (Selection.make_position (Document.block_id_of_int 2) 3) in
  
  let union = Selection.union doc sel1 sel2 in
  check int "union start block" 0 (union.anchor.block_id :> int);
  check int "union start offset" 5 union.anchor.offset;
  check int "union end block" 2 (union.focus.block_id :> int);
  check int "union end offset" 3 union.focus.offset

let test_expand_to_word () =
  let open Alcotest in
  let doc = make_test_doc () in
  
  (* Position in middle of "world" *)
  let pos = Selection.make_position (Document.block_id_of_int 0) 8 in
  let sel = Selection.collapsed pos in
  
  match Selection.expand_to_word doc sel with
  | None -> fail "Expected word expansion"
  | Some expanded ->
      check int "word start offset" 6 expanded.anchor.offset;
      check int "word end offset" 11 expanded.focus.offset;
  
  (* Position between words *)
  let pos2 = Selection.make_position (Document.block_id_of_int 0) 5 in
  let sel2 = Selection.collapsed pos2 in
  
  match Selection.expand_to_word doc sel2 with
  | None -> fail "Expected word expansion"
  | Some expanded ->
      check int "next word start" 6 expanded.anchor.offset;
      check int "next word end" 11 expanded.focus.offset

let test_expand_to_line () =
  let open Alcotest in
  let code_block = Document.codeblock ~id:(Document.block_id_of_int 10) 
    "line one\nline two\nline three" in
  let doc = { Document.blocks = [code_block] } in
  
  (* Position in middle of second line *)
  let pos = Selection.make_position (Document.block_id_of_int 10) 13 in
  let sel = Selection.collapsed pos in
  
  match Selection.expand_to_line doc sel with
  | None -> fail "Expected line expansion"
  | Some expanded ->
      check int "line start offset" 9 expanded.anchor.offset;
      check int "line end offset" 17 expanded.focus.offset

let test_expand_to_block () =
  let open Alcotest in
  let doc = make_test_doc () in
  
  let pos = Selection.make_position (Document.block_id_of_int 1) 8 in
  let sel = Selection.collapsed pos in
  
  match Selection.expand_to_block doc sel with
  | None -> fail "Expected block expansion"
  | Some expanded ->
      check int "block start offset" 0 expanded.anchor.offset;
      check int "block end offset" 16 expanded.focus.offset  (* "second paragraph" *)

let test_merge_overlapping () =
  let open Alcotest in
  let doc = make_test_doc () in
  
  (* Create overlapping selections *)
  let sel1 = Selection.make 
    (Selection.make_position (Document.block_id_of_int 0) 0)
    (Selection.make_position (Document.block_id_of_int 0) 5) in
  let sel2 = Selection.make 
    (Selection.make_position (Document.block_id_of_int 0) 3)
    (Selection.make_position (Document.block_id_of_int 0) 8) in
  let sel3 = Selection.make 
    (Selection.make_position (Document.block_id_of_int 1) 0)
    (Selection.make_position (Document.block_id_of_int 1) 5) in
  
  let merged = Selection.merge_overlapping doc [sel1; sel2; sel3] in
  check int "merged selection count" 2 (List.length merged);
  
  (* First merged selection should combine sel1 and sel2 *)
  let first = List.nth merged 0 in
  check int "first merged start" 0 first.anchor.offset;
  check int "first merged end" 8 first.focus.offset;
  
  (* Second selection should remain unchanged *)
  let second = List.nth merged 1 in
  check int "second block" 1 (second.anchor.block_id :> int);
  check int "second start" 0 second.anchor.offset;
  check int "second end" 5 second.focus.offset

let test_get_text () =
  let open Alcotest in
  let doc = make_test_doc () in
  
  (* Single block selection *)
  let sel1 = Selection.make 
    (Selection.make_position (Document.block_id_of_int 0) 6)
    (Selection.make_position (Document.block_id_of_int 0) 11) in
  check string "single block text" "world" (Selection.get_text doc sel1);
  
  (* Multi-block selection *)
  let sel2 = Selection.make 
    (Selection.make_position (Document.block_id_of_int 0) 6)
    (Selection.make_position (Document.block_id_of_int 1) 6) in
  check string "multi-block text" "world\nsecond" (Selection.get_text doc sel2);
  
  (* Full document selection *)
  let sel3 = Selection.make 
    (Selection.make_position (Document.block_id_of_int 0) 0)
    (Selection.make_position (Document.block_id_of_int 2) 11) in
  check string "full document text" "hello world\nsecond paragraph\nthird block" 
    (Selection.get_text doc sel3)

let () =
  let open Alcotest in
  run "Selection" [
    "basic", [
      test_case "basic selection operations" `Quick test_basic_selection;
      test_case "position comparison" `Quick test_position_comparison;
      test_case "normalize selection" `Quick test_normalize_selection;
    ];
    "containment", [
      test_case "contains position" `Quick test_selection_contains;
      test_case "intersection" `Quick test_selection_intersection;
      test_case "union" `Quick test_selection_union;
    ];
    "expansion", [
      test_case "expand to word" `Quick test_expand_to_word;
      test_case "expand to line" `Quick test_expand_to_line;
      test_case "expand to block" `Quick test_expand_to_block;
    ];
    "multi-cursor", [
      test_case "merge overlapping" `Quick test_merge_overlapping;
    ];
    "text", [
      test_case "get text" `Quick test_get_text;
    ];
  ]