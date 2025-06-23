open Quill

let test_inline_length () =
  let open Alcotest in
  (* Test simple text run *)
  let run = Document.run ~id:(Document.inline_id_of_int 0) "hello" in
  check int "text run length" 5 (Text_length.inline run);
  
  (* Test code span *)
  let code = Document.make_inline ~id:(Document.inline_id_of_int 1) (Document.Code_span "foo") in
  check int "code span length" 5 (Text_length.inline code);  (* `foo` *)
  
  (* Test emphasis *)
  let emph = Document.make_inline ~id:(Document.inline_id_of_int 2) (Document.Emph run) in
  check int "emphasis length" 7 (Text_length.inline emph);  (* *hello* *)
  
  (* Test strong *)
  let strong = Document.make_inline ~id:(Document.inline_id_of_int 3) (Document.Strong run) in
  check int "strong length" 9 (Text_length.inline strong);  (* **hello** *)
  
  (* Test sequence *)
  let seq = Document.make_inline ~id:(Document.inline_id_of_int 4) 
    (Document.Seq [run; code; emph]) in
  check int "sequence length" 17 (Text_length.inline seq);  (* hello + `foo` + *hello* *)
  
  (* Test hard break *)
  let hard_break = Document.make_inline ~id:(Document.inline_id_of_int 5) 
    (Document.Break `Hard) in
  check int "hard break length" 1 (Text_length.inline hard_break);
  
  (* Test soft break *)
  let soft_break = Document.make_inline ~id:(Document.inline_id_of_int 6) 
    (Document.Break `Soft) in
  check int "soft break length" 1 (Text_length.inline soft_break);
  
  (* Test image *)
  let img = Document.make_inline ~id:(Document.inline_id_of_int 7)
    (Document.Image { alt = run; src = "test.png" }) in
  check int "image length" 19 (Text_length.inline img);  (* ![hello](test.png) *)
  
  (* Test link *)
  let link = Document.link ~id:(Document.inline_id_of_int 8) ~href:"http://test.com" run in
  check int "link length" 24 (Text_length.inline link);  (* [hello](http://test.com) *)
  
  (* Test raw HTML *)
  let html = Document.make_inline ~id:(Document.inline_id_of_int 9)
    (Document.Raw_html "<span>test</span>") in
  check int "raw html length" 17 (Text_length.inline html)

let test_block_length () =
  let open Alcotest in
  let text = Document.run ~id:(Document.inline_id_of_int 0) "hello world" in
  
  (* Test paragraph *)
  let para = Document.paragraph ~id:(Document.block_id_of_int 0) text in
  check int "paragraph length" 12 (Text_length.block para);  (* hello world\n *)
  
  (* Test heading *)
  let h1 = Document.heading ~id:(Document.block_id_of_int 1) 1 text in
  check int "h1 length" 14 (Text_length.block h1);  (* # hello world\n *)
  
  let h3 = Document.heading ~id:(Document.block_id_of_int 2) 3 text in
  check int "h3 length" 16 (Text_length.block h3);  (* ### hello world\n *)
  
  (* Test code block *)
  let code = Document.codeblock ~id:(Document.block_id_of_int 3) "print('hi')" in
  check int "code block length" 20 (Text_length.block code);  (* ```\nprint('hi')\n```\n *)
  
  let code_lang = Document.codeblock ~language:"python" ~id:(Document.block_id_of_int 4) "print('hi')" in
  check int "code block with lang length" 26 (Text_length.block code_lang);  (* ```python\nprint('hi')\n```\n *)
  
  (* Test blank line *)
  let blank = Document.make_block ~id:(Document.block_id_of_int 5) Document.Blank_line in
  check int "blank line length" 1 (Text_length.block blank);
  
  (* Test thematic break *)
  let hr = Document.make_block ~id:(Document.block_id_of_int 6) Document.Thematic_break in
  check int "thematic break length" 4 (Text_length.block hr);  (* ---\n *)
  
  (* Test HTML block *)
  let html = Document.make_block ~id:(Document.block_id_of_int 7) 
    (Document.Html_block "<div>test</div>") in
  check int "html block length" 16 (Text_length.block html)  (* <div>test</div>\n *)

let test_block_quote_length () =
  let open Alcotest in
  let text = Document.run ~id:(Document.inline_id_of_int 0) "quoted" in
  let para = Document.paragraph ~id:(Document.block_id_of_int 0) text in
  let quote = Document.make_block ~id:(Document.block_id_of_int 1)
    (Document.Block_quote [para]) in
  
  (* Block quote adds "> " prefix to each line *)
  (* "> quoted\n" = 9 characters *)
  check int "block quote length" 9 (Text_length.block quote)

let test_list_length () =
  let open Alcotest in
  let text1 = Document.run ~id:(Document.inline_id_of_int 0) "first" in
  let text2 = Document.run ~id:(Document.inline_id_of_int 1) "second" in
  let para1 = Document.paragraph ~id:(Document.block_id_of_int 0) text1 in
  let para2 = Document.paragraph ~id:(Document.block_id_of_int 1) text2 in
  
  (* Unordered list *)
  let ul = Document.make_block ~id:(Document.block_id_of_int 2)
    (Document.List (Unordered '-', Tight, [[para1]; [para2]])) in
  (* "- first\n" (8) + "- second\n" (9) = 17 *)
  check int "unordered list length" 17 (Text_length.block ul);
  
  (* Ordered list *)
  let ol = Document.make_block ~id:(Document.block_id_of_int 3)
    (Document.List (Ordered (1, '.'), Tight, [[para1]; [para2]])) in
  (* "1. first\n" (9) + "1. second\n" (10) = 19 *)
  check int "ordered list length" 19 (Text_length.block ol)

let test_document_length () =
  let open Alcotest in
  let text = Document.run ~id:(Document.inline_id_of_int 0) "hello" in
  let para = Document.paragraph ~id:(Document.block_id_of_int 0) text in
  let heading = Document.heading ~id:(Document.block_id_of_int 1) 1 text in
  let doc = { Document.blocks = [heading; para] } in
  
  (* "# hello\n" (8) + "hello\n" (6) = 14 *)
  check int "document length" 14 (Text_length.document doc)

let test_up_to_block () =
  let open Alcotest in
  let text = Document.run ~id:(Document.inline_id_of_int 0) "hello" in
  let para1 = Document.paragraph ~id:(Document.block_id_of_int 0) text in
  let para2 = Document.paragraph ~id:(Document.block_id_of_int 1) text in
  let para3 = Document.paragraph ~id:(Document.block_id_of_int 2) text in
  let doc = { Document.blocks = [para1; para2; para3] } in
  
  check int "up to first block" 0 
    (Text_length.up_to_block doc (Document.block_id_of_int 0));
  check int "up to second block" 6 
    (Text_length.up_to_block doc (Document.block_id_of_int 1));
  check int "up to third block" 12 
    (Text_length.up_to_block doc (Document.block_id_of_int 2));
  check int "up to non-existent block" 18 
    (Text_length.up_to_block doc (Document.block_id_of_int 99))

let test_find_position () =
  let open Alcotest in
  let text1 = Document.run ~id:(Document.inline_id_of_int 0) "hello" in
  let text2 = Document.run ~id:(Document.inline_id_of_int 1) "world" in
  let para1 = Document.paragraph ~id:(Document.block_id_of_int 0) text1 in
  let para2 = Document.paragraph ~id:(Document.block_id_of_int 1) text2 in
  let doc = { Document.blocks = [para1; para2] } in
  
  (* Document: "hello\nworld\n" *)
  
  (* Test position 0 *)
  (match Text_length.find_position doc 0 with
   | Some (bid, off) ->
       check int "position 0 block_id" 0 (bid :> int);
       check int "position 0 offset" 0 off
   | None -> fail "position 0: expected position");
  
  (* Test position 3 *)
  (match Text_length.find_position doc 3 with
   | Some (bid, off) ->
       check int "position 3 block_id" 0 (bid :> int);
       check int "position 3 offset" 3 off
   | None -> fail "position 3: expected position");
  
  (* Test position 6 (after newline) *)
  (match Text_length.find_position doc 6 with
   | Some (bid, off) ->
       check int "position 6 block_id" 1 (bid :> int);
       check int "position 6 offset" 0 off
   | None -> fail "position 6: expected position");
  
  (* Test position 9 *)
  (match Text_length.find_position doc 9 with
   | Some (bid, off) ->
       check int "position 9 block_id" 1 (bid :> int);
       check int "position 9 offset" 3 off
   | None -> fail "position 9: expected position");
  
  (* Test position beyond document *)
  (match Text_length.find_position doc 100 with
   | None -> () (* Good *)
   | Some _ -> fail "position beyond document: expected None")

let () =
  let open Alcotest in
  run "Text_length" [
    "inline", [
      test_case "inline lengths" `Quick test_inline_length;
    ];
    "block", [
      test_case "block lengths" `Quick test_block_length;
      test_case "block quote length" `Quick test_block_quote_length;
      test_case "list length" `Quick test_list_length;
    ];
    "document", [
      test_case "document length" `Quick test_document_length;
      test_case "up to block" `Quick test_up_to_block;
      test_case "find position" `Quick test_find_position;
    ];
  ]