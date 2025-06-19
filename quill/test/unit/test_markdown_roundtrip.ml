open Alcotest

let test_code_with_output () =
  let markdown_text =
    {|```ocaml
print_string "Hello, World!"
```

<!-- quill=output_start -->
Hello, World!
<!-- quill=output_end -->|}
  in

  (* Parse the markdown *)
  let doc = Quill.Markdown.parse markdown_text in

  (* Find the code block *)
  let code_blocks =
    List.filter_map
      (fun block ->
        match block.Quill.Document.content with
        | Quill.Document.Codeblock _ -> Some block
        | _ -> None)
      (Quill.Document.get_blocks doc)
  in

  check int "code block count" 1 (List.length code_blocks);

  (* Check the code block *)
  match code_blocks with
  | [] -> fail "Expected at least one code block"
  | block :: _ -> (
      match block.Quill.Document.content with
      | Quill.Document.Codeblock { code; language; output } -> (
          check string "code content" "print_string \"Hello, World!\"" code;
          check (option string) "language" (Some "ocaml") language;
          check bool "has output" true (Option.is_some output);

          (* Check the output *)
          match output with
          | Some [ output_block ] -> (
              match output_block.Quill.Document.content with
              | Quill.Document.Paragraph inline ->
                  let text = Quill.Document.inline_to_text inline in
                  check string "output text" "Hello, World!" text
              | _ -> fail "Expected paragraph in output")
          | _ -> fail "Expected exactly one output block")
      | _ -> fail "Expected codeblock")

let test_serialize_with_output () =
  let original =
    {|```ocaml
let x = 42
```

<!-- quill=output_start -->
42
<!-- quill=output_end -->|}
  in

  let doc = Quill.Markdown.parse original in
  let serialized = Quill.Markdown.serialize doc in

  (* The serialized version should contain the output markers *)
  let contains s sub =
    try
      let _ = String.index_from s 0 (String.get sub 0) in
      let rec check_at pos =
        if pos + String.length sub > String.length s then false
        else if String.sub s pos (String.length sub) = sub then true
        else
          try
            let next = String.index_from s (pos + 1) (String.get sub 0) in
            check_at next
          with Not_found -> false
      in
      check_at (String.index_from s 0 (String.get sub 0))
    with Not_found -> false
  in
  check bool "contains start marker" true
    (contains serialized "<!-- quill=output_start -->");
  check bool "contains end marker" true
    (contains serialized "<!-- quill=output_end -->");
  check bool "contains output" true (contains serialized "42")

let test_roundtrip_preserves_output () =
  let original_doc = Quill.Document.empty in

  (* Create a code block with output *)
  let code_inline =
    Quill.Document.run ~id:(Quill.Document.inline_id_of_int 0) "42"
  in
  let output_block =
    Quill.Document.paragraph ~id:(Quill.Document.block_id_of_int 1) code_inline
  in
  let code_block =
    Quill.Document.codeblock ~language:"python" ~output:[ output_block ]
      ~id:(Quill.Document.block_id_of_int 2)
      "print 42"
  in
  let doc = Quill.Document.add_block original_doc code_block in

  (* Serialize and parse back *)
  let markdown = Quill.Markdown.serialize doc in
  let parsed_doc = Quill.Markdown.parse markdown in

  (* Check that the output is preserved *)
  match Quill.Document.get_blocks parsed_doc with
  | [ block ] -> (
      match block.Quill.Document.content with
      | Quill.Document.Codeblock { code; language; output } ->
          check string "code" "print 42" code;
          check (option string) "language" (Some "python") language;
          check bool "has output" true (Option.is_some output)
      | _ -> fail "Expected codeblock")
  | _ -> fail "Expected one block"

let () =
  run "Markdown roundtrip"
    [
      ( "parsing",
        [ test_case "Parse code with output" `Quick test_code_with_output ] );
      ( "serialization",
        [
          test_case "Serialize with output" `Quick test_serialize_with_output;
          test_case "Roundtrip preserves output" `Quick
            test_roundtrip_preserves_output;
        ] );
    ]
