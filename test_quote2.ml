let () =
  let md = "> Hello\n> World" in
  Printf.printf "Input:\n%s\n\n" md;
  
  let doc = Quill.Markdown.parse md in
  Printf.printf "Parsed document has %d blocks\n" (List.length (Quill.Document.get_blocks doc));
  
  List.iter (fun block ->
    match block.Quill.Document.content with
    | Quill.Document.Block_quote inner ->
        Printf.printf "Block quote with %d inner blocks\n" (List.length inner);
        List.iter (fun inner_block ->
          match inner_block.Quill.Document.content with
          | Quill.Document.Paragraph inline ->
              Printf.printf "  - Paragraph: %s\n" (Quill.Document.inline_to_text inline)
          | _ -> Printf.printf "  - Other block\n"
        ) inner
    | _ -> Printf.printf "Other block type\n"
  ) (Quill.Document.get_blocks doc);
  
  let output = Quill.Markdown.serialize doc in
  Printf.printf "\nOutput:\n%s\n" output