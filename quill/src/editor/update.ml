open Model

type msg =
  | Focus_inline_by_id of int
  | Set_document of block list
  | Replace_block_codeblock of int
  | Split_block of int * int * int
  | Set_codeblock_output of int * string

let log fmt =
  Printf.ksprintf
    (fun s ->
      Js_of_ocaml.Console.console##log (Js_of_ocaml.Js.string ("[update] " ^ s)))
    fmt

let rec set_focused_inline_by_id (inline : inline) (target_id : int) =
  let id_match = inline.id = target_id in
  let inline_content =
    match inline.inline_content with
    | Run s -> Run s
    | Code_span s -> Code_span s
    | Emph ic -> Emph (set_focused_inline_by_id ic target_id)
    | Strong ic -> Strong (set_focused_inline_by_id ic target_id)
    | Seq items ->
        Seq (List.map (fun i -> set_focused_inline_by_id i target_id) items)
  in
  let child_focused =
    match inline_content with
    | Emph ic | Strong ic -> ic.focused
    | Seq items -> List.exists (fun (i : inline) -> i.focused) items
    | _ -> false
  in
  { inline with focused = id_match || child_focused; inline_content }

let rec set_focused_block_by_id (block : block) (target_id : int) : block =
  let content =
    match block.content with
    | Paragraph inline -> Paragraph (set_focused_inline_by_id inline target_id)
    | Heading (lvl, inline) ->
        Heading (lvl, set_focused_inline_by_id inline target_id)
    | Blocks bs ->
        Blocks (List.map (fun b -> set_focused_block_by_id b target_id) bs)
    | Codeblock s -> Codeblock s
    | Blank_line () -> Blank_line ()
  in
  let child_focused =
    match content with
    | Paragraph inline | Heading (_, inline) -> inline.focused
    | Blocks bs -> List.exists (fun b -> b.focused) bs
    | _ -> false
  in
  { block with focused = child_focused; content }

let set_focused_document_by_id (doc : block list) (target_id : int) : block list
    =
  List.map (fun b -> set_focused_block_by_id b target_id) doc

let rec set_codeblock_output_in_block (block : block) (target_id : int)
    (output : string) : block =
  match block.content with
  | Codeblock { code; output = _ } when block.id = target_id ->
      { block with content = Codeblock { code; output = Some output } }
  | Blocks bs ->
      {
        block with
        content =
          Blocks
            (List.map
               (fun b -> set_codeblock_output_in_block b target_id output)
               bs);
      }
  | _ -> block

let update (m : model) (message : msg) : model =
  match message with
  | Focus_inline_by_id inline_id ->
      let new_document = set_focused_document_by_id m.document inline_id in
      { document = new_document }
  | Set_document docs -> { document = docs }
  | Set_codeblock_output (block_id, output) ->
      let new_document =
        List.map
          (fun b -> set_codeblock_output_in_block b block_id output)
          m.document
      in
      { document = new_document }
  | Replace_block_codeblock block_id ->
      let new_document =
        List.map
          (fun b ->
            if b.id = block_id then
              {
                id = block_id;
                content = Codeblock { code = ""; output = None };
                focused = false;
              }
            else b)
          m.document
      in
      { document = new_document }
  | Split_block (block_id, run_id, offset) ->
      let new_document =
        List.map
          (fun b ->
            if b.id = block_id then
              match b.content with
              | Paragraph inline ->
                  let before, after = Model.split_inline inline run_id offset in
                  let new_block1 =
                    match before with
                    | Some i ->
                        {
                          id = next_block_id ();
                          content = Paragraph i;
                          focused = false;
                        }
                    | None ->
                        {
                          id = next_block_id ();
                          content =
                            Paragraph
                              {
                                id = next_run_id ();
                                inline_content = Run "";
                                focused = false;
                              };
                          focused = false;
                        }
                  in
                  let new_block2 =
                    match after with
                    | Some i ->
                        {
                          id = next_block_id ();
                          content = Paragraph i;
                          focused = false;
                        }
                    | None ->
                        {
                          id = next_block_id ();
                          content =
                            Paragraph
                              {
                                id = next_run_id ();
                                inline_content = Run "";
                                focused = false;
                              };
                          focused = false;
                        }
                  in
                  [ new_block1; new_block2 ]
              | _ -> [ b ] (* Only split paragraphs *)
            else [ b ])
          m.document
        |> List.flatten
      in
      { document = new_document }
