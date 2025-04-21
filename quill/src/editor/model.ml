type inline_content =
  | Run of string
  | Emph of inline_content
  | Strong of inline_content
  | Seq of inline list

and inline = { id : int; content : inline_content; focused : bool }

type block_content =
  | Paragraph of inline
  | Codeblock of string
  | Blocks of block list

and block = { id : int; content : block_content; focused : bool }

type model = { document : block list }

let next_id = ref 0

let init : model =
  let mk_block id content : block = { id; content; focused = false } in
  let mk_inline id content : inline = { id; content; focused = false } in
  {
    document =
      [
        mk_block 0 (Paragraph (mk_inline 0 (Run "Welcome to Quill!")));
        mk_block 1
          (Paragraph
             (mk_inline 1
                (Seq
                   [
                     mk_inline 2 (Run "This is a ");
                     mk_inline 3 (Emph (Run "rich"));
                     mk_inline 4 (Run " text editor.");
                   ])));
      ];
  }

let rec inline_of_cmarkit inline =
  let open Cmarkit in
  let mk content : inline option =
    let id = !next_id in
    incr next_id;
    Some { id; content; focused = false }
  in
  match inline with
  | Inline.Text (s, _) -> mk (Run s)
  | Inline.Emphasis (inner, _) -> (
      match inline_of_cmarkit (Inline.Emphasis.inline inner) with
      | Some i -> mk (Emph i.content)
      | None -> mk (Emph (Run "")))
  | Inline.Strong_emphasis (inner, _) -> (
      match inline_of_cmarkit (Inline.Emphasis.inline inner) with
      | Some i -> mk (Strong i.content)
      | None -> mk (Strong (Run "")))
  | Inline.Inlines (items, _) ->
      let inlines = List.filter_map inline_of_cmarkit items in
      if inlines = [] then None else mk (Seq inlines)
  | _ -> None

let rec block_content_of_cmarkit cb =
  let open Cmarkit in
  match cb with
  | Block.Paragraph (p, _) ->
      let norm = Inline.normalize (Block.Paragraph.inline p) in
      let inline =
        match inline_of_cmarkit norm with
        | Some i -> i
        | None ->
            let id = !next_id in
            incr next_id;
            { id; content = Run ""; focused = false }
      in
      Paragraph inline
  | Block.Code_block (codeblock, _) ->
      let codelines = Block.Code_block.code codeblock in
      let code =
        codelines
        |> List.map (fun l -> Block_line.to_string l)
        |> String.concat "\n"
      in
      Codeblock code
  | Block.Blocks (items, _) ->
      let children = List.map block_of_cmarkit items in
      Blocks children
  | _ -> Paragraph { id = !next_id; content = Run ""; focused = false }

and block_of_cmarkit cb : block =
  let id = !next_id in
  incr next_id;
  { id; content = block_content_of_cmarkit cb; focused = false }

and document_of_cmarkit root =
  next_id := 0;
  match Cmarkit.Block.normalize root with
  | Cmarkit.Block.Blocks (items, _) -> List.map block_of_cmarkit items
  | other -> [ block_of_cmarkit other ]

let document_of_md text =
  let open Cmarkit in
  let doc = Doc.of_string ~strict:true text in
  let block = Doc.block doc in
  let normalized_block = Block.normalize block in
  document_of_cmarkit normalized_block

let block_content_of_md text =
  let open Cmarkit in
  let doc = Doc.of_string ~strict:true text in
  let block = Doc.block doc in
  let normalized_block = Block.normalize block in
  block_content_of_cmarkit normalized_block
