open Cmarkit

type inline_content =
  | Run of string
  | Emph of inline_content
  | Strong of inline_content
  | Seq of inline list

and inline = { id : int; content : inline_content; focused : bool }

type block_content =
  | Paragraph of inline
  | Codeblock of string
  | Heading of int * inline
  | Blocks of block list

and block = { id : int; content : block_content; focused : bool }

type model = { document : block list }

let next_id = ref 0
let init : model = { document = [] }

let rec inline_of_cmarkit inline =
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
  match cb with
  | Block.Paragraph (p, _) ->
      let norm = Inline.normalize (Block.Paragraph.inline p) in
      Paragraph
        (match inline_of_cmarkit norm with
        | Some i -> i
        | None -> { id = !next_id; content = Run ""; focused = false })
  | Block.Code_block (codeblock, _) ->
      let codelines = Block.Code_block.code codeblock in
      let code =
        List.map Block_line.to_string codelines |> String.concat "\n"
      in
      Codeblock code
  | Block.Heading (h, _) ->
      let level = Block.Heading.level h in
      let inline = Inline.normalize (Block.Heading.inline h) in
      Heading
        ( level,
          match inline_of_cmarkit inline with
          | Some i -> i
          | None -> { id = !next_id; content = Run ""; focused = false } )
  | Block.Blocks (items, _) -> Blocks (List.map block_of_cmarkit items)
  | _ -> Paragraph { id = !next_id; content = Run ""; focused = false }

and block_of_cmarkit cb : block =
  let id = !next_id in
  incr next_id;
  { id; content = block_content_of_cmarkit cb; focused = false }

and document_of_cmarkit root =
  next_id := 0;
  match Block.normalize root with
  | Block.Blocks (items, _) -> List.map block_of_cmarkit items
  | other -> [ block_of_cmarkit other ]

let document_of_md text =
  let doc = Doc.of_string ~strict:true text in
  let block = Doc.block doc in
  let normalized_block = Block.normalize block in
  document_of_cmarkit normalized_block

let block_content_of_md text =
  let doc = Doc.of_string ~strict:true text in
  let block = Doc.block doc in
  let normalized_block = Block.normalize block in
  block_content_of_cmarkit normalized_block

let inline_of_md txt =
  let doc = Doc.of_string ~strict:true txt in
  let block = Doc.block doc in
  match Block.normalize block with
  | Block.Paragraph (p, _) -> (
      let inline = Inline.normalize (Block.Paragraph.inline p) in
      match inline_of_cmarkit inline with
      | Some i -> i
      | None -> { id = !next_id; content = Run ""; focused = false })
  | _ -> { id = !next_id; content = Run ""; focused = false }
