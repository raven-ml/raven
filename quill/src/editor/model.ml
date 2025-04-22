open Cmarkit

type inline_content =
  | Run of string
  | Emph of inline
  | Strong of inline
  | Code_span of string
  | Seq of inline list

and inline = { id : int; inline_content : inline_content; focused : bool }

type codeblock_content = { code : string; output : string option }

type block_content =
  | Paragraph of inline
  | Codeblock of codeblock_content
  | Heading of int * inline
  | Blank_line of unit
  | Blocks of block list

and block = { id : int; content : block_content; focused : bool }

type model = { document : block list }

let next_block_id_ref = ref 0
let next_run_id_ref = ref 0

let next_block_id () =
  let id = !next_block_id_ref in
  incr next_block_id_ref;
  id

let next_run_id () =
  let id = !next_run_id_ref in
  incr next_run_id_ref;
  id

let init : model = { document = [] }

let rec inline_of_cmarkit inline =
  let mk inline_content : inline =
    let id = next_block_id () in
    { id; inline_content; focused = false }
  in
  match inline with
  | Inline.Text (s, _) -> mk (Run s)
  | Inline.Emphasis (inner, _) ->
      mk (Emph (inline_of_cmarkit (Inline.Emphasis.inline inner)))
  | Inline.Strong_emphasis (inner, _) ->
      mk (Strong (inline_of_cmarkit (Inline.Emphasis.inline inner)))
  | Inline.Code_span (s, _) ->
      let s = Inline.Code_span.code s in
      mk (Code_span s)
  | Inline.Inlines (items, _) ->
      let inlines = List.map inline_of_cmarkit items in
      mk (Seq inlines)
  | _ -> mk (Seq [])

let rec block_content_of_cmarkit cb =
  match cb with
  | Block.Paragraph (p, _) ->
      let norm = Inline.normalize (Block.Paragraph.inline p) in
      Paragraph (inline_of_cmarkit norm)
  | Block.Code_block (codeblock, _) ->
      let codelines = Block.Code_block.code codeblock in
      let code =
        List.map Block_line.to_string codelines |> String.concat "\n"
      in
      Codeblock { code; output = None }
  | Block.Heading (h, _) ->
      let level = Block.Heading.level h in
      let inline = Inline.normalize (Block.Heading.inline h) in
      Heading (level, inline_of_cmarkit inline)
  | Block.Blocks (items, _) -> Blocks (List.map block_of_cmarkit items)
  | Block.Blank_line _ -> Blank_line ()
  | _ ->
      Paragraph
        { id = next_run_id (); inline_content = Run ""; focused = false }

and block_of_cmarkit cb : block =
  let id = next_block_id () in
  { id; content = block_content_of_cmarkit cb; focused = false }

and document_of_cmarkit root =
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

let block_of_md text =
  let doc = Doc.of_string ~strict:true text in
  let block = Doc.block doc in
  let normalized_block = Block.normalize block in
  block_of_cmarkit normalized_block

let inline_content_of_md txt =
  let doc = Doc.of_string ~strict:true txt in
  let block = Doc.block doc in
  match Block.normalize block with
  | Block.Paragraph (p, _) ->
      let inline = Inline.normalize (Block.Paragraph.inline p) in
      let i = inline_of_cmarkit inline in
      i.inline_content
  | _ -> Run ""

let inline_of_md txt : inline =
  let inline_content = inline_content_of_md txt in
  let id = next_run_id () in
  { id; inline_content; focused = false }

let rec cmarkit_of_inline (i : inline) : Inline.t =
  match i.inline_content with
  | Run s -> Inline.Text (s, Meta.none)
  | Emph ic ->
      Inline.Emphasis (Inline.Emphasis.make (cmarkit_of_inline ic), Meta.none)
  | Strong ic ->
      Inline.Strong_emphasis
        (Inline.Emphasis.make (cmarkit_of_inline ic), Meta.none)
  | Code_span s -> Inline.Code_span (Inline.Code_span.of_string s, Meta.none)
  | Seq items -> Inline.Inlines (List.map cmarkit_of_inline items, Meta.none)

and cmarkit_of_inline_content (ic : inline_content) : Inline.t =
  match ic with
  | Run s -> Inline.Text (s, Meta.none)
  | Emph ic ->
      Inline.Emphasis (Inline.Emphasis.make (cmarkit_of_inline ic), Meta.none)
  | Strong ic ->
      Inline.Strong_emphasis
        (Inline.Emphasis.make (cmarkit_of_inline ic), Meta.none)
  | Code_span s -> Inline.Code_span (Inline.Code_span.of_string s, Meta.none)
  | Seq items -> Inline.Inlines (List.map cmarkit_of_inline items, Meta.none)

let rec cmarkit_of_block_content (bc : block_content) : Block.t =
  match bc with
  | Paragraph inline ->
      Block.Paragraph
        (Block.Paragraph.make (cmarkit_of_inline inline), Meta.none)
  | Codeblock code ->
      let lines = Block_line.list_of_string code.code in
      Block.Code_block (Block.Code_block.make lines, Meta.none)
  | Heading (level, inline) ->
      Block.Heading
        (Block.Heading.make ~level (cmarkit_of_inline inline), Meta.none)
  | Blank_line () -> Block.Blank_line ("", Meta.none)
  | Blocks bs -> Block.Blocks (List.map cmarkit_of_block bs, Meta.none)

and cmarkit_of_block (b : block) : Block.t = cmarkit_of_block_content b.content

let cmarkit_of_document (doc : block list) : Block.t =
  Block.Blocks (List.map cmarkit_of_block doc, Meta.none)

let md_of_model (model : model) : string =
  let block = cmarkit_of_document model.document in
  let doc = Doc.make block in
  Cmarkit_commonmark.of_doc doc

let rec split_inline (inline : inline) (target_id : int) (offset : int) :
    inline option * inline option =
  if inline.id = target_id then
    match inline.inline_content with
    | Run s ->
        let before = String.sub s 0 offset in
        let after = String.sub s offset (String.length s - offset) in
        let before_inline =
          if before = "" then None
          else
            Some
              {
                id = next_run_id ();
                inline_content = Run before;
                focused = false;
              }
        in
        let after_inline =
          if after = "" then None
          else
            Some
              {
                id = next_run_id ();
                inline_content = Run after;
                focused = false;
              }
        in
        (before_inline, after_inline)
    | Code_span s ->
        let before = String.sub s 0 offset in
        let after = String.sub s offset (String.length s - offset) in
        let before_inline =
          if before = "" then None
          else
            Some
              {
                id = next_run_id ();
                inline_content = Code_span before;
                focused = false;
              }
        in
        let after_inline =
          if after = "" then None
          else
            Some
              {
                id = next_run_id ();
                inline_content = Code_span after;
                focused = false;
              }
        in
        (before_inline, after_inline)
    | Emph _ | Strong _ | Seq _ ->
        (* Cannot split non-leaf nodes directly at this ID *)
        (Some inline, None)
  else
    match inline.inline_content with
    | Emph inner ->
        let before, after = split_inline inner target_id offset in
        let before' =
          Option.map
            (fun b ->
              { id = next_run_id (); inline_content = Emph b; focused = false })
            before
        in
        let after' =
          Option.map
            (fun a ->
              { id = next_run_id (); inline_content = Emph a; focused = false })
            after
        in
        (before', after')
    | Strong inner ->
        let before, after = split_inline inner target_id offset in
        let before' =
          Option.map
            (fun b ->
              {
                id = next_run_id ();
                inline_content = Strong b;
                focused = false;
              })
            before
        in
        let after' =
          Option.map
            (fun a ->
              {
                id = next_run_id ();
                inline_content = Strong a;
                focused = false;
              })
            after
        in
        (before', after')
    | Seq items ->
        let rec split_seq acc = function
          | [] -> (List.rev acc, []) (* Target not found *)
          | item :: rest -> (
              let before, after = split_inline item target_id offset in
              match (before, after) with
              | Some b, Some a ->
                  let before_seq = List.rev acc @ [ b ] in
                  let after_seq = a :: rest in
                  (before_seq, after_seq)
              | Some b, None -> split_seq (b :: acc) rest
              | None, Some a ->
                  let before_seq = List.rev acc in
                  let after_seq = a :: rest in
                  (before_seq, after_seq)
              | None, None -> split_seq (item :: acc) rest)
        in
        let before_items, after_items = split_seq [] items in
        let before_seq =
          if before_items = [] then None
          else
            Some
              {
                id = next_run_id ();
                inline_content = Seq before_items;
                focused = false;
              }
        in
        let after_seq =
          if after_items = [] then None
          else
            Some
              {
                id = next_run_id ();
                inline_content = Seq after_items;
                focused = false;
              }
        in
        (before_seq, after_seq)
    | Run _ | Code_span _ -> (Some inline, None)
