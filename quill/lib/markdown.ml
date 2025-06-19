(** Markdown parsing and serialization using Cmarkit *)

open Cmarkit

exception Parse_error of string

(** Parse markdown text into a document *)
let parse markdown_text =
  let doc = Cmarkit.Doc.of_string markdown_text in
  let label_defs = Doc.defs doc in
  let root = Doc.block doc in

  (* Track ID generation *)
  let next_block_id = ref 0 in
  let next_inline_id = ref 0 in

  let get_next_block_id () =
    let id = !next_block_id in
    incr next_block_id;
    id
  in

  let get_next_inline_id () =
    let id = !next_inline_id in
    incr next_inline_id;
    id
  in

  (* Convert Cmarkit inline to our inline type *)
  let rec inline_of_cmarkit inline' : Document.inline =
    let make_inline content : Document.inline =
      { id = get_next_inline_id (); content }
    in

    match inline' with
    | Inline.Text (s, _) -> make_inline (Document.Run s)
    | Inline.Code_span (s, _) ->
        let code = Inline.Code_span.code s in
        make_inline (Document.Code_span code)
    | Inline.Emphasis (inner, _) ->
        let child = inline_of_cmarkit (Inline.Emphasis.inline inner) in
        make_inline (Document.Emph child)
    | Inline.Strong_emphasis (inner, _) ->
        let child = inline_of_cmarkit (Inline.Emphasis.inline inner) in
        make_inline (Document.Strong child)
    | Inline.Break (b, _) ->
        let typ = Inline.Break.type' b in
        let break_type = match typ with `Hard -> `Hard | `Soft -> `Soft in
        make_inline (Document.Break break_type)
    | Inline.Image (l, _) ->
        let alt = inline_of_cmarkit (Inline.Link.text l) in
        let ref = Inline.Link.reference l in
        let src =
          match ref with
          | `Inline (def, _) -> (
              match Link_definition.dest def with
              | Some (dest, _) -> dest
              | None -> "")
          | `Ref (_, _, label) -> (
              match Label.Map.find_opt (Label.key label) label_defs with
              | Some (Link_definition.Def (link_def, _)) -> (
                  match Link_definition.dest link_def with
                  | Some (dest, _) -> dest
                  | None -> "")
              | _ -> "")
        in
        make_inline (Document.Image { alt; src })
    | Inline.Link (l, _) ->
        let text = inline_of_cmarkit (Inline.Link.text l) in
        let ref = Inline.Link.reference l in
        let href =
          match ref with
          | `Inline (def, _) -> (
              match Link_definition.dest def with
              | Some (dest, _) -> dest
              | None -> "")
          | `Ref (_, _, label) -> (
              match Label.Map.find_opt (Label.key label) label_defs with
              | Some (Link_definition.Def (link_def, _)) -> (
                  match Link_definition.dest link_def with
                  | Some (dest, _) -> dest
                  | None -> "")
              | _ -> "")
        in
        make_inline (Document.Link { text; href })
    | Inline.Inlines (items, _) ->
        let inlines = List.map inline_of_cmarkit items in
        make_inline (Document.Seq inlines)
    | Inline.Raw_html (html_lines, _) ->
        (* HTML in inline context uses tight lines *)
        let html =
          String.concat "" (List.map Block_line.tight_to_string html_lines)
        in
        make_inline (Document.Raw_html html)
    | Inline.Autolink (l, _) ->
        let dest, _ = Inline.Autolink.link l in
        let text_inline = make_inline (Document.Run dest) in
        make_inline (Document.Link { text = text_inline; href = dest })
    | _ ->
        (* Unknown/unsupported inline type *)
        make_inline (Document.Run "")
  in

  (* Convert Cmarkit block to our block content type *)
  let rec block_content_of_cmarkit cmarkit_block =
    match cmarkit_block with
    | Block.Paragraph (p, _) ->
        let inline = inline_of_cmarkit (Block.Paragraph.inline p) in
        Document.Paragraph inline
    | Block.Code_block (cb, _) ->
        let info_string = Block.Code_block.info_string cb in
        let language =
          match info_string with
          | None -> None
          | Some (s, _) ->
              let lang = String.trim s in
              if lang = "" then None else Some lang
        in
        let code_lines = Block.Code_block.code cb in
        let code =
          String.concat "\n" (List.map Block_line.to_string code_lines)
        in
        Document.Codeblock { code; language; output = None }
    | Block.Heading (h, _) ->
        let level = Block.Heading.level h in
        let inline = inline_of_cmarkit (Block.Heading.inline h) in
        Document.Heading (level, inline)
    | Block.Blank_line _ -> Document.Blank_line
    | Block.Block_quote (bq, _) ->
        let inner_block = Block.Block_quote.block bq in
        let blocks = match inner_block with
          | Block.Blocks (blocks, _) -> process_blocks blocks
          | single -> process_blocks [ single ]
        in
        Document.Block_quote blocks
    | Block.Thematic_break _ -> Document.Thematic_break
    | Block.List (list', _) ->
        let list_type =
          match Block.List'.type' list' with
          | `Unordered c -> Document.Unordered c
          | `Ordered (start, c) -> Document.Ordered (start, c)
        in
        let list_spacing =
          match Block.List'.tight list' with
          | true -> Document.Tight
          | false -> Document.Loose
        in
        let items =
          List.map
            (fun (item, _) ->
              let inner_block = Block.List_item.block item in
              let blocks = match inner_block with
                | Block.Blocks (blocks, _) -> process_blocks blocks
                | single -> process_blocks [ single ]
              in
              blocks)
            (Block.List'.items list')
        in
        Document.List (list_type, list_spacing, items)
    | Block.Html_block (hb, _) ->
        let html = String.concat "\n" (List.map Block_line.to_string hb) in
        Document.Html_block html
    | _ ->
        (* Unknown/unsupported block type *)
        Document.Blank_line
  (* Check if a block is an output start marker *)
  and is_output_start block =
    match block with
    | Block.Html_block (hb, _) ->
        let html = String.concat "\n" (List.map Block_line.to_string hb) in
        String.trim html = "<!-- quill=output_start -->"
    | _ -> false
  (* Check if a block is an output end marker *)
  and is_output_end block =
    match block with
    | Block.Html_block (hb, _) ->
        let html = String.concat "\n" (List.map Block_line.to_string hb) in
        String.trim html = "<!-- quill=output_end -->"
    | _ -> false
  (* Collect output blocks between markers *)
  and collect_output_blocks blocks =
    let rec loop acc = function
      | [] -> (List.rev acc, [])
      | block :: rest when is_output_end block -> (List.rev acc, rest)
      | block :: rest -> loop (block :: acc) rest
    in
    loop [] blocks
  (* Process blocks with output handling *)
  and process_blocks blocks =
    let rec process acc = function
      | [] -> List.rev acc
      | Block.Code_block (cb, _) :: rest ->
          let info_string = Block.Code_block.info_string cb in
          let language =
            match info_string with
            | None -> None
            | Some (s, _) ->
                let lang = String.trim s in
                if lang = "" then None else Some lang
          in
          let code_lines = Block.Code_block.code cb in
          let code =
            String.concat "\n" (List.map Block_line.to_string code_lines)
          in

          (* Check if next non-blank block is output start marker *)
          let rec skip_blanks = function
            | Block.Blank_line _ :: rest -> skip_blanks rest
            | blocks -> blocks
          in
          let output, remaining =
            match skip_blanks rest with
            | block :: rest' when is_output_start block ->
                let output_blocks, rest'' = collect_output_blocks rest' in
                let processed_output =
                  List.map
                    (fun b ->
                      let content = block_content_of_cmarkit b in
                      let id = get_next_block_id () in
                      { Document.id; content })
                    output_blocks
                in
                (Some processed_output, rest'')
            | _ -> (None, rest)
          in

          let content = Document.Codeblock { code; language; output } in
          let id = get_next_block_id () in
          let block = { Document.id; content } in
          process (block :: acc) remaining
      | block :: rest ->
          let content = block_content_of_cmarkit block in
          let id = get_next_block_id () in
          let processed_block = { Document.id; content } in
          process (processed_block :: acc) rest
    in
    process [] blocks
  in

  (* Process all blocks *)
  let processed_blocks =
    match Block.normalize root with
    | Block.Blocks (blocks, _) -> process_blocks blocks
    | other -> process_blocks [ other ]
  in

  (* Return the document with all blocks *)
  { Document.blocks = processed_blocks }

(** Convert our inline back to Cmarkit inline *)
let rec cmarkit_of_inline (inline : Document.inline) =
  match inline.content with
  | Document.Run text -> Inline.Text (text, Meta.none)
  | Document.Code_span code ->
      Inline.Code_span (Inline.Code_span.of_string code, Meta.none)
  | Document.Emph child ->
      let child_inline = cmarkit_of_inline child in
      Inline.Emphasis (Inline.Emphasis.make child_inline, Meta.none)
  | Document.Strong child ->
      let child_inline = cmarkit_of_inline child in
      Inline.Strong_emphasis (Inline.Emphasis.make child_inline, Meta.none)
  | Document.Seq inlines ->
      let cmarkit_inlines = List.map cmarkit_of_inline inlines in
      Inline.Inlines (cmarkit_inlines, Meta.none)
  | Document.Break break_type ->
      let bt = match break_type with `Hard -> `Hard | `Soft -> `Soft in
      Inline.Break (Inline.Break.make bt, Meta.none)
  | Document.Image { alt; src } ->
      let alt_inline = cmarkit_of_inline alt in
      let link_def = Link_definition.make ~dest:(src, Meta.none) () in
      let link = Inline.Link.make alt_inline (`Inline (link_def, Meta.none)) in
      Inline.Image (link, Meta.none)
  | Document.Link { text; href } ->
      let text_inline = cmarkit_of_inline text in
      let link_def = Link_definition.make ~dest:(href, Meta.none) () in
      let link = Inline.Link.make text_inline (`Inline (link_def, Meta.none)) in
      Inline.Link (link, Meta.none)
  | Document.Raw_html html ->
      let lines = Block_line.tight_list_of_string html in
      Inline.Raw_html (lines, Meta.none)

(** Convert our block content back to Cmarkit block *)
let rec cmarkit_of_block_content content =
  match content with
  | Document.Paragraph inline ->
      let cmarkit_inline = cmarkit_of_inline inline in
      Block.Paragraph (Block.Paragraph.make cmarkit_inline, Meta.none)
  | Document.Codeblock { code; language; output } -> (
      let info_string = Option.map (fun s -> (s, Meta.none)) language in
      let code_lines =
        List.map (fun s -> (s, Meta.none)) (String.split_on_char '\n' code)
      in
      let cb = Block.Code_block.make ?info_string code_lines in
      let code_block = Block.Code_block (cb, Meta.none) in

      (* Handle output blocks if present *)
      match output with
      | None -> code_block
      | Some output_blocks ->
          let start_marker =
            Block.Html_block
              ([ ("<!-- quill=output_start -->", Meta.none) ], Meta.none)
          in
          let end_marker =
            Block.Html_block
              ([ ("<!-- quill=output_end -->", Meta.none) ], Meta.none)
          in
          let output_cmarkit =
            List.map
              (fun b -> cmarkit_of_block_content b.Document.content)
              output_blocks
          in
          Block.Blocks
            ( [ code_block; start_marker ] @ output_cmarkit @ [ end_marker ],
              Meta.none ))
  | Document.Heading (level, inline) ->
      let cmarkit_inline = cmarkit_of_inline inline in
      let heading = Block.Heading.make ~level cmarkit_inline in
      Block.Heading (heading, Meta.none)
  | Document.Blank_line -> Block.Blank_line ("", Meta.none)
  | Document.Block_quote blocks ->
      let cmarkit_blocks =
        List.map (fun b -> cmarkit_of_block_content b.Document.content) blocks
      in
      (* For block quotes, we need to ensure proper nesting *)
      let block_content = match cmarkit_blocks with
      | [] -> Block.Blank_line ("", Meta.none)
      | [single] -> single
      | multiple -> Block.Blocks (multiple, Meta.none)
      in
      Block.Block_quote (Block.Block_quote.make block_content, Meta.none)
  | Document.Thematic_break ->
      Block.Thematic_break (Block.Thematic_break.make (), Meta.none)
  | Document.List (list_type, list_spacing, items) ->
      let tight =
        match list_spacing with
        | Document.Tight -> true
        | Document.Loose -> false
      in
      let type' =
        match list_type with
        | Document.Unordered c -> `Unordered c
        | Document.Ordered (start, c) -> `Ordered (start, c)
      in
      let cmarkit_items =
        List.map
          (fun blocks ->
            let cmarkit_blocks =
              List.map
                (fun b -> cmarkit_of_block_content b.Document.content)
                blocks
            in
            let combined = Block.Blocks (cmarkit_blocks, Meta.none) in
            (Block.List_item.make combined, Meta.none))
          items
      in
      let list = Block.List'.make type' ~tight cmarkit_items in
      Block.List (list, Meta.none)
  | Document.Html_block html ->
      (* For HTML blocks, preserve the content as-is without escaping *)
      let lines =
        List.map (fun s -> (s, Meta.none)) (String.split_on_char '\n' html)
      in
      Block.Html_block (lines, Meta.none)

(** Serialize document to markdown *)
let serialize doc =
  let cmarkit_blocks =
    List.map
      (fun b -> cmarkit_of_block_content b.Document.content)
      doc.Document.blocks
  in
  let combined = Block.Blocks (cmarkit_blocks, Meta.none) in
  let cmarkit_doc = Doc.make combined in
  Cmarkit_commonmark.of_doc cmarkit_doc
