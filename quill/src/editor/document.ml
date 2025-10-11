open Cmarkit

type inline_content =
  | Run of string
  | Emph of inline
  | Strong of inline
  | Code_span of string
  | Seq of inline list
  | Break of [ `Hard | `Soft ] (* Hard or soft line break *)
  | Image of { alt : inline; src : string } (* Image with alt text and source *)
  | Link of { text : inline; href : string } (* Link with text and URL *)
  | Raw_html of string (* Raw HTML content *)

and inline = { id : int; inline_content : inline_content; focused : bool }

type codeblock_content = {
  code : string;
  output : block option;
  info : string option;
}

and block_content =
  | Paragraph of inline
  | Codeblock of codeblock_content
  | Heading of int * inline
  | Blank_line of unit
  | Block_quote of block list
  | Thematic_break
  | List of list_type * list_spacing * block list list
  | Html_block of string
  | Link_reference_definition of Link_definition.t node
  | Blocks of block list

and list_type = Ordered of int * char | Unordered of char
and list_spacing = Tight | Loose
and block = { id : int; content : block_content; focused : bool }

type t = block list

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

let inline ?(focused = false) inline_content =
  let id = next_run_id () in
  { id; inline_content; focused }

let run ?focused content = inline ?focused (Run content)
let emph ?focused content = inline ?focused (Emph content)
let strong ?focused content = inline ?focused (Strong content)
let code_span ?focused content = inline ?focused (Code_span content)
let seq ?focused content = inline ?focused (Seq content)

let block ?(focused = false) content =
  let id = next_block_id () in
  { id; content; focused }

let paragraph ?focused inline = block ?focused (Paragraph inline)

let codeblock ?output ?info ?focused code =
  block ?focused (Codeblock { code; output; info })

let heading ?focused level inline = block ?focused (Heading (level, inline))
let blank_line ?focused () = block ?focused (Blank_line ())
let blocks ?focused items = block ?focused (Blocks items)
let block_quote ?focused items = block ?focused (Block_quote items)
let thematic_break ?focused () = block ?focused Thematic_break

let list ?focused list_type spacing items =
  block ?focused (List (list_type, spacing, items))

let html_block ?focused html = block ?focused (Html_block html)

let link_reference_definition ?focused ld =
  block ?focused (Link_reference_definition ld)

let init : t = []
let empty = init

let rec inline_of_cmarkit label_defs inline' =
  match inline' with
  | Inline.Text (s, _) -> run s
  | Inline.Emphasis (inner, _) ->
      emph (inline_of_cmarkit label_defs (Inline.Emphasis.inline inner))
  | Inline.Strong_emphasis (inner, _) ->
      strong (inline_of_cmarkit label_defs (Inline.Emphasis.inline inner))
  | Inline.Code_span (s, _) ->
      let s = Inline.Code_span.code s in
      code_span s
  | Inline.Inlines (items, _) ->
      let inlines = List.map (inline_of_cmarkit label_defs) items in
      seq inlines
  | Inline.Break (b, _) ->
      let typ = Inline.Break.type' b in
      inline (Break typ)
  | Inline.Image (l, _) ->
      let alt = inline_of_cmarkit label_defs (Inline.Link.text l) in
      let ref = Inline.Link.reference l in
      let src =
        match ref with
        | `Inline (def, _) -> (
            match Link_definition.dest def with
            | Some (dest, _) -> dest
            | None -> "" (* Fallback for undefined references *))
        | `Ref (_, _, label) -> (
            match Label.Map.find_opt (Label.key label) label_defs with
            | Some (Link_definition.Def (link_def, _)) -> (
                match Link_definition.dest link_def with
                | Some (dest, _) -> dest
                | None -> "" (* Fallback for undefined references *))
            | _ -> "" (* Fallback for undefined or non-link references *))
      in
      inline (Image { alt; src })
  | Inline.Link (l, _) ->
      let text = inline_of_cmarkit label_defs (Inline.Link.text l) in
      let ref = Inline.Link.reference l in
      let href =
        match ref with
        | `Inline (def, _) -> (
            match Link_definition.dest def with
            | Some (dest, _) -> dest
            | None -> "" (* Fallback for undefined references *))
        | `Ref (_, _, label) -> (
            match Label.Map.find_opt (Label.key label) label_defs with
            | Some (Link_definition.Def (link_def, _)) -> (
                match Link_definition.dest link_def with
                | Some (dest, _) -> dest
                | None -> "" (* Fallback for undefined references *))
            | _ -> "" (* Fallback for undefined or non-link references *))
      in
      inline (Link { text; href })
  | Inline.Raw_html (block_lines, _) ->
      let html =
        List.map Block_line.tight_to_string block_lines |> String.concat ""
      in
      inline (Raw_html html)
  | _ -> seq []

let html_block_to_string html =
  List.map Block_line.to_string html |> String.concat "\n"

let is_output_start html =
  String.trim (html_block_to_string html) = "<!-- quill=output_start -->"

let is_output_end html =
  String.trim (html_block_to_string html) = "<!-- quill=output_end -->"

let rec block_content_of_cmarkit label_defs cb =
  match cb with
  | Block.Paragraph (p, _) ->
      let norm = Inline.normalize (Block.Paragraph.inline p) in
      Paragraph (inline_of_cmarkit label_defs norm)
  | Block.Code_block (codeblock, _) ->
      let codelines = Block.Code_block.code codeblock in
      let code =
        List.map Block_line.to_string codelines |> String.concat "\n"
      in
      let info =
        match Block.Code_block.info_string codeblock with
        | Some (info_str, _) -> Some (String.trim info_str)
        | None -> None
      in
      Codeblock { code; output = None; info }
  | Block.Heading (h, _) ->
      let level = Block.Heading.level h in
      let inline = Inline.normalize (Block.Heading.inline h) in
      Heading (level, inline_of_cmarkit label_defs inline)
  | Block.Block_quote (bq, _) ->
      let blocks = Block.Block_quote.block bq in
      let parsed_blocks = process_cmarkit_block label_defs blocks in
      Block_quote parsed_blocks
  | Block.Thematic_break _ -> Thematic_break
  | Block.List (l, _) ->
      let list_type =
        match Block.List'.type' l with
        | `Unordered c -> Unordered c
        | `Ordered (start, c) -> Ordered (start, c)
      in
      let spacing = if Block.List'.tight l then Tight else Loose in
      let items =
        List.map
          (fun (item, _) ->
            let blocks = Block.List_item.block item in
            process_cmarkit_block label_defs blocks)
          (Block.List'.items l)
      in
      List (list_type, spacing, items)
  | Block.Html_block (html, _) ->
      let html_string =
        List.map Block_line.to_string html |> String.concat "\n"
      in
      Html_block html_string
  | Block.Link_reference_definition ld -> Link_reference_definition ld
  | Block.Blocks (items, _) ->
      Blocks (List.map (block_of_cmarkit label_defs) items)
  | Block.Blank_line _ -> Blank_line ()
  | _ ->
      (* Ignore unknown block types *)
      Paragraph
        { id = next_run_id (); inline_content = Run ""; focused = false }

and process_cmarkit_block label_defs block =
  match Block.normalize block with
  | Block.Blocks (items, _) -> process_blocks label_defs items
  | other -> process_blocks label_defs [ other ]

and block_of_cmarkit label_defs cb : block =
  let id = next_block_id () in
  { id; content = block_content_of_cmarkit label_defs cb; focused = false }

and process_blocks label_defs (cmarkit_blocks : Block.t list) : t =
  match cmarkit_blocks with
  | [] -> []
  | Block.Code_block (codeblock, _) :: rest -> (
      let code =
        List.map Block_line.to_string (Block.Code_block.code codeblock)
        |> String.concat "\n"
      in
      let info =
        match Block.Code_block.info_string codeblock with
        | Some (info_str, _) -> Some (String.trim info_str)
        | None -> None
      in
      match rest with
      | Block.Html_block (html, _) :: rest' when is_output_start html ->
          let output_blocks, remaining = collect_output_blocks rest' in
          let output =
            match List.map (block_of_cmarkit label_defs) output_blocks with
            | [] -> None
            | [ b ] -> Some b
            | bs -> Some (blocks bs)
          in
          let codeblock =
            {
              id = next_block_id ();
              content = Codeblock { code; output; info };
              focused = false;
            }
          in
          codeblock :: process_blocks label_defs remaining
      | _ ->
          let codeblock =
            {
              id = next_block_id ();
              content = Codeblock { code; output = None; info };
              focused = false;
            }
          in
          codeblock :: process_blocks label_defs rest)
  | other :: rest ->
      let block = block_of_cmarkit label_defs other in
      block :: process_blocks label_defs rest

and collect_output_blocks blocks =
  let rec loop acc = function
    | [] -> (List.rev acc, [])
    | Block.Html_block (html, _) :: rest when is_output_end html ->
        (List.rev acc, rest)
    | b :: rest -> loop (b :: acc) rest
  in
  loop [] blocks

let document_of_cmarkit doc =
  let label_defs = Doc.defs doc in
  let root = Doc.block doc in
  match Block.normalize root with
  | Block.Blocks (items, _) -> process_blocks label_defs items
  | other -> process_blocks label_defs [ other ]

let of_markdown text =
  let doc = Doc.of_string ~strict:true text in
  document_of_cmarkit doc

let block_content_of_md text =
  let doc = Doc.of_string ~strict:true text in
  let label_defs = Doc.defs doc in
  let block = Doc.block doc in
  let normalized_block = Block.normalize block in
  block_content_of_cmarkit label_defs normalized_block

let block_of_md text =
  let doc = Doc.of_string ~strict:true text in
  let label_defs = Doc.defs doc in
  let block = Doc.block doc in
  let normalized_block = Block.normalize block in
  block_of_cmarkit label_defs normalized_block

let inline_content_of_md txt =
  let doc = Doc.of_string ~strict:true txt in
  let label_defs = Doc.defs doc in
  let block = Doc.block doc in
  match Block.normalize block with
  | Block.Paragraph (p, _) ->
      let inline = Inline.normalize (Block.Paragraph.inline p) in
      let i = inline_of_cmarkit label_defs inline in
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
  | Break typ -> Inline.Break (Inline.Break.make typ, Meta.none)
  | Image { alt; src } ->
      let link_def = Link_definition.make ~dest:(src, Meta.none) () in
      let reference = `Inline (link_def, Meta.none) in
      let link = Inline.Link.make (cmarkit_of_inline alt) reference in
      Inline.Image (link, Meta.none)
  | Link { text; href } ->
      let link_def = Link_definition.make ~dest:(href, Meta.none) () in
      let reference = `Inline (link_def, Meta.none) in
      let link = Inline.Link.make (cmarkit_of_inline text) reference in
      Inline.Link (link, Meta.none)
  | Raw_html html ->
      let lines = Block_line.tight_list_of_string html in
      Inline.Raw_html (lines, Meta.none)

let rec cmarkit_of_block_content (bc : block_content) : Block.t =
  match bc with
  | Paragraph inline ->
      Block.Paragraph
        (Block.Paragraph.make (cmarkit_of_inline inline), Meta.none)
  | Codeblock { code; output; info } ->
      let info_string =
        match info with Some i -> Some (i, Meta.none) | None -> None
      in
      let code_block =
        Block.Code_block
          ( Block.Code_block.make ?info_string (Block_line.list_of_string code),
            Meta.none )
      in
      let output_blocks =
        match output with
        | None -> []
        | Some block ->
            let start_html =
              Block.Html_block
                ( Block_line.list_of_string "<!-- quill=output_start -->",
                  Meta.none )
            in
            let end_html =
              Block.Html_block
                ( Block_line.list_of_string "<!-- quill=output_end -->",
                  Meta.none )
            in
            let output_cmarkit = cmarkit_of_block block in
            [ start_html; output_cmarkit; end_html ]
      in
      Block.Blocks ([ code_block ] @ output_blocks, Meta.none)
  | Heading (level, inline) ->
      Block.Heading
        (Block.Heading.make ~level (cmarkit_of_inline inline), Meta.none)
  | Block_quote blocks ->
      let cmarkit_blocks = List.map cmarkit_of_block blocks in
      let combined = Block.Blocks (cmarkit_blocks, Meta.none) in
      Block.Block_quote (Block.Block_quote.make combined, Meta.none)
  | Thematic_break ->
      Block.Thematic_break (Block.Thematic_break.make (), Meta.none)
  | List (list_type, spacing, items) ->
      let cmarkit_type =
        match list_type with
        | Unordered c -> `Unordered c
        | Ordered (start, c) -> `Ordered (start, c)
      in
      let tight = spacing = Tight in
      let cmarkit_items =
        List.map
          (fun blocks ->
            let cmarkit_blocks = List.map cmarkit_of_block blocks in
            let combined = Block.Blocks (cmarkit_blocks, Meta.none) in
            (Block.List_item.make combined, Meta.none))
          items
      in
      Block.List (Block.List'.make cmarkit_type ~tight cmarkit_items, Meta.none)
  | Html_block html ->
      Block.Html_block (Block_line.list_of_string html, Meta.none)
  | Link_reference_definition ld -> Block.Link_reference_definition ld
  | Blank_line () -> Block.Blank_line ("", Meta.none)
  | Blocks bs -> Block.Blocks (List.map cmarkit_of_block bs, Meta.none)

and cmarkit_of_block (b : block) : Block.t = cmarkit_of_block_content b.content

let cmarkit_of_document (doc : block list) : Block.t =
  let blocks = List.map cmarkit_of_block doc in
  let flattened =
    List.map (function Block.Blocks (bs, _) -> bs | b -> [ b ]) blocks
    |> List.flatten
  in
  Block.Blocks (flattened, Meta.none)

let to_markdown document : string =
  let block = cmarkit_of_document document in
  let doc = Doc.make block in
  Cmarkit_commonmark.of_doc doc

let rec find_inline_in_inline (inline : inline) id : inline option =
  if inline.id = id then Some inline
  else
    match inline.inline_content with
    | Emph inner -> find_inline_in_inline inner id
    | Strong inner -> find_inline_in_inline inner id
    | Seq items ->
        let rec find_in_seq = function
          | [] -> None
          | item :: rest -> (
              match find_inline_in_inline item id with
              | Some found -> Some found
              | None -> find_in_seq rest)
        in
        find_in_seq items
    | Run _ | Code_span _ | Break _ | Raw_html _ -> None
    | Image { alt; _ } -> find_inline_in_inline alt id
    | Link { text; _ } -> find_inline_in_inline text id

let rec find_inline_in_blocks blocks id =
  match blocks with
  | [] -> None
  | block :: rest -> (
      match find_inline_in_block block id with
      | Some inline -> Some inline
      | None -> find_inline_in_blocks rest id)

and find_inline_in_block (block : block) id : inline option =
  match block.content with
  | Paragraph inline | Heading (_, inline) -> find_inline_in_inline inline id
  | Blocks bs | Block_quote bs -> find_inline_in_blocks bs id
  | List (_, _, items) ->
      let rec find_in_items = function
        | [] -> None
        | item :: rest -> (
            match find_inline_in_blocks item id with
            | Some inline -> Some inline
            | None -> find_in_items rest)
      in
      find_in_items items
  | Codeblock _ | Blank_line _ | Thematic_break | Html_block _
  | Link_reference_definition _ ->
      None

let split_inline (inline : inline) (offset : int) : inline * inline =
  match inline.inline_content with
  | Run s ->
      let before = String.sub s 0 offset in
      let after = String.sub s offset (String.length s - offset) in
      let before_inline = run before in
      let after_inline = run after in
      (before_inline, after_inline)
  | Code_span s ->
      let before = String.sub s 0 offset in
      let after = String.sub s offset (String.length s - offset) in
      let before_inline = code_span before in
      let after_inline = code_span after in
      (before_inline, after_inline)
  | Emph _ | Strong _ | Seq _ | Break _ | Image _ | Link _ | Raw_html _ ->
      failwith "Cannot split inline with this content"

let rec replace_inline_in_inline (inline : inline) id (new_inline : inline) =
  if inline.id = id then new_inline
  else
    match inline.inline_content with
    | Emph inner when inner.id = id ->
        let new_content = Emph new_inline in
        { inline with inline_content = new_content }
    | Strong inner when inner.id = id ->
        let new_content = Strong new_inline in
        { inline with inline_content = new_content }
    | Seq items ->
        let updated_items =
          List.map (fun i -> replace_inline_in_inline i id new_inline) items
        in
        { inline with inline_content = Seq updated_items }
    | _ -> inline

let rec replace_inline_in_block (block : block) id (new_inline : inline) =
  match block.content with
  | Paragraph inline ->
      let new_inline = replace_inline_in_inline inline id new_inline in
      paragraph new_inline
  | Heading (level, inline) ->
      let new_inline = replace_inline_in_inline inline id new_inline in
      heading level new_inline
  | Blocks bs ->
      let updated_blocks =
        List.map (fun b -> replace_inline_in_block b id new_inline) bs
      in
      blocks updated_blocks
  | Block_quote bs ->
      let updated_blocks =
        List.map (fun b -> replace_inline_in_block b id new_inline) bs
      in
      block_quote updated_blocks
  | List (list_type, spacing, items) ->
      let updated_items =
        List.map
          (fun item ->
            List.map (fun b -> replace_inline_in_block b id new_inline) item)
          items
      in
      list list_type spacing updated_items
  | _ -> block

let normalize_blanklines (blocks : t) =
  let rec aux acc = function
    | [] -> List.rev acc
    | ({ content = Blank_line (); _ } as b1)
      :: { content = Blank_line (); id; _ }
      :: ({ content = Blank_line (); _ } as b2)
      :: rest ->
        aux
          (b1
          :: { id; content = Paragraph (run ""); focused = false }
          :: b2 :: acc)
          rest
    | el :: rest -> aux (el :: acc) rest
  in
  aux [] blocks

let rec set_focused_inline_by_id (inline : inline) (target_id : int) =
  let id_match = inline.id = target_id in
  let inline_content, child_focused =
    match inline.inline_content with
    | Run s -> (Run s, false)
    | Code_span s -> (Code_span s, false)
    | Emph ic ->
        let ic' = set_focused_inline_by_id ic target_id in
        (Emph ic', ic'.focused)
    | Strong ic ->
        let ic' = set_focused_inline_by_id ic target_id in
        (Strong ic', ic'.focused)
    | Seq items ->
        let items' =
          List.map (fun i -> set_focused_inline_by_id i target_id) items
        in
        let focused = List.exists (fun (i : inline) -> i.focused) items' in
        (Seq items', focused)
    | Break typ -> (Break typ, false)
    | Image { alt; src } ->
        let alt' = set_focused_inline_by_id alt target_id in
        (Image { alt = alt'; src }, alt'.focused)
    | Link { text; href } ->
        let text' = set_focused_inline_by_id text target_id in
        (Link { text = text'; href }, text'.focused)
    | Raw_html html -> (Raw_html html, false)
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
    | Block_quote bs ->
        Block_quote (List.map (fun b -> set_focused_block_by_id b target_id) bs)
    | List (list_type, spacing, items) ->
        let updated_items =
          List.map
            (fun item ->
              List.map (fun b -> set_focused_block_by_id b target_id) item)
            items
        in
        List (list_type, spacing, updated_items)
    | Codeblock s -> Codeblock s
    | Blank_line () -> Blank_line ()
    | Thematic_break -> Thematic_break
    | Html_block s -> Html_block s
    | Link_reference_definition ld -> Link_reference_definition ld
  in
  let child_focused =
    match content with
    | Paragraph inline | Heading (_, inline) -> inline.focused
    | Blocks bs | Block_quote bs -> List.exists (fun b -> b.focused) bs
    | List (_, _, items) ->
        List.exists (fun item -> List.exists (fun b -> b.focused) item) items
    | _ -> false
  in
  { block with focused = child_focused; content }

let set_focused_document_by_id (doc : t) (target_id : int) : t =
  List.map (fun b -> set_focused_block_by_id b target_id) doc

let rec set_codeblock_output_in_block (block : block) (target_id : int)
    (output : block) : block =
  match block.content with
  | Codeblock { code; output = _; info } when block.id = target_id ->
      { block with content = Codeblock { code; output = Some output; info } }
  | Blocks bs ->
      {
        block with
        content =
          Blocks
            (List.map
               (fun b -> set_codeblock_output_in_block b target_id output)
               bs);
      }
  | Block_quote bs ->
      {
        block with
        content =
          Block_quote
            (List.map
               (fun b -> set_codeblock_output_in_block b target_id output)
               bs);
      }
  | List (list_type, spacing, items) ->
      {
        block with
        content =
          List
            ( list_type,
              spacing,
              List.map
                (fun item ->
                  List.map
                    (fun b -> set_codeblock_output_in_block b target_id output)
                    item)
                items );
      }
  | _ -> block

let rec clear_focus_block (block : block) : block =
  let content =
    match block.content with
    | Paragraph inline -> Paragraph (clear_focus_inline inline)
    | Heading (lvl, inline) -> Heading (lvl, clear_focus_inline inline)
    | Blocks bs -> Blocks (List.map clear_focus_block bs)
    | Block_quote bs -> Block_quote (List.map clear_focus_block bs)
    | List (list_type, spacing, items) ->
        List
          ( list_type,
            spacing,
            List.map (fun item -> List.map clear_focus_block item) items )
    | Codeblock c -> Codeblock c
    | Blank_line () -> Blank_line ()
    | Thematic_break -> Thematic_break
    | Html_block s -> Html_block s
    | Link_reference_definition ld -> Link_reference_definition ld
  in
  { block with focused = false; content }

and clear_focus_inline (inline : inline) : inline =
  let inline_content =
    match inline.inline_content with
    | Run s -> Run s
    | Code_span s -> Code_span s
    | Emph ic -> Emph (clear_focus_inline ic)
    | Strong ic -> Strong (clear_focus_inline ic)
    | Seq items -> Seq (List.map clear_focus_inline items)
    | Break typ -> Break typ
    | Image { alt; src } -> Image { alt = clear_focus_inline alt; src }
    | Link { text; href } -> Link { text = clear_focus_inline text; href }
    | Raw_html html -> Raw_html html
  in
  { inline with focused = false; inline_content }

let rec inline_content_to_plain (inline_content : inline_content) =
  match inline_content with
  | Run s -> s
  | Emph i -> inline_to_plain i
  | Strong i -> inline_to_plain i
  | Code_span s -> s
  | Seq items -> String.concat "" (List.map inline_to_plain items)
  | Break _ -> " "
  | Image { alt; _ } -> inline_to_plain alt
  | Link { text; _ } -> inline_to_plain text
  | Raw_html _ -> ""

and inline_to_plain (inline : inline) =
  inline_content_to_plain inline.inline_content

let reset_ids () =
  next_block_id_ref := 0;
  next_run_id_ref := 0

let focus_inline_by_id document inline_id =
  document |> List.map clear_focus_block |> fun doc ->
  set_focused_document_by_id doc inline_id

let focus_block_by_id document block_id =
  document |> List.map clear_focus_block
  |> List.map (fun block ->
         if block.id = block_id then { block with focused = true } else block)

let replace_block_with_codeblock document ~block_id =
  document
  |> List.map (fun block ->
         if block.id = block_id then
           {
             block with
             content = Codeblock { code = ""; output = None; info = None };
             focused = false;
           }
         else block)

let rec update_codeblock_in_block block target_id code =
  if block.id = target_id then
    match block.content with
    | Codeblock { output; info; _ } ->
        { block with content = Codeblock { code; output; info } }
    | _ -> block
  else
    match block.content with
    | Blocks bs ->
        let content =
          Blocks
            (List.map (fun b -> update_codeblock_in_block b target_id code) bs)
        in
        { block with content }
    | _ -> block

let update_codeblock document ~block_id ~code =
  List.map (fun block -> update_codeblock_in_block block block_id code) document

let set_codeblock_output document ~block_id output_block =
  List.map
    (fun block -> set_codeblock_output_in_block block block_id output_block)
    document

let split_block_at_inline document ~block_id ~inline_id ~offset =
  let split_block block =
    match find_inline_in_block block inline_id with
    | None -> [ block ]
    | Some inline ->
        let before, after = split_inline inline offset in
        let new_block1 = replace_inline_in_block block inline_id before in
        let new_block2 = replace_inline_in_block block inline_id after in
        [ new_block1; new_block2 ]
  in
  document
  |> List.map (fun block ->
         if block.id = block_id then split_block block else [ block ])
  |> List.flatten

let find_block document ~block_id =
  List.find_opt (fun block -> block.id = block_id) document

let index_of_block document ~block_id =
  let rec aux idx = function
    | [] -> None
    | block :: rest ->
        if block.id = block_id then Some idx else aux (idx + 1) rest
  in
  aux 0 document

let find_block_of_inline document ~inline_id =
  let rec search_block index = function
    | [] -> None
    | block :: rest -> (
        match find_inline_in_block block inline_id with
        | Some _ -> Some (block, index)
        | None -> search_block (index + 1) rest)
  in
  search_block 0 document

let block_ids_between document ~start_id ~end_id =
  match
    ( index_of_block document ~block_id:start_id,
      index_of_block document ~block_id:end_id )
  with
  | Some start_idx, Some end_idx ->
      let i_min = min start_idx end_idx in
      let i_max = max start_idx end_idx in
      document
      |> List.mapi (fun idx block ->
             if idx >= i_min && idx <= i_max then Some block.id else None)
      |> List.filter_map (fun x -> x)
  | _ -> []

let slice_between document ~start_id ~end_id =
  match
    ( index_of_block document ~block_id:start_id,
      index_of_block document ~block_id:end_id )
  with
  | Some start_idx, Some end_idx ->
      let i_min = min start_idx end_idx in
      let i_max = max start_idx end_idx in
      let rec aux idx acc = function
        | [] -> List.rev acc
        | block :: rest ->
            if idx > i_max then List.rev acc
            else
              let acc = if idx >= i_min then block :: acc else acc in
              aux (idx + 1) acc rest
      in
      aux 0 [] document
  | _ -> []
