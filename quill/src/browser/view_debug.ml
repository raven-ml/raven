open Vdom

let rec inline_to_debug_string (indent : int) (i : Quill.Document.inline) : string =
  let indent_str = String.make (indent * 2) ' ' in
  let content =
    match i.content with
    | Run s -> Printf.sprintf "%sRun \"%s\"" indent_str s
    | Emph inline' ->
        Printf.sprintf "%sEmph (\n%s\n%s)" indent_str
          (inline_to_debug_string (indent + 1) inline')
          indent_str
    | Strong inline' ->
        Printf.sprintf "%sStrong (\n%s\n%s)" indent_str
          (inline_to_debug_string (indent + 1) inline')
          indent_str
    | Code_span s -> Printf.sprintf "%sCode_span \"%s\"" indent_str s
    | Seq items ->
        let items_str =
          List.map (inline_to_debug_string (indent + 1)) items
          |> String.concat "\n"
        in
        Printf.sprintf "%sSeq [\n%s\n%s]" indent_str items_str indent_str
    | Break typ ->
        let typ_str = match typ with `Hard -> "Hard" | `Soft -> "Soft" in
        Printf.sprintf "%sBreak %s" indent_str typ_str
    | Image { alt; src } ->
        Printf.sprintf "%sImage { alt = (\n%s\n%s); src = \"%s\" }" indent_str
          (inline_to_debug_string (indent + 1) alt)
          indent_str src
    | Link { text; href } ->
        Printf.sprintf "%sLink { text = (\n%s\n%s); href = \"%s\" }" indent_str
          (inline_to_debug_string (indent + 1) text)
          indent_str href
    | Raw_html html -> Printf.sprintf "%sRaw_html \"%s\"" indent_str html
  in
  Printf.sprintf "%sid=%d %s" indent_str (i.id :> int) content

let rec has_focused_inline (view_state : Quill.View.t) (b : Quill.Document.block) : bool =
  let rec check_inline (i : Quill.Document.inline) : bool =
    (match view_state.focused_inline with
     | Some fid -> fid = i.id
     | None -> false)
    ||
    match i.content with
    | Run _ -> false
    | Emph inline' -> check_inline inline'
    | Strong inline' -> check_inline inline'
    | Code_span _ -> false
    | Seq items -> List.exists check_inline items
    | Break _ -> false
    | Image { alt; _ } -> check_inline alt
    | Link { text; _ } -> check_inline text
    | Raw_html _ -> false
  in
  match b.content with
  | Paragraph inline -> check_inline inline
  | Heading (_, inline) -> check_inline inline
  | Codeblock _ -> false
  | Block_quote bs -> List.exists (has_focused_inline view_state) bs
  | List (_, _, items) ->
      List.exists (fun item -> List.exists (has_focused_inline view_state) item) items
  | Blank_line -> false
  | Thematic_break -> false
  | Html_block _ -> false

let rec block_to_debug_string (view_state : Quill.View.t) (indent : int) (b : Quill.Document.block) : string =
  let indent_str = String.make (indent * 2) ' ' in
  let content =
    match b.content with
    | Paragraph inline ->
        Printf.sprintf "%sParagraph (\n%s\n%s)" indent_str
          (inline_to_debug_string (indent + 1) inline)
          indent_str
    | Codeblock { code; language; output } ->
        let inner_parts =
          [
            Printf.sprintf "%sCode: %s"
              (String.make ((indent + 1) * 2) ' ')
              code;
          ]
        in
        let inner_parts =
          match language with
          | None -> inner_parts
          | Some lang ->
              inner_parts @ [
                Printf.sprintf "%sLanguage: %s"
                  (String.make ((indent + 1) * 2) ' ')
                  lang
              ]
        in
        let inner_parts =
          match output with
          | None -> inner_parts
          | Some output_blocks ->
              inner_parts
              @ [
                  Printf.sprintf "%sOutput:"
                    (String.make ((indent + 1) * 2) ' ');
                ]
              @ List.map (block_to_debug_string view_state (indent + 2)) output_blocks
        in
        let inner_str = String.concat "\n" inner_parts in
        Printf.sprintf "%sCodeblock (\n%s\n%s)" indent_str inner_str indent_str
    | Heading (level, inline) ->
        Printf.sprintf "%sHeading %d (\n%s\n%s)" indent_str level
          (inline_to_debug_string (indent + 1) inline)
          indent_str
    | Blank_line ->
        Printf.sprintf "%sBlank_line" indent_str
    | Thematic_break -> Printf.sprintf "%sThematic_break" indent_str
    | Block_quote blocks ->
        let blocks_str =
          List.map (block_to_debug_string view_state (indent + 1)) blocks
          |> String.concat "\n"
        in
        Printf.sprintf "%sBlock_quote [\n%s\n%s]" indent_str blocks_str
          indent_str
    | List (list_type, spacing, items) ->
        let type_str =
          match list_type with
          | Unordered c -> Printf.sprintf "Unordered '%c'" c
          | Ordered (start, c) -> Printf.sprintf "Ordered (%d, '%c')" start c
        in
        let spacing_str =
          match spacing with Tight -> "Tight" | Loose -> "Loose"
        in
        let items_str =
          List.map
            (fun item ->
              let item_str =
                List.map (block_to_debug_string view_state (indent + 2)) item
                |> String.concat "\n"
              in
              Printf.sprintf "%s[\n%s\n%s]"
                (String.make ((indent + 1) * 2) ' ')
                item_str
                (String.make ((indent + 1) * 2) ' '))
            items
          |> String.concat "\n"
        in
        Printf.sprintf "%sList (%s, %s) [\n%s\n%s]" indent_str type_str
          spacing_str items_str indent_str
    | Html_block html ->
        Printf.sprintf "%sHtml_block \"%s\"" indent_str (String.escaped html)
  in
  let is_focused = 
    match view_state.focused_block with
    | Some fid -> fid = b.id
    | None -> false
  in
  let focus_str =
    if is_focused then " [FOCUSED]"
    else if has_focused_inline view_state b then " [CONTAINS FOCUS]"
    else ""
  in
  Printf.sprintf "%sid=%d %s%s" indent_str (b.id :> int) content focus_str

let document_to_debug_string (model : Model.t) : string =
  let document = Quill.Engine.get_document model.engine in
  let view_state = Quill.Engine.get_view model.engine in
  let blocks = Quill.Document.get_blocks document in
  
  let doc_ast =
    List.map (block_to_debug_string view_state 0) blocks |> String.concat "\n"
  in
  
  (* Convert document to markdown for display *)
  let doc_str = Quill.Markdown.serialize document
  in
  
  (* Add view state info *)
  let view_info = 
    match view_state.selection with
    | None -> "No selection"
    | Some sel ->
        if Quill.View.is_collapsed sel then
          Printf.sprintf "Cursor at block %d, offset %d" 
            (sel.focus.block_id :> int) sel.focus.offset
        else
          Printf.sprintf "Selection from block %d offset %d to block %d offset %d"
            (sel.anchor.block_id :> int) sel.anchor.offset
            (sel.focus.block_id :> int) sel.focus.offset
  in
  
  Printf.sprintf "View: %s\n\nDocument [\n%s\n]\n\n---\n\n%s" view_info doc_ast doc_str

let view model =
  let debug_str = document_to_debug_string model in
  elt "pre" [ text debug_str ]