open Model
open Vdom

let rec inline_to_debug_string (indent : int) (i : inline) : string =
  let indent_str = String.make (indent * 2) ' ' in
  let content =
    match i.inline_content with
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
  let focus_str = if i.focused then "[FOCUSED]" else "" in
  Printf.sprintf "%sid=%d %s%s" indent_str i.id content focus_str

let rec has_focused_inline (b : block) : bool =
  let rec check_inline (i : inline) : bool =
    i.focused
    ||
    match i.inline_content with
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
  | Blocks bs -> List.exists has_focused_inline bs
  | Blank_line () -> false

let rec block_to_debug_string (indent : int) (b : block) : string =
  let indent_str = String.make (indent * 2) ' ' in
  let content =
    match b.content with
    | Paragraph inline ->
        Printf.sprintf "%sParagraph (\n%s\n%s)" indent_str
          (inline_to_debug_string (indent + 1) inline)
          indent_str
    | Codeblock { code; output } ->
        let inner_parts =
          [
            Printf.sprintf "%sCode: %s"
              (String.make ((indent + 1) * 2) ' ')
              code;
          ]
        in
        let inner_parts =
          match output with
          | None -> inner_parts
          | Some output_block ->
              inner_parts
              @ [
                  Printf.sprintf "%sOutput:"
                    (String.make ((indent + 1) * 2) ' ');
                  block_to_debug_string (indent + 2) output_block;
                ]
        in
        let inner_str = String.concat "\n" inner_parts in
        Printf.sprintf "%sCodeblock (\n%s\n%s)" indent_str inner_str indent_str
    | Heading (level, inline) ->
        Printf.sprintf "%sHeading %d (\n%s\n%s)" indent_str level
          (inline_to_debug_string (indent + 1) inline)
          indent_str
    | Blocks bs ->
        let bs_str =
          List.map (block_to_debug_string (indent + 1)) bs |> String.concat "\n"
        in
        Printf.sprintf "%sBlocks [\n%s\n%s]" indent_str bs_str indent_str
    | Blank_line () ->
        Printf.sprintf "%sBlank_line ()%s" indent_str
          (String.make ((indent + 1) * 2) ' ')
  in
  let focus_str =
    if b.focused then " [FOCUSED]"
    else if has_focused_inline b then " [CONTAINS FOCUS]"
    else ""
  in
  Printf.sprintf "%sid=%d %s%s" indent_str b.id content focus_str

let document_to_debug_string model : string =
  let doc_ast =
    List.map (block_to_debug_string 0) model.document |> String.concat "\n"
  in
  let doc_str = Model.md_of_model model in
  Printf.sprintf "Document [\n%s\n]\n\n---\n\n%s" doc_ast doc_str

let view model =
  let debug_str = document_to_debug_string model in
  elt "pre" [ text debug_str ]
