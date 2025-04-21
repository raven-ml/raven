open Model
open Vdom

let rec inline_to_debug_string (indent : int) (i : inline) : string =
  let indent_str = String.make (indent * 2) ' ' in
  let content =
    match i.content with
    | Run s -> Printf.sprintf "%sRun \"%s\"" indent_str s
    | Emph ic ->
        Printf.sprintf "%sEmph (\n%s\n%s)" indent_str
          (inline_content_to_debug_string (indent + 1) ic)
          indent_str
    | Strong ic ->
        Printf.sprintf "%sStrong (\n%s\n%s)" indent_str
          (inline_content_to_debug_string (indent + 1) ic)
          indent_str
    | Seq items ->
        let items_str =
          List.map (inline_to_debug_string (indent + 1)) items
          |> String.concat "\n"
        in
        Printf.sprintf "%sSeq [\n%s\n%s]" indent_str items_str indent_str
  in
  let focus_str = if i.focused then " [FOCUSED]" else "" in
  Printf.sprintf "%sid=%d %s%s" indent_str i.id content focus_str

and inline_content_to_debug_string (indent : int) (ic : inline_content) : string
    =
  let indent_str = String.make (indent * 2) ' ' in
  match ic with
  | Run s -> Printf.sprintf "%sRun \"%s\"" indent_str s
  | Emph ic ->
      Printf.sprintf "%sEmph (\n%s\n%s)" indent_str
        (inline_content_to_debug_string (indent + 1) ic)
        indent_str
  | Strong ic ->
      Printf.sprintf "%sStrong (\n%s\n%s)" indent_str
        (inline_content_to_debug_string (indent + 1) ic)
        indent_str
  | Seq items ->
      let items_str =
        List.map (inline_to_debug_string (indent + 1)) items
        |> String.concat "\n"
      in
      Printf.sprintf "%sSeq [\n%s\n%s]" indent_str items_str indent_str

let rec has_focused_inline (b : block) : bool =
  let rec check_inline (i : inline) : bool =
    i.focused
    ||
    match i.content with
    | Run _ -> false
    | Emph ic -> check_inline_content ic
    | Strong ic -> check_inline_content ic
    | Seq items -> List.exists check_inline items
  and check_inline_content (ic : inline_content) : bool =
    match ic with
    | Run _ -> false
    | Emph ic -> check_inline_content ic
    | Strong ic -> check_inline_content ic
    | Seq items -> List.exists check_inline items
  in
  match b.content with
  | Paragraph inline -> check_inline inline
  | Heading (_, inline) -> check_inline inline
  | Codeblock _ -> false
  | Blocks bs -> List.exists has_focused_inline bs

let rec block_to_debug_string (indent : int) (b : block) : string =
  let indent_str = String.make (indent * 2) ' ' in
  let content =
    match b.content with
    | Paragraph inline ->
        Printf.sprintf "%sParagraph (\n%s\n%s)" indent_str
          (inline_to_debug_string (indent + 1) inline)
          indent_str
    | Codeblock code ->
        Printf.sprintf "%sCodeblock (\n%s%s\n%s)" indent_str
          (String.make ((indent + 1) * 2) ' ')
          code indent_str
    | Heading (level, inline) ->
        Printf.sprintf "%sHeading %d (\n%s\n%s)" indent_str level
          (inline_to_debug_string (indent + 1) inline)
          indent_str
    | Blocks bs ->
        let bs_str =
          List.map (block_to_debug_string (indent + 1)) bs |> String.concat "\n"
        in
        Printf.sprintf "%sBlocks [\n%s\n%s]" indent_str bs_str indent_str
  in
  let focus_str =
    if b.focused then " [FOCUSED]"
    else if has_focused_inline b then " [CONTAINS FOCUS]"
    else ""
  in
  Printf.sprintf "%sid=%d %s%s" indent_str b.id content focus_str

let document_to_debug_string (doc : block list) : string =
  let doc_str = List.map (block_to_debug_string 0) doc |> String.concat "\n" in
  Printf.sprintf "Document [\n%s\n]" doc_str

let view model =
  let debug_str = document_to_debug_string model.document in
  elt "pre" [ text debug_str ]
