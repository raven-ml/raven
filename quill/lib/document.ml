type block_id = int
type inline_id = int

let block_id_of_int n = n
let inline_id_of_int n = n

type inline_content =
  | Run of string
  | Emph of inline
  | Strong of inline
  | Code_span of string
  | Seq of inline list
  | Break of [ `Hard | `Soft ]
  | Image of { alt : inline; src : string }
  | Link of { text : inline; href : string }
  | Raw_html of string

and inline = { id : inline_id; content : inline_content }

type list_type = Ordered of int * char | Unordered of char
type list_spacing = Tight | Loose

type block_content =
  | Paragraph of inline
  | Codeblock of {
      code : string;
      language : string option;
      output :
        block list option (* Output blocks associated with this code block *);
    }
  | Heading of int * inline
  | Blank_line
  | Block_quote of block list
  | Thematic_break
  | List of list_type * list_spacing * block list list
  | Html_block of string

and block = { id : block_id; content : block_content }

type t = { blocks : block list }

let empty = { blocks = [] }

let make_inline ~id inline_content =
  let inline : inline = { id; content = inline_content } in
  inline

let make_block ~id block_content =
  let block : block = { id; content = block_content } in
  block

let run ?(focused = false) ~id text =
  ignore focused;
  (* unused for now *)
  let content = Run text in
  make_inline ~id content

let paragraph ~id inline = make_block ~id (Paragraph inline)

let codeblock ?language ?output ~id code =
  make_block ~id (Codeblock { code; language; output })

let heading ~id level inline = make_block ~id (Heading (level, inline))
let html_block ~id html = make_block ~id (Html_block html)
let add_block doc block = { blocks = doc.blocks @ [ block ] }

let rec insert_after_aux blocks block_id new_block =
  match blocks with
  | [] -> []
  | b :: rest ->
      if b.id = block_id then b :: new_block :: rest
      else b :: insert_after_aux rest block_id new_block

let insert_after doc block_id block =
  { blocks = insert_after_aux doc.blocks block_id block }

let remove_block doc block_id =
  { blocks = List.filter (fun b -> b.id <> block_id) doc.blocks }

let update_block_content doc block_id content =
  let blocks =
    List.map
      (fun b -> if b.id = block_id then { b with content } else b)
      doc.blocks
  in
  { blocks }

let find_block doc block_id =
  List.find_opt (fun b -> b.id = block_id) doc.blocks

let rec find_inline_in_content (content : inline_content) inline_id =
  let find_in_inline (inline : inline) =
    if inline.id = inline_id then Some inline
    else find_inline_in_content inline.content inline_id
  in
  match content with
  | Run _ | Code_span _ | Break _ | Raw_html _ -> None
  | Emph inline | Strong inline -> find_in_inline inline
  | Image { alt; _ } -> find_in_inline alt
  | Link { text; _ } -> find_in_inline text
  | Seq inlines -> List.find_map find_in_inline inlines

let rec find_inline_in_block_content (content : block_content) inline_id =
  match content with
  | Paragraph inline | Heading (_, inline) ->
      if inline.id = inline_id then Some inline
      else find_inline_in_content inline.content inline_id
  | Block_quote blocks ->
      List.find_map
        (fun b -> find_inline_in_block_content b.content inline_id)
        blocks
  | List (_, _, items) ->
      List.find_map
        (fun blocks ->
          List.find_map
            (fun b -> find_inline_in_block_content b.content inline_id)
            blocks)
        items
  | Codeblock _ | Blank_line | Thematic_break | Html_block _ -> None

let find_inline doc inline_id =
  List.find_map
    (fun b -> find_inline_in_block_content b.content inline_id)
    doc.blocks

let get_blocks doc = doc.blocks
let block_count doc = List.length doc.blocks

let blocks_from doc block_id =
  let rec aux blocks found =
    match blocks with
    | [] -> []
    | b :: rest ->
        if found || b.id = block_id then b.id :: aux rest true
        else aux rest false
  in
  aux doc.blocks false

let previous_block doc block_id =
  let rec aux prev blocks =
    match blocks with
    | [] -> None
    | b :: rest -> if b.id = block_id then prev else aux (Some b.id) rest
  in
  aux None doc.blocks

let next_block doc block_id =
  let rec aux found blocks =
    match blocks with
    | [] -> None
    | b :: rest -> if found then Some b.id else aux (b.id = block_id) rest
  in
  aux false doc.blocks

let rec inline_to_text (inline : inline) =
  match inline.content with
  | Run text -> text
  | Emph inline | Strong inline -> inline_to_text inline
  | Code_span text -> text
  | Seq inlines -> String.concat "" (List.map inline_to_text inlines)
  | Break `Hard -> "\n"
  | Break `Soft -> " "
  | Image { alt; _ } -> inline_to_text alt
  | Link { text; _ } -> inline_to_text text
  | Raw_html _ -> ""

let rec block_to_text (block : block) =
  match block.content with
  | Paragraph inline -> inline_to_text inline ^ "\n"
  | Codeblock { code; _ } -> code ^ "\n"
  | Heading (_, inline) -> inline_to_text inline ^ "\n"
  | Blank_line -> "\n"
  | Block_quote blocks -> String.concat "" (List.map block_to_text blocks)
  | Thematic_break -> "---\n"
  | List (_, _, items) ->
      String.concat ""
        (List.map
           (fun blocks -> String.concat "" (List.map block_to_text blocks))
           items)
  | Html_block html -> html ^ "\n"

let to_plain_text doc = String.concat "" (List.map block_to_text doc.blocks)

let get_codeblocks doc =
  List.filter_map
    (fun block ->
      match block.content with
      | Codeblock { code; language; output = _ } ->
          Some (block.id, code, language)
      | _ -> None)
    doc.blocks

(* Helper to create styled inline elements *)
let emph ~id inline = make_inline ~id (Emph inline)
let strong ~id inline = make_inline ~id (Strong inline)
let code_span ~id text = make_inline ~id (Code_span text)
let link ~id ~href text = make_inline ~id (Link { text; href })

(* Check if an inline has a specific style *)
let rec has_style style (inline : inline) =
  match (style, inline.content) with
  | `Bold, Strong _ -> true
  | `Italic, Emph _ -> true
  | `Code, Code_span _ -> true
  | _, (Emph inner | Strong inner) -> has_style style inner
  | _, Link { text; _ } -> has_style style text
  | _, Seq inlines -> List.exists (has_style style) inlines
  | _ -> false

(* Remove a style from an inline element *)
let rec remove_style ~next_id style (inline : inline) =
  match (style, inline.content) with
  | `Bold, Strong inner -> (inner, next_id)
  | `Italic, Emph inner -> (inner, next_id)
  | `Code, Code_span text ->
      let id = next_id in
      (run ~id text, next_id + 1)
  | _, Emph inner ->
      let inner', next_id = remove_style ~next_id style inner in
      let id = next_id in
      (emph ~id inner', next_id + 1)
  | _, Strong inner ->
      let inner', next_id = remove_style ~next_id style inner in
      let id = next_id in
      (strong ~id inner', next_id + 1)
  | _, Link { text; href } ->
      let text', next_id = remove_style ~next_id style text in
      (link ~id:inline.id ~href text', next_id)
  | _, Seq inlines ->
      let inlines', next_id =
        List.fold_left
          (fun (acc, next_id) inline ->
            let inline', next_id = remove_style ~next_id style inline in
            (inline' :: acc, next_id))
          ([], next_id) inlines
      in
      (make_inline ~id:inline.id (Seq (List.rev inlines')), next_id)
  | _ -> (inline, next_id)

(* Apply a style to an inline element *)
let apply_style ~next_id style (inline : inline) =
  if has_style style inline then (inline, next_id)
  else
    let id = next_id in
    match style with
    | `Bold -> (strong ~id inline, next_id + 1)
    | `Italic -> (emph ~id inline, next_id + 1)
    | `Code -> (
        match inline.content with
        | Run text -> (code_span ~id text, next_id + 1)
        | _ -> (inline, next_id))
(* Can't make non-text into code *)
