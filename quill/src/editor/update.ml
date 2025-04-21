open Model

type msg = Focus_inline_by_id of int | Set_document of block list

let log fmt =
  Printf.ksprintf
    (fun s ->
      Js_of_ocaml.Console.console##log (Js_of_ocaml.Js.string ("[update] " ^ s)))
    fmt

let rec set_focused_inline_by_id (inline : inline) (target_id : int) : inline =
  let id_match = inline.id = target_id in
  let content =
    match inline.content with
    | Run s -> Run s
    | Code_span s -> Code_span s
    | Emph ic -> Emph (set_focused_inline_by_id ic target_id)
    | Strong ic -> Strong (set_focused_inline_by_id ic target_id)
    | Seq items ->
        Seq (List.map (fun i -> set_focused_inline_by_id i target_id) items)
  in
  let child_focused =
    match content with
    | Emph ic | Strong ic -> ic.focused
    | Seq items -> List.exists (fun (i : inline) -> i.focused) items
    | _ -> false
  in
  { inline with focused = id_match || child_focused; content }

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

let update (m : model) (message : msg) : model =
  match message with
  | Focus_inline_by_id inline_id ->
      let new_document = set_focused_document_by_id m.document inline_id in
      { document = new_document }
  | Set_document docs -> { document = docs }
