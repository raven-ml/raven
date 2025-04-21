open Model

type msg = Focus_inline of int * int | Set_document of block list

let log fmt =
  Printf.ksprintf
    (fun s ->
      Js_of_ocaml.Console.console##log (Js_of_ocaml.Js.string ("[update] " ^ s)))
    fmt

let rec clear_focus_block (b : block) : block =
  {
    b with
    focused = false;
    content =
      (match b.content with
      | Paragraph inline -> Paragraph (clear_focus_inline inline)
      | Codeblock s -> Codeblock s
      | Heading (level, inline) -> Heading (level, clear_focus_inline inline)
      | Blocks bs -> Blocks (List.map clear_focus_block bs)
      | Blank_line () -> Blank_line ());
  }

and clear_focus_inline (i : inline) : inline =
  {
    i with
    focused = false;
    content =
      (match i.content with
      | Run s -> Run s
      | Emph ic -> Emph (clear_focus_inline_content ic)
      | Strong ic -> Strong (clear_focus_inline_content ic)
      | Seq items -> Seq (List.map clear_focus_inline items));
  }

and clear_focus_inline_content (ic : inline_content) : inline_content =
  match ic with
  | Run s -> Run s
  | Emph ic -> Emph (clear_focus_inline_content ic)
  | Strong ic -> Strong (clear_focus_inline_content ic)
  | Seq items -> Seq (List.map clear_focus_inline items)

let rec set_focus_block (b : block) (target_id : int) (run_j : int) : block =
  if b.id = target_id then
    match b.content with
    | Paragraph inline ->
        let new_inline = set_focus_inline inline run_j in
        { b with content = Paragraph new_inline }
    | Heading (level, inline) ->
        let new_inline = set_focus_inline inline run_j in
        { b with content = Heading (level, new_inline) }
    | Blocks bs ->
        {
          b with
          content =
            Blocks (List.map (fun b' -> set_focus_block b' target_id run_j) bs);
        }
    | _ -> b (* No inlines to focus in Codeblock or Blank_line *)
  else
    match b.content with
    | Blocks bs ->
        {
          b with
          content =
            Blocks (List.map (fun b' -> set_focus_block b' target_id run_j) bs);
        }
    | _ -> clear_focus_block b

and set_focus_inline (inline : inline) (run_j : int) : inline =
  match inline.content with
  | Seq items ->
      let new_items =
        List.mapi
          (fun j item : inline -> { item with focused = j = run_j })
          items
      in
      { inline with content = Seq new_items }
  | _ -> { inline with focused = true }

let set_focus_document (doc : block list) (block_id : int) (run_j : int) :
    block list =
  List.map (fun b -> set_focus_block b block_id run_j) doc

let update (m : model) (message : msg) : model =
  match message with
  | Focus_inline (block_id, run_j) ->
      log "Focus_inline: block_id=%d, run_j=%d" block_id run_j;
      let new_document = set_focus_document m.document block_id run_j in
      { document = new_document }
  | Set_document docs ->
      log "Set_document: %d blocks" (List.length docs);
      { document = docs }
