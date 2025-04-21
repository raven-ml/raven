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
      | Blocks bs -> Blocks (List.map clear_focus_block bs));
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

let set_focus_document (doc : block list) (block_idx : int) (run_j : int) :
    block list =
  List.mapi
    (fun i b ->
      if i = block_idx then
        match b.content with
        | Paragraph inline ->
            let new_inline =
              match inline.content with
              | Seq items ->
                  let new_items =
                    List.mapi
                      (fun j (item : inline) ->
                        { item with focused = j = run_j })
                      items
                  in
                  { inline with content = Seq new_items }
              | _ -> { inline with focused = true }
            in
            { b with content = Paragraph new_inline }
        | Heading (level, inline) ->
            let new_inline =
              match inline.content with
              | Seq items ->
                  let new_items =
                    List.mapi
                      (fun j (item : inline) ->
                        { item with focused = j = run_j })
                      items
                  in
                  { inline with content = Seq new_items }
              | _ -> { inline with focused = true }
            in
            { b with content = Heading (level, new_inline) }
        | _ -> b
      else clear_focus_block b)
    doc

let update (m : model) (message : msg) : model =
  match message with
  | Focus_inline (block_idx, run_j) ->
      log "Focus_inline: block=%d, run_j=%d" block_idx run_j;
      let new_document = set_focus_document m.document block_idx run_j in
      { document = new_document }
  | Set_document docs ->
      log "Set_document: %d blocks" (List.length docs);
      { document = docs }
