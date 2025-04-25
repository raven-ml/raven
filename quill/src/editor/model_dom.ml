open Model
open Brr_ext

let log fmt =
  Printf.ksprintf
    (fun s -> Brr.Console.(log [ Jstr.v ("[model_dom] " ^ s) ]))
    fmt

let parse_dom (root : El.t) : Model.block list =
  match El.find_first_by_selector (Jstr.v "#editor") ~root with
  | None ->
      log "Could not find editor element";
      []
  | Some editor_div ->
      Model.next_block_id_ref := 0;
      Model.next_run_id_ref := 0;
      let content = Jstr.to_string (El.text_content editor_div) in
      log "Parsing content: %s" content;
      Model.document_of_md (String.trim content)

let rec text_length_inline inline : int =
  match inline.inline_content with
  | Run s -> String.length s
  | Code_span s -> 2 + String.length s
  | Emph inner -> 2 + text_length_inline inner
  | Strong inner -> 4 + text_length_inline inner
  | Seq items ->
      List.fold_left (fun acc i -> acc + text_length_inline i) 0 items
  | Break _ -> 1
  | Image { alt; src } -> 2 + text_length_inline alt + 3 + String.length src + 1
  | Link { text; href } ->
      1 + text_length_inline text + 3 + String.length href + 1
  | Raw_html html -> String.length html

and text_length_block block : int =
  match block.content with
  | Paragraph inline -> text_length_inline inline + 1 (* inline + "\n" *)
  | Heading (level, inline) ->
      level + 1 + text_length_inline inline + 1 (* "# " + inline + "\n" *)
  | Codeblock { code; _ } ->
      4 + String.length code + 4 + 1 (* "```\n" + code + "\n```" + "\n" *)
  | Blank_line () -> 1 (* "\n" *)
  | Blocks bs -> List.fold_left (fun acc b -> acc + text_length_block b) 0 bs

let rec find_in_blocks (blocks : block list) (offset : int) :
    (string * int) option =
  let rec loop blocks cumulative =
    match blocks with
    | [] -> None
    | b :: rest ->
        let len = text_length_block b in
        if offset < cumulative + len then find_in_block b (offset - cumulative)
        else loop rest (cumulative + len)
  in
  loop blocks 0

and find_in_block (block : block) (offset : int) : (string * int) option =
  match block.content with
  | Paragraph inline -> find_in_inline inline offset
  | Heading (level, inline) ->
      let syntax_len = level + 1 in
      (* "# " *)
      if offset < syntax_len then None
      else find_in_inline inline (offset - syntax_len)
  | Codeblock { code; _ } ->
      let syntax_start_len = 4 in
      (* "```\n" *)
      let code_len = String.length code in
      if offset >= syntax_start_len && offset <= syntax_start_len + code_len
      then Some ("block", block.id)
      else None
  | Blank_line () -> if offset = 0 then Some ("block", block.id) else None
  | Blocks bs -> find_in_blocks bs offset

and find_in_inline (inline : inline) (offset : int) : (string * int) option =
  match inline.inline_content with
  | Run s ->
      if offset <= String.length s then Some ("inline", inline.id) else None
  | Code_span s ->
      let syntax_len = 1 in
      (* "`" *)
      let content_len = String.length s in
      if offset >= syntax_len && offset <= syntax_len + content_len then
        Some ("inline", inline.id)
      else None
  | Emph inner ->
      let syntax_len = 1 in
      (* "*" *)
      let inner_len = text_length_inline inner in
      if offset >= syntax_len && offset <= syntax_len + inner_len then
        find_in_inline inner (offset - syntax_len)
      else None
  | Strong inner ->
      let syntax_len = 2 in
      (* "**" *)
      let inner_len = text_length_inline inner in
      if offset >= syntax_len && offset <= syntax_len + inner_len then
        find_in_inline inner (offset - syntax_len)
      else None
  | Seq items ->
      let rec loop items cumulative =
        match items with
        | [] -> None
        | i :: rest ->
            let len = text_length_inline i in
            if offset <= cumulative + len then
              find_in_inline i (offset - cumulative)
            else loop rest (cumulative + len)
      in
      loop items 0
  | Break _ -> if offset = 0 then Some ("inline", inline.id) else None
  | Image { alt; _ } -> find_in_inline alt offset
  | Link { text; _ } -> find_in_inline text offset
  | Raw_html html ->
      if offset <= String.length html then Some ("inline", inline.id) else None
