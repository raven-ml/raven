(** Text length calculation utilities for document elements *)

open Document

(** Calculate the markdown representation length of an inline element *)
let rec inline (i : inline) : int =
  match i.content with
  | Run s -> String.length s
  | Code_span s -> 2 + String.length s  (* `code` *)
  | Emph inner -> 2 + inline inner      (* *emph* *)
  | Strong inner -> 4 + inline inner    (* **strong** *)
  | Seq items -> List.fold_left (fun acc item -> acc + inline item) 0 items
  | Break `Hard -> 1  (* \n *)
  | Break `Soft -> 1  (* space *)
  | Image { alt; src } -> 
      2 + inline alt + 3 + String.length src + 1  (* ![alt](src) *)
  | Link { text; href } ->
      1 + inline text + 2 + String.length href + 1  (* [text](href) *)
  | Raw_html html -> String.length html

(** Calculate the markdown representation length of a block element *)
let rec block (b : block) : int =
  match b.content with
  | Paragraph i -> inline i + 1  (* inline + \n *)
  | Heading (level, i) ->
      level + 1 + inline i + 1  (* ### heading\n *)
  | Codeblock { code; language; _ } ->
      let fence = 3 in  (* ``` *)
      let lang_len = match language with None -> 0 | Some l -> String.length l in
      fence + lang_len + 1 + String.length code + 1 + fence + 1
      (* ```lang\ncode\n```\n *)
  | Blank_line -> 1  (* \n *)
  | Thematic_break -> 4  (* ---\n *)
  | Block_quote blocks ->
      List.fold_left (fun acc b -> 
        (* Each line in the block gets "> " prefix *)
        let lines = String.split_on_char '\n' (block_to_string b) in
        let prefixed_length = List.fold_left (fun sum line ->
          sum + 2 + String.length line + 1  (* "> " + line + \n *)
        ) 0 lines in
        acc + prefixed_length
      ) 0 blocks
  | List (list_type, _spacing, items) ->
      let marker_len = match list_type with
        | Ordered _ -> 3  (* "1. " *)
        | Unordered _ -> 2  (* "- " *)
      in
      List.fold_left (fun acc item ->
        acc + List.fold_left (fun acc b ->
          (* First line gets marker, rest get spaces *)
          let lines = String.split_on_char '\n' (block_to_string b) in
          match lines with
          | [] -> acc
          | first :: rest ->
              acc + marker_len + String.length first + 1 +
              List.fold_left (fun sum line ->
                sum + marker_len + String.length line + 1
              ) 0 rest
        ) 0 item
      ) 0 items
  | Html_block html -> String.length html + 1

(** Helper to convert block to string for line counting *)
and block_to_string (b : block) : string =
  match b.content with
  | Paragraph i -> inline_to_text i
  | Heading (_, i) -> inline_to_text i
  | Codeblock { code; _ } -> code
  | Blank_line -> ""
  | Thematic_break -> "---"
  | Block_quote _ -> "[blockquote]"  (* Simplified for line counting *)
  | List _ -> "[list]"  (* Simplified for line counting *)
  | Html_block html -> html

(** Calculate total length of a document *)
let document (doc : t) : int =
  List.fold_left (fun acc b -> acc + block b) 0 doc.blocks

(** Calculate length up to a specific block (exclusive) *)
let up_to_block (doc : t) (block_id : block_id) : int =
  let rec calc acc = function
    | [] -> acc
    | b :: rest ->
        if b.id = block_id then acc
        else calc (acc + block b) rest
  in
  calc 0 doc.blocks

(** Calculate length of text within a block up to an offset *)
let in_block_up_to_offset (b : block) (offset : int) : int =
  match b.content with
  | Paragraph i | Heading (_, i) ->
      let text = inline_to_text i in
      min offset (String.length text)
  | Codeblock { code; _ } ->
      min offset (String.length code)
  | _ -> 0  (* Other blocks don't have meaningful offsets *)

(** Get the block and local offset at a document position *)
let find_position (doc : t) (doc_offset : int) : (block_id * int) option =
  let rec find acc_len = function
    | [] -> None
    | b :: rest ->
        let b_len = block b in
        if acc_len + b_len > doc_offset then
          Some (b.id, doc_offset - acc_len)
        else
          find (acc_len + b_len) rest
  in
  find 0 doc.blocks