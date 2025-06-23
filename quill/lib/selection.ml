(** Selection and range utilities *)

open Document

(** A position in the document *)
type position = View.position = {
  block_id : block_id;
  offset : int;
}

(** A selection range with anchor and focus positions *)
type t = View.selection = {
  anchor : position;
  focus : position;
}

(** Create a position *)
let make_position block_id offset = { block_id; offset }

(** Create a selection *)
let make anchor focus = { anchor; focus }

(** Create a collapsed selection at a position *)
let collapsed pos = { anchor = pos; focus = pos }

(** Check if selection is collapsed *)
let is_collapsed sel = 
  sel.anchor.block_id = sel.focus.block_id && 
  sel.anchor.offset = sel.focus.offset

(** Compare two positions in document order *)
let compare_positions (doc : Document.t) (p1 : position) (p2 : position) : int =
  if p1.block_id = p2.block_id then
    compare p1.offset p2.offset
  else
    (* Find blocks in document order *)
    let rec find_order = function
      | [] -> 0  (* Neither found *)
      | b :: rest ->
          if b.id = p1.block_id then -1  (* p1 comes first *)
          else if b.id = p2.block_id then 1  (* p2 comes first *)
          else find_order rest
    in
    find_order doc.blocks

(** Normalize selection so start comes before end *)
let normalize (doc : Document.t) (sel : t) : t =
  if compare_positions doc sel.anchor sel.focus <= 0 then
    sel  (* Already normalized *)
  else
    { anchor = sel.focus; focus = sel.anchor }  (* Swap *)

(** Get the start position of a selection *)
let start (doc : Document.t) (sel : t) : position =
  let norm = normalize doc sel in
  norm.anchor

(** Get the end position of a selection *)
let end_ (doc : Document.t) (sel : t) : position =
  let norm = normalize doc sel in
  norm.focus

(** Check if a position is within a selection *)
let contains_position (doc : Document.t) (sel : t) (pos : position) : bool =
  let norm = normalize doc sel in
  compare_positions doc norm.anchor pos <= 0 &&
  compare_positions doc pos norm.focus <= 0

(** Check if two selections intersect *)
let intersects (doc : Document.t) (sel1 : t) (sel2 : t) : bool =
  let norm1 = normalize doc sel1 in
  let norm2 = normalize doc sel2 in
  (* They intersect if start1 <= end2 && start2 <= end1 *)
  compare_positions doc norm1.anchor norm2.focus <= 0 &&
  compare_positions doc norm2.anchor norm1.focus <= 0

(** Get the intersection of two selections *)
let intersection (doc : Document.t) (sel1 : t) (sel2 : t) : t option =
  if not (intersects doc sel1 sel2) then None
  else
    let norm1 = normalize doc sel1 in
    let norm2 = normalize doc sel2 in
    (* Intersection starts at the later start and ends at the earlier end *)
    let start_pos = 
      if compare_positions doc norm1.anchor norm2.anchor >= 0 
      then norm1.anchor else norm2.anchor 
    in
    let end_pos = 
      if compare_positions doc norm1.focus norm2.focus <= 0 
      then norm1.focus else norm2.focus 
    in
    Some { anchor = start_pos; focus = end_pos }

(** Get the union of two selections (smallest selection containing both) *)
let union (doc : Document.t) (sel1 : t) (sel2 : t) : t =
  let norm1 = normalize doc sel1 in
  let norm2 = normalize doc sel2 in
  (* Union starts at the earlier start and ends at the later end *)
  let start_pos = 
    if compare_positions doc norm1.anchor norm2.anchor <= 0 
    then norm1.anchor else norm2.anchor 
  in
  let end_pos = 
    if compare_positions doc norm1.focus norm2.focus >= 0 
    then norm1.focus else norm2.focus 
  in
  { anchor = start_pos; focus = end_pos }

(** Expand selection to word boundaries *)
let expand_to_word (doc : Document.t) (sel : t) : t option =
  match Document.find_block doc sel.focus.block_id with
  | None -> None
  | Some block ->
      let text = match block.content with
        | Paragraph inline | Heading (_, inline) -> 
            Document.inline_to_text inline
        | Codeblock { code; _ } -> code
        | _ -> ""
      in
      if text = "" then None
      else
        (* Find word boundaries around focus position *)
        let start_boundary, end_boundary = 
          Text.find_word_boundaries text sel.focus.offset 
        in
        Some {
          anchor = { sel.focus with offset = start_boundary };
          focus = { sel.focus with offset = end_boundary };
        }

(** Expand selection to line boundaries *)
let expand_to_line (doc : Document.t) (sel : t) : t option =
  match Document.find_block doc sel.focus.block_id with
  | None -> None
  | Some block ->
      let text = match block.content with
        | Paragraph inline | Heading (_, inline) -> 
            Document.inline_to_text inline
        | Codeblock { code; _ } -> code
        | _ -> ""
      in
      if text = "" then None
      else
        (* Find line boundaries *)
        let lines = String.split_on_char '\n' text in
        let rec find_line offset line_start = function
          | [] -> (line_start, String.length text)
          | line :: rest ->
              let line_end = line_start + String.length line in
              if offset <= line_end then
                (line_start, line_end)
              else
                find_line offset (line_end + 1) rest
        in
        let start_offset, end_offset = find_line sel.focus.offset 0 lines in
        Some {
          anchor = { sel.focus with offset = start_offset };
          focus = { sel.focus with offset = end_offset };
        }

(** Expand selection to entire block *)
let expand_to_block (doc : Document.t) (sel : t) : t option =
  match Document.find_block doc sel.focus.block_id with
  | None -> None
  | Some block ->
      let text = match block.content with
        | Paragraph inline | Heading (_, inline) -> 
            Document.inline_to_text inline
        | Codeblock { code; _ } -> code
        | _ -> ""
      in
      Some {
        anchor = { block_id = block.id; offset = 0 };
        focus = { block_id = block.id; offset = String.length text };
      }

(** Multi-cursor support: merge overlapping selections *)
let merge_overlapping (doc : Document.t) (selections : t list) : t list =
  let sorted = List.sort (fun s1 s2 ->
    compare_positions doc (start doc s1) (start doc s2)
  ) selections in
  
  let rec merge acc = function
    | [] -> List.rev acc
    | [s] -> List.rev (s :: acc)
    | s1 :: s2 :: rest ->
        if intersects doc s1 s2 then
          (* Merge s1 and s2 *)
          merge acc (union doc s1 s2 :: rest)
        else
          merge (s1 :: acc) (s2 :: rest)
  in
  merge [] sorted

(** Get text content of a selection *)
let get_text (doc : Document.t) (sel : t) : string =
  let norm = normalize doc sel in
  if norm.anchor.block_id = norm.focus.block_id then
    (* Single block selection *)
    match Document.find_block doc norm.anchor.block_id with
    | None -> ""
    | Some block ->
        let text = match block.content with
          | Paragraph inline | Heading (_, inline) -> 
              Document.inline_to_text inline
          | Codeblock { code; _ } -> code
          | _ -> ""
        in
        let start_offset = max 0 (min norm.anchor.offset (String.length text)) in
        let end_offset = max 0 (min norm.focus.offset (String.length text)) in
        String.sub text start_offset (end_offset - start_offset)
  else
    (* Multi-block selection *)
    let rec collect acc in_range = function
      | [] -> String.concat "\n" (List.rev acc)
      | b :: rest ->
          if b.id = norm.anchor.block_id then
            (* Start block *)
            let text = match b.content with
              | Document.Paragraph inline | Document.Heading (_, inline) -> 
                  Document.inline_to_text inline
              | Document.Codeblock { code; _ } -> code
              | _ -> ""
            in
            let start_offset = max 0 (min norm.anchor.offset (String.length text)) in
            let partial = String.sub text start_offset (String.length text - start_offset) in
            collect (partial :: acc) true rest
          else if b.id = norm.focus.block_id then
            (* End block *)
            let text = match b.content with
              | Document.Paragraph inline | Document.Heading (_, inline) -> 
                  Document.inline_to_text inline
              | Document.Codeblock { code; _ } -> code
              | _ -> ""
            in
            let end_offset = max 0 (min norm.focus.offset (String.length text)) in
            let partial = String.sub text 0 end_offset in
            collect (partial :: acc) false rest
          else if in_range then
            (* Middle block *)
            let text = match b.content with
              | Document.Paragraph inline | Document.Heading (_, inline) -> 
                  Document.inline_to_text inline
              | Document.Codeblock { code; _ } -> code
              | _ -> ""
            in
            collect (text :: acc) true rest
          else
            collect acc false rest
    in
    collect [] false doc.blocks