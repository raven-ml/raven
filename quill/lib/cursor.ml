type position = View.position = { block_id : Document.block_id; offset : int }

let find_block_at_offset doc target_offset =
  let rec aux current_offset blocks =
    match blocks with
    | [] -> None
    | block :: rest ->
        let block_text = Document.block_to_text block in
        let block_len = String.length block_text in
        if current_offset + block_len > target_offset then
          Some block.Document.id
        else aux (current_offset + block_len) rest
  in
  aux 0 (Document.get_blocks doc)

let block_start_offset doc block_id =
  let rec aux current_offset blocks =
    match blocks with
    | [] -> 0
    | block :: rest ->
        if block.Document.id = block_id then current_offset
        else
          let block_text = Document.block_to_text block in
          aux (current_offset + String.length block_text) rest
  in
  aux 0 (Document.get_blocks doc)

let document_offset doc position =
  block_start_offset doc position.block_id + position.offset

let move_cursor doc pos direction =
  match direction with
  | `Left -> (
      if pos.offset > 0 then Some { pos with offset = pos.offset - 1 }
      else
        match Document.previous_block doc pos.block_id with
        | None -> None
        | Some prev_id -> (
            match Document.find_block doc prev_id with
            | None -> None
            | Some block ->
                let text = Document.block_to_text block in
                Some { block_id = prev_id; offset = String.length text - 1 }))
  | `Right -> (
      match Document.find_block doc pos.block_id with
      | None -> None
      | Some block -> (
          let text = Document.block_to_text block in
          if pos.offset < String.length text - 1 then
            Some { pos with offset = pos.offset + 1 }
          else
            match Document.next_block doc pos.block_id with
            | None -> None
            | Some next_id -> Some { block_id = next_id; offset = 0 }))
  | `Up | `Down -> (
      (* Simple implementation - just move to previous/next block *)
      let target_id =
        if direction = `Up then Document.previous_block doc pos.block_id
        else Document.next_block doc pos.block_id
      in
      match target_id with
      | None -> None
      | Some id -> Some { block_id = id; offset = min pos.offset 0 })

let find_word_boundaries text offset =
  let is_word_char c =
    (c >= 'a' && c <= 'z')
    || (c >= 'A' && c <= 'Z')
    || (c >= '0' && c <= '9')
    || c = '_'
  in

  let start =
    let rec aux i =
      if i <= 0 then 0
      else if not (is_word_char text.[i]) then i + 1
      else aux (i - 1)
    in
    aux (min offset (String.length text - 1))
  in

  let end_pos =
    let rec aux i =
      if i >= String.length text then String.length text
      else if not (is_word_char text.[i]) then i
      else aux (i + 1)
    in
    aux offset
  in

  (start, end_pos)
