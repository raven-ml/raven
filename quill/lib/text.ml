let insert_at text offset str =
  let before = String.sub text 0 offset in
  let after = String.sub text offset (String.length text - offset) in
  before ^ str ^ after

let delete_range text start_offset end_offset =
  let before = String.sub text 0 start_offset in
  let after = String.sub text end_offset (String.length text - end_offset) in
  before ^ after

let split_at text offset =
  let before = String.sub text 0 offset in
  let after = String.sub text offset (String.length text - offset) in
  (before, after)

let is_word_char c =
  match c with
  | 'a'..'z' | 'A'..'Z' | '0'..'9' | '_' -> true
  | _ -> false

let find_word_boundaries text offset =
  let len = String.length text in
  let offset = max 0 (min offset len) in
  
  (* Find start of word *)
  let rec find_start i =
    if i <= 0 then 0
    else if not (is_word_char text.[i - 1]) then i
    else find_start (i - 1)
  in
  
  (* Find end of word *)
  let rec find_end i =
    if i >= len then len
    else if not (is_word_char text.[i]) then i
    else find_end (i + 1)
  in
  
  (* Handle case where we're not on a word character *)
  if offset < len && not (is_word_char text.[offset]) then
    (* Find next word *)
    let rec find_next i =
      if i >= len then (len, len)
      else if is_word_char text.[i] then
        (i, find_end i)
      else find_next (i + 1)
    in
    find_next offset
  else
    (find_start offset, find_end offset)
