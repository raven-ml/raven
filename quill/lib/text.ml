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
