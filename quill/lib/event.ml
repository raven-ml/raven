type t =
  | Text_inserted of Document.block_id * int * string
  | Text_deleted of Document.block_id * int * int
  | Block_inserted of int * Document.block
  | Block_removed of Document.block_id
  | Selection_changed of View.selection option
  | Focus_changed of Document.block_id option * Document.inline_id option
  | Execution_completed of Document.block_id * Execution.execution_result

let to_string = function
  | Text_inserted (id, offset, text) ->
      Printf.sprintf "Text_inserted(block=%d, offset=%d, text=%S)" id offset
        text
  | Text_deleted (id, start_offset, end_offset) ->
      Printf.sprintf "Text_deleted(block=%d, start=%d, end=%d)" id start_offset
        end_offset
  | Block_inserted (pos, block) ->
      Printf.sprintf "Block_inserted(pos=%d, block=%d)" pos block.id
  | Block_removed id -> Printf.sprintf "Block_removed(block=%d)" id
  | Selection_changed sel -> (
      match sel with
      | None -> "Selection_changed(None)"
      | Some s ->
          Printf.sprintf
            "Selection_changed(anchor=block%d:%d, focus=block%d:%d)"
            s.anchor.block_id s.anchor.offset s.focus.block_id s.focus.offset)
  | Focus_changed (block_id, inline_id) ->
      let block_str =
        match block_id with None -> "None" | Some id -> string_of_int id
      in
      let inline_str =
        match inline_id with None -> "None" | Some id -> string_of_int id
      in
      Printf.sprintf "Focus_changed(block=%s, inline=%s)" block_str inline_str
  | Execution_completed (id, result) ->
      Printf.sprintf "Execution_completed(block=%d, output=%S)" id result.output
