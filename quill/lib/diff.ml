type change = Keep of int | Insert of int * string | Delete of int * int

let compute_changes old_text new_text =
  (* Simple implementation - can be enhanced with proper diff algorithm *)
  if old_text = new_text then [ Keep (String.length old_text) ]
  else
    let old_len = String.length old_text in
    let new_len = String.length new_text in

    (* Find common prefix *)
    let rec find_prefix i =
      if i >= old_len || i >= new_len then i
      else if old_text.[i] = new_text.[i] then find_prefix (i + 1)
      else i
    in
    let prefix_len = find_prefix 0 in

    (* Find common suffix *)
    let rec find_suffix i =
      if i >= old_len - prefix_len || i >= new_len - prefix_len then i
      else if old_text.[old_len - 1 - i] = new_text.[new_len - 1 - i] then
        find_suffix (i + 1)
      else i
    in
    let suffix_len = find_suffix 0 in

    let changes = ref [] in

    if prefix_len > 0 then changes := Keep prefix_len :: !changes;

    let old_middle_len = old_len - prefix_len - suffix_len in
    let new_middle_len = new_len - prefix_len - suffix_len in

    if old_middle_len > 0 then
      changes := Delete (prefix_len, prefix_len + old_middle_len) :: !changes;

    (if new_middle_len > 0 then
       let middle_text = String.sub new_text prefix_len new_middle_len in
       changes := Insert (prefix_len, middle_text) :: !changes);

    if suffix_len > 0 then changes := Keep suffix_len :: !changes;

    List.rev !changes

let changes_to_operations block_id changes =
  List.filter_map
    (function
      | Keep _ -> None
      | Insert (offset, text) ->
          Some (Command.Insert_text (block_id, offset, text))
      | Delete (start_offset, end_offset) ->
          Some (Command.Delete_range (block_id, start_offset, end_offset)))
    changes
