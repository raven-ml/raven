type block_status = [ `Idle | `Running ]

type selection_status =
  [ `None | `Caret | `Range_single | `Range_start | `Range_end | `Range_middle ]

type block = {
  id : int;
  focused : bool;
  status : block_status;
  selection : selection_status;
  content : Document.block_content;
}

type document = block list

type diff =
  | Added of block
  | Removed of block
  | Updated of { before : block; after : block }

let classify_selection state idx block_id =
  let open State in
  match state.selection with
  | No_selection -> `None
  | Caret caret -> if caret.block_id = block_id then `Caret else `None
  | Range { anchor; focus } -> (
      match
        ( Document.index_of_block state.document ~block_id:anchor.block_id,
          Document.index_of_block state.document ~block_id:focus.block_id )
      with
      | Some anchor_idx, Some focus_idx ->
          let start_idx = min anchor_idx focus_idx in
          let end_idx = max anchor_idx focus_idx in
          if idx < start_idx || idx > end_idx then `None
          else if start_idx = end_idx then `Range_single
          else if idx = start_idx then `Range_start
          else if idx = end_idx then `Range_end
          else `Range_middle
      | _ -> `None)

let of_state (state : State.t) : document =
  let open State in
  state.document
  |> List.mapi (fun idx (blk : Document.block) ->
         let status =
           if State.is_block_running state blk.id then `Running else `Idle
         in
         let selection = classify_selection state idx blk.id in
         {
           id = blk.id;
           focused = blk.focused;
           status;
           selection;
           content = blk.content;
         })

module Int_map = Map.Make (Int)

let to_map (doc : document) =
  List.fold_left (fun acc blk -> Int_map.add blk.id blk acc) Int_map.empty doc

let diff before after =
  let before_map = to_map before in
  let after_map = to_map after in
  let changes =
    Int_map.fold
      (fun id before_blk acc ->
        match Int_map.find_opt id after_map with
        | None -> Removed before_blk :: acc
        | Some after_blk ->
            if before_blk = after_blk then acc
            else Updated { before = before_blk; after = after_blk } :: acc)
      before_map []
  in
  Int_map.fold
    (fun id after_blk acc ->
      match Int_map.find_opt id before_map with
      | None -> Added after_blk :: acc
      | Some _ -> acc)
    after_map changes
  |> List.rev
