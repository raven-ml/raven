open Model

type msg =
  | Focus_inline_by_id of int
  | Focus_block_by_id of int
  | Set_document of block list
  | Replace_block_codeblock of int
  | Split_block of int * int * int
  | Update_codeblock of int * string
  | Code_execution_finished of int * Quill_api.code_execution_result

let log fmt =
  Printf.ksprintf (fun s -> Brr.Console.(log [ Jstr.v ("[update] " ^ s) ])) fmt

let rec update_codeblock (block : block) (target_id : int) (new_code : string) :
    block =
  if block.id = target_id then
    match block.content with
    | Codeblock { output; _ } ->
        { block with content = Codeblock { code = new_code; output } }
    | _ -> block
  else
    match block.content with
    | Blocks bs ->
        {
          block with
          content =
            Blocks
              (List.map (fun b -> update_codeblock b target_id new_code) bs);
        }
    | _ -> block

let focus_inline_by_id document inline_id =
  set_focused_document_by_id (List.map clear_focus_block document) inline_id

let focus_block_by_id document block_id =
  document |> List.map clear_focus_block
  |> List.map (fun b ->
         if b.id = block_id then { b with focused = true } else b)

let update (m : model) (message : msg) : model =
  match message with
  | Focus_inline_by_id inline_id ->
      let new_document = focus_inline_by_id m.document inline_id in
      { document = new_document }
  | Focus_block_by_id block_id ->
      let new_document = focus_block_by_id m.document block_id in
      { document = new_document }
  | Set_document docs -> { document = docs }
  | Replace_block_codeblock block_id ->
      let new_document =
        List.map
          (fun b ->
            if b.id = block_id then
              {
                id = block_id;
                content = Codeblock { code = ""; output = None };
                focused = false;
              }
            else b)
          m.document
      in
      { document = new_document }
  | Update_codeblock (block_id, new_code) ->
      let new_document =
        List.map (fun b -> update_codeblock b block_id new_code) m.document
      in
      { document = new_document }
  | Code_execution_finished (block_id, result) ->
      log "Received code execution result for block %d" block_id;
      let output_text =
        match (result.error, result.status) with
        | Some err, `Error ->
            log "Execution error for block %d: %s" block_id err;
            "Error: " ^ err
        | None, `Error ->
            log "Unknown execution error for block %d" block_id;
            "Unknown error"
        | _, `Success ->
            log "Execution success for block %d" block_id;
            result.output
      in
      let output_block = Model.block_of_md output_text in
      let new_document =
        List.map
          (fun b -> set_codeblock_output_in_block b block_id output_block)
          m.document
      in
      { document = new_document }
  | Split_block (block_id, run_id, offset) ->
      let new_document =
        List.map
          (fun b ->
            if b.id = block_id then (
              match Model.find_inline_in_block b run_id with
              | None ->
                  log "No inline content with id %d found in block %d" run_id
                    block_id;
                  [ b ] (* No split if no inline content *)
              | Some inline ->
                  log "Splitting inline content with id %d in block %d" run_id
                    block_id;
                  let before, after = Model.split_inline inline offset in
                  let new_block1 =
                    Model.replace_inline_in_block b run_id before
                  in
                  let new_block2 =
                    Model.replace_inline_in_block b run_id after
                  in
                  [ new_block1; new_block2 ])
            else [ b ])
          m.document
        |> List.flatten
      in
      { document = new_document }
