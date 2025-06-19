type msg =
  | Execute_command of Quill.Command.t
  | Code_execution_finished of Quill.Document.block_id * Quill_api.code_execution_result
  | Set_document_markdown of string
  | Execute_block of Quill.Document.block_id
  | Execute_all

let log fmt =
  Printf.ksprintf (fun s -> Brr.Console.(log [ Jstr.v ("[update] " ^ s) ])) fmt

let update (m : Model.t) (message : msg) : Model.t * Quill.Effect.t list =
  match message with
  | Execute_command cmd ->
      let new_state, effects = Quill.Engine.execute m.engine cmd in
      ({ engine = new_state }, effects)
  | Code_execution_finished (block_id, result) ->
      log "Received code execution result for block %d" (block_id :> int);
      let execution_result = 
        {
          Quill.Execution.output = result.output;
          error = (match result.status with
            | `Error -> result.error
            | `Success -> None);
          timestamp = Brr.Performance.now_ms Brr.G.performance /. 1000.0;
        }
      in
      (match (result.error, result.status) with
      | Some err, `Error ->
          log "Execution error for block %d: %s" (block_id :> int) err
      | None, `Error ->
          log "Unknown execution error for block %d" (block_id :> int)
      | _, `Success ->
          log "Execution success for block %d" (block_id :> int));
      let cmd = Quill.Command.Set_execution_result (block_id, execution_result) in
      let new_state, effects = Quill.Engine.execute m.engine cmd in
      ({ engine = new_state }, effects)
  | Set_document_markdown markdown ->
      (* Parse markdown into Quill document *)
      let new_document = Quill.Markdown.parse markdown in
      let new_engine = Quill.Engine.make new_document in
      ({ engine = new_engine }, [])
  | Execute_block block_id ->
      let cmd = Quill.Command.Execute_block block_id in
      let new_state, effects = Quill.Engine.execute m.engine cmd in
      ({ engine = new_state }, effects)
  | Execute_all ->
      let cmd = Quill.Command.Execute_all in
      let new_state, effects = Quill.Engine.execute m.engine cmd in
      ({ engine = new_state }, effects)