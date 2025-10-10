open Brr
open Quill_editor
open Quill_markdown

let log fmt =
  Printf.ksprintf (fun s -> Console.(log [ Jstr.v ("[main] " ^ s) ])) fmt

let app =
  Vdom.simple_app ~init:Model.init ~update:Update.update ~view:View.view ()

let code_execution_handler mounted_app block_id code =
  log "Handler: Executing code for block %d" block_id;
  let result_fut = Api.execute_code code in
  Fut.await result_fut (fun result ->
      (* Determine the result (Ok or Error) *)
      let api_result =
        match result with
        | Ok output -> output
        | Error err ->
            Quill_api.
              {
                output = "";
                error = Some (Api.string_of_api_error err);
                status = `Error;
              }
      in
      log
        "Handler: Future completed for block %d. Dispatching \
         Code_execution_finished."
        block_id;
      Vdom_blit.process mounted_app
        (Update.Code_execution_finished (block_id, api_result)))

let () =
  match Document.find_el_by_id G.document (Jstr.v "editor-app") with
  | None -> Console.(error [ Jstr.v "No #editor-app element found" ])
  | Some container_el ->
      let container_jv = El.to_jv container_el in
      let mounted_app = Vdom_blit.run ~container:(Obj.magic container_jv) app in

      let code_execution_handler = code_execution_handler mounted_app in
      Events.setup_event_listeners ~code_execution_handler container_el
        mounted_app;

      let path = Jstr.to_string (Window.location G.window |> Uri.path) in

      log "Fetching initial document for path: %s" path;
      let fut = Api.fetch_document path in
      Fut.await fut (function
        | Ok response_text ->
            let document = document_of_md response_text in
            Vdom_blit.process mounted_app (Update.Set_document document)
        | Error err ->
            Console.(
              error
                [
                  Jstr.v "Failed to fetch document: ";
                  Jstr.v (Api.string_of_api_error err);
                ]))
