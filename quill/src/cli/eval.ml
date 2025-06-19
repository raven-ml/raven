open Quill

(* Evaluate code blocks in markdown *)
let eval_code_block code =
  try
    let result = Quill_top_unix.eval code in
    match result.Quill_top.error with
    | Some err -> Error err
    | None -> Ok result.Quill_top.output
  with exn -> Error (Printexc.to_string exn)

let unescape_output output =
  (* Remove escape characters added by the formatter *)
  let output = Str.global_replace (Str.regexp "\\\\-") "-" output in
  let output = Str.global_replace (Str.regexp "\\\\\\[") "[" output in
  let output = Str.global_replace (Str.regexp "\\\\\\]") "]" output in
  output

let rec process_block (block : Document.block) : Document.block =
  match block.content with
  | Codeblock { code; language = _; output = _ } -> (
      match eval_code_block code with
      | Ok output ->
          let output = unescape_output output in
          let output = String.trim output in
          (* Create output blocks - use HTML block to avoid escaping *)
          let output_blocks = [
            Document.html_block ~id:(Document.block_id_of_int 0) output
          ] in
          {
            block with
            content = Codeblock { code; language = None; output = Some output_blocks };
          }
      | Error err ->
          let error_blocks = [
            Document.html_block ~id:(Document.block_id_of_int 0) ("Error: " ^ err)
          ] in
          { block with content = Codeblock { code; language = None; output = Some error_blocks } }
      )
  | Block_quote blocks ->
      { block with content = Block_quote (List.map process_block blocks) }
  | List (list_type, spacing, items) ->
      let processed_items = List.map (List.map process_block) items in
      { block with content = List (list_type, spacing, processed_items) }
  | _ -> block

let eval_markdown md =
  let document = Markdown.parse md in
  let blocks = Document.get_blocks document in
  let evaluated_blocks = List.map process_block blocks in
  let new_doc = { Document.blocks = evaluated_blocks } in
  Markdown.serialize new_doc

let eval_file path =
  try
    let ic = open_in path in
    let content = really_input_string ic (in_channel_length ic) in
    close_in ic;
    Ok (eval_markdown content)
  with
  | Sys_error msg -> Error ("Error reading file: " ^ msg)
  | exn -> Error ("Error: " ^ Printexc.to_string exn)

let eval_stdin () =
  try
    let lines = ref [] in
    (try
       while true do
         lines := input_line stdin :: !lines
       done
     with End_of_file -> ());
    let content = String.concat "\n" (List.rev !lines) in
    Ok (eval_markdown content)
  with exn -> Error ("Error reading stdin: " ^ Printexc.to_string exn)
