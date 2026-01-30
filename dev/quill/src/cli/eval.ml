(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

open Quill_editor.Document

(* Evaluate code blocks in markdown *)
let eval_code_block ?(print_all = false) code =
  try
    let result = Quill_top_unix.eval ~print_all code in
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

let rec process_block block =
  match block.content with
  | Codeblock { code; info; _ } -> (
      (* Check if we should print all values based on info string *)
      let print_all =
        match info with
        | Some info_str ->
            String.contains info_str 'v' || String.contains info_str 'V'
        | None -> false
      in
      match eval_code_block ~print_all code with
      | Ok output ->
          let output = unescape_output output in
          let formatted_output = Quill_top.format_output output in
          let output_block =
            Quill_editor.Document.html_block formatted_output
          in
          {
            block with
            content = Codeblock { code; output = Some output_block; info };
          }
      | Error err ->
          let error_block =
            Quill_editor.Document.html_block ("```\n" ^ err ^ "\n```")
          in
          {
            block with
            content = Codeblock { code; output = Some error_block; info };
          })
  | Block_quote blocks ->
      { block with content = Block_quote (List.map process_block blocks) }
  | Blocks blocks ->
      { block with content = Blocks (List.map process_block blocks) }
  | List (list_type, spacing, items) ->
      let processed_items = List.map (List.map process_block) items in
      { block with content = List (list_type, spacing, processed_items) }
  | _ -> block

let eval_markdown md =
  let blocks = Quill_editor.Document.of_markdown md in
  let evaluated_blocks = List.map process_block blocks in
  Quill_editor.Document.to_markdown evaluated_blocks

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
