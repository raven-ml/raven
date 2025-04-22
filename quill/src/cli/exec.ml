open Cmarkit

let execute_code id code =
  let result = Quill_top.eval ~id code in
  let output = result.output in
  let error =
    match result.error with Some e -> "\nError: " ^ e | None -> ""
  in
  output ^ error

let process_block_list id blocks =
  let rec loop acc = function
    | [] -> List.rev acc
    | block :: rest -> (
        match block with
        | Block.Code_block (cb, _meta) ->
            let info_opt = Block.Code_block.info_string cb in
            let language_opt =
              info_opt |> Option.map fst
              |> (fun x ->
              Option.bind x Block.Code_block.language_of_info_string)
              |> Option.map fst
            in
            if language_opt = Some "ocaml" then
              let code =
                Block.Code_block.code cb
                |> List.map Block_line.to_string
                |> String.concat "\n"
              in
              let output = execute_code id code in
              let output_lines = Block_line.list_of_string output in
              let output_cb =
                let info_string = ("output", Meta.none) in
                Block.Code_block.make ~info_string output_lines
              in
              let output_block = Block.Code_block (output_cb, Meta.none) in
              match rest with
              | Block.Code_block (cb', _) :: rest' ->
                  let info_opt = Block.Code_block.info_string cb' in
                  let language_opt =
                    info_opt |> Option.map fst
                    |> (fun x ->
                    Option.bind x Block.Code_block.language_of_info_string)
                    |> Option.map fst
                  in
                  if language_opt = Some "output" then
                    (* Replace the existing output block *)
                    loop (output_block :: block :: acc) rest'
                  else loop (output_block :: block :: acc) rest
              | _ ->
                  (* Insert a new output block after the code block *)
                  loop (output_block :: block :: acc) rest
            else loop (block :: acc) rest
        | _ -> loop (block :: acc) rest)
  in
  loop [] blocks

let process_md id md =
  let doc = Doc.of_string md in
  let block = Doc.block doc in
  match Block.normalize block with
  | Block.Blocks (blocks, meta) ->
      let processed_blocks = process_block_list id blocks in
      let new_block = Block.Blocks (processed_blocks, meta) in
      let new_doc = Doc.make new_block in
      Cmarkit_commonmark.of_doc new_doc
  | _ -> failwith "Unexpected document structure"

let exec file =
  if not (Sys.file_exists file) then (
    Printf.eprintf "Error: File '%s' does not exist.\n" file;
    exit 1);
  if not (String.ends_with ~suffix:".md" file) then (
    Printf.eprintf "Error: File '%s' must be a Markdown (.md) file.\n" file;
    exit 1);
  let id = "exec" in
  Quill_top.initialize_toplevel id;
  let _ = Quill_top.eval ~id {|
#require "ndarray";;
#require "hugin";;
|} in
  let original_md = Utils.read_text_from_file file in
  let new_md = process_md id original_md in
  if new_md <> original_md then (
    Utils.write_text_to_file file new_md;
    Printf.printf "Execution completed for '%s'.\n" file)
  else Printf.printf "No changes for '%s'.\n" file
