(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

type execution_result = {
  output : string; (* Captured stdout + value results *)
  error : string option; (* Captured stderr (errors and warnings) *)
  status : [ `Success | `Error ];
}

let initialize_toplevel () =
  Sys.interactive := false;
  Toploop.initialize_toplevel_env ();
  Toploop.input_name := "//toplevel//";
  Sys.interactive := true

let ensure_terminator code =
  let trimmed_code = String.trim code in
  if trimmed_code = "" || String.ends_with ~suffix:";;" trimmed_code then code
  else code ^ ";;"

let execute printval pp_out pp_err s =
  let s = ensure_terminator s in

  (* Use from_string to preserve source for error reporting *)
  let lb = Lexing.from_string s in
  (* Set up location info for better error reporting *)
  lb.lex_curr_p <-
    { pos_fname = "//toplevel//"; pos_lnum = 1; pos_bol = 0; pos_cnum = 0 };

  let overall_success = ref true in

  let old_warnings_formatter = !Location.formatter_for_warnings in
  Location.formatter_for_warnings := pp_err;

  (* Store original source for error context *)
  let orig_input_lexbuf = !Location.input_lexbuf in
  Location.input_lexbuf := Some lb;

  (* First collect all phrases *)
  let phrases = ref [] in
  (try
     while true do
       let phr = !Toploop.parse_toplevel_phrase lb in
       phrases := phr :: !phrases
     done
   with End_of_file -> ());

  let phrases = List.rev !phrases in
  let num_phrases = List.length phrases in

  Fun.protect
    (fun () ->
      List.iteri
        (fun i phr ->
          try
            let is_last = i = num_phrases - 1 in
            (* Always print the last value, even in non-verbose mode *)
            let should_print = printval || is_last in
            let exec_success = Toploop.execute_phrase should_print pp_out phr in
            overall_success := !overall_success && exec_success
          with
          | Sys.Break ->
              overall_success := false;
              Format.fprintf pp_err "Interrupted.@."
          | x ->
              overall_success := false;
              Errors.report_error pp_err x)
        phrases)
    ~finally:(fun () ->
      Location.formatter_for_warnings := old_warnings_formatter;
      Location.input_lexbuf := orig_input_lexbuf;
      Format.pp_print_flush pp_out ();
      Format.pp_print_flush pp_err ());

  !overall_success

let format_output output =
  let output = String.trim output in
  if output = "" then output
  else
    (* Split output into lines and check for mixed content *)
    let lines = String.split_on_char '\n' output in

    (* Check if this is a Hugin figure output by looking for the pattern *)
    let is_figure_output =
      List.exists
        (fun line ->
          let trimmed = String.trim line in
          String.starts_with ~prefix:"- : Hugin.Figure.t" trimmed
          || String.starts_with ~prefix:"- : Hugin.figure" trimmed)
        lines
      && List.exists
           (fun line ->
             let trimmed = String.trim line in
             String.starts_with ~prefix:"![figure]" trimmed)
           lines
    in

    if is_figure_output then
      (* Extract only the image markdown, skip the type signature *)
      let rec extract_images = function
        | [] -> []
        | line :: rest ->
            let trimmed = String.trim line in
            if String.starts_with ~prefix:"![" trimmed then
              line :: extract_images rest
            else extract_images rest
      in
      String.concat "\n" (extract_images lines)
    else
      (* Original logic for non-figure outputs *)
      let rec split_content acc_code acc_markdown = function
        | [] -> (List.rev acc_code, List.rev acc_markdown)
        | line :: rest ->
            let trimmed = String.trim line in
            if
              String.starts_with ~prefix:"![" trimmed
              || String.starts_with ~prefix:"<img" trimmed
              || String.starts_with ~prefix:"<svg" trimmed
              || String.starts_with ~prefix:"<figure" trimmed
              || String.starts_with ~prefix:"</figure" trimmed
            then split_content acc_code (line :: acc_markdown) rest
            else if trimmed = "" then
              (* Skip empty lines between sections *)
              split_content acc_code acc_markdown rest
            else split_content (line :: acc_code) acc_markdown rest
      in

      let code_lines, markdown_lines = split_content [] [] lines in

      match (code_lines, markdown_lines) with
      | [], [] -> ""
      | [], md_lines ->
          (* Pure markdown output *)
          String.concat "\n" md_lines
      | code_lines, [] ->
          (* Pure code output *)
          "```\n" ^ String.concat "\n" code_lines ^ "\n```"
      | code_lines, md_lines ->
          (* Mixed content: code block followed by markdown *)
          "```\n"
          ^ String.concat "\n" code_lines
          ^ "\n```\n"
          ^ String.concat "\n" md_lines
