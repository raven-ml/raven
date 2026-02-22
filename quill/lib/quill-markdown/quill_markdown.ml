(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(* ───── Parsing ───── *)

let is_blank s =
  let n = String.length s in
  let rec loop i =
    if i >= n then true
    else
      match s.[i] with ' ' | '\t' | '\n' | '\r' -> loop (i + 1) | _ -> false
  in
  loop 0

let trim_blank_lines s =
  let n = String.length s in
  if n = 0 then s
  else
    let i = ref 0 in
    while !i < n && (s.[!i] = '\n' || s.[!i] = '\r') do
      incr i
    done;
    let j = ref (n - 1) in
    while !j >= !i && (s.[!j] = '\n' || s.[!j] = '\r') do
      decr j
    done;
    if !i > !j then "" else String.sub s !i (!j - !i + 1)

let code_block_range b =
  match b with
  | Cmarkit.Block.Code_block (cb, meta) ->
      let language =
        match Cmarkit.Block.Code_block.info_string cb with
        | Some (info, _) -> (
            match Cmarkit.Block.Code_block.language_of_info_string info with
            | Some (lang, _) -> lang
            | None -> "")
        | None -> ""
      in
      let code_lines = Cmarkit.Block.Code_block.code cb in
      let code =
        String.concat "\n" (List.map (fun (line, _) -> line) code_lines)
      in
      let loc = Cmarkit.Meta.textloc meta in
      let first = Cmarkit.Textloc.first_byte loc in
      let last = Cmarkit.Textloc.last_byte loc in
      Some (first, last, language, code)
  | _ -> None

let cell_id_open = "<!-- quill:cell id=\""
let cell_id_close = "\" -->"
let output_open = "<!-- quill:output -->"
let output_close = "<!-- /quill:output -->"

let find_substring haystack needle start =
  let nlen = String.length needle in
  let hlen = String.length haystack in
  if nlen = 0 then Some start
  else
    let limit = hlen - nlen in
    let rec loop i =
      if i > limit then None
      else if String.sub haystack i nlen = needle then Some i
      else loop (i + 1)
    in
    loop start

let try_parse_cell_id s start =
  match find_substring s cell_id_open start with
  | Some open_pos -> (
      let id_start = open_pos + String.length cell_id_open in
      match find_substring s cell_id_close id_start with
      | Some close_pos ->
          let id = String.sub s id_start (close_pos - id_start) in
          let comment_end = close_pos + String.length cell_id_close in
          Some (id, open_pos, comment_end)
      | None -> None)
  | None -> None

let strip_leading_cell_id s =
  let s_trimmed = trim_blank_lines s in
  match try_parse_cell_id s_trimmed 0 with
  | Some (id, 0, comment_end) ->
      let rest =
        if comment_end < String.length s_trimmed then
          String.sub s_trimmed comment_end
            (String.length s_trimmed - comment_end)
        else ""
      in
      Some (id, trim_blank_lines rest)
  | _ -> None

let strip_trailing_cell_id s =
  let s_trimmed = trim_blank_lines s in
  let len = String.length s_trimmed in
  let marker_len = String.length cell_id_open + String.length cell_id_close in
  if len < marker_len then None
  else
    (* Scan backwards for the last newline to find the last line *)
    let last_line_start =
      let rec loop i =
        if i < 0 then 0 else if s_trimmed.[i] = '\n' then i + 1 else loop (i - 1)
      in
      loop (len - 1)
    in
    let last_line =
      String.sub s_trimmed last_line_start (len - last_line_start)
    in
    match try_parse_cell_id last_line 0 with
    | Some (id, 0, comment_end) when comment_end = String.length last_line ->
        let rest =
          if last_line_start > 0 then String.sub s_trimmed 0 last_line_start
          else ""
        in
        Some (id, trim_blank_lines rest)
    | _ -> None

let out_marker_prefix = "<!-- out:"
let out_marker_suffix = " -->"

let parse_output_sections content =
  let lines = String.split_on_char '\n' content in
  let flush_section tag buf acc =
    let text = Buffer.contents buf in
    Buffer.clear buf;
    if is_blank text then acc
    else
      let trimmed =
        let n = String.length text in
        let j = ref (n - 1) in
        while !j >= 0 && text.[!j] = '\n' do
          decr j
        done;
        if !j < 0 then "" else String.sub text 0 (!j + 1)
      in
      let output =
        match tag with
        | "stdout" -> Quill.Cell.Stdout trimmed
        | "stderr" -> Quill.Cell.Stderr trimmed
        | "error" -> Quill.Cell.Error trimmed
        | display_tag ->
            (* "display MIME" *)
            let prefix = "display " in
            let plen = String.length prefix in
            if
              String.length display_tag > plen
              && String.sub display_tag 0 plen = prefix
            then
              let mime =
                String.sub display_tag plen (String.length display_tag - plen)
              in
              Quill.Cell.Display { mime; data = trimmed }
            else (* Unknown tag, treat as stdout *)
              Quill.Cell.Stdout trimmed
      in
      output :: acc
  in
  let has_markers =
    List.exists
      (fun l ->
        let t = String.trim l in
        String.length t > String.length out_marker_prefix
        && String.sub t 0 (String.length out_marker_prefix) = out_marker_prefix)
      lines
  in
  if not has_markers then
    (* Backward compat: no markers, treat entire content as Stdout *)
    if is_blank content then []
    else
      let trimmed =
        let n = String.length content in
        if n > 0 && content.[n - 1] = '\n' then String.sub content 0 (n - 1)
        else content
      in
      [ Quill.Cell.Stdout trimmed ]
  else
    let buf = Buffer.create 256 in
    let tag = ref "" in
    let acc = ref [] in
    List.iter
      (fun line ->
        let trimmed = String.trim line in
        let plen = String.length out_marker_prefix in
        let slen = String.length out_marker_suffix in
        if
          String.length trimmed > plen + slen
          && String.sub trimmed 0 plen = out_marker_prefix
          && String.sub trimmed (String.length trimmed - slen) slen
             = out_marker_suffix
        then (
          (* Flush previous section *)
          if !tag <> "" then acc := flush_section !tag buf !acc;
          (* Extract tag name *)
          let tag_str =
            String.sub trimmed plen (String.length trimmed - plen - slen)
          in
          tag := tag_str)
        else (
          Buffer.add_string buf line;
          Buffer.add_char buf '\n'))
      lines;
    (* Flush last section *)
    if !tag <> "" then acc := flush_section !tag buf !acc;
    List.rev !acc

let parse_outputs md ~after =
  let len = String.length md in
  let pos = ref after in
  while !pos < len && (md.[!pos] = '\n' || md.[!pos] = '\r') do
    incr pos
  done;
  match find_substring md output_open !pos with
  | Some open_pos when open_pos = !pos -> (
      let content_start = open_pos + String.length output_open in
      let content_start =
        if content_start < len && md.[content_start] = '\n' then
          content_start + 1
        else content_start
      in
      match find_substring md output_close content_start with
      | Some close_pos ->
          let content =
            String.sub md content_start (close_pos - content_start)
          in
          let outputs = parse_output_sections content in
          let end_pos = close_pos + String.length output_close in
          Some (outputs, end_pos)
      | None -> None)
  | _ -> None

let of_string md =
  let doc = Cmarkit.Doc.of_string ~locs:true md in
  let top_blocks =
    match Cmarkit.Doc.block doc with
    | Cmarkit.Block.Blocks (bs, _) -> bs
    | b -> [ b ]
  in
  (* Collect code block byte ranges *)
  let code_ranges = List.filter_map code_block_range top_blocks in
  (* Build cells by slicing the original text at code block boundaries *)
  let cells = ref [] in
  let cursor = ref 0 in
  List.iter
    (fun (first, last, lang, code) ->
      (* Text between previous position and this code block *)
      let code_id = ref None in
      (if !cursor < first then
         let gap = String.sub md !cursor (first - !cursor) in
         (* Extract trailing cell ID for the code block *)
         let gap =
           match strip_trailing_cell_id gap with
           | Some (id, rest) ->
               code_id := Some id;
               rest
           | None -> gap
         in
         (* Extract leading cell ID for the text cell *)
         let text_id, gap =
           match strip_leading_cell_id gap with
           | Some (id, rest) -> (Some id, rest)
           | None -> (None, gap)
         in
         let gap = trim_blank_lines gap in
         if not (is_blank gap) then
           cells := Quill.Cell.text ?id:text_id gap :: !cells);
      (* The code block itself *)
      let cell = Quill.Cell.code ?id:!code_id ~language:lang code in
      (* Check for output markers immediately after the code block *)
      let cell, end_pos =
        match parse_outputs md ~after:(last + 1) with
        | Some (outputs, end_pos) ->
            (Quill.Cell.set_outputs outputs cell, end_pos)
        | None -> (cell, last + 1)
      in
      cells := cell :: !cells;
      cursor := end_pos)
    code_ranges;
  (* Remaining text after last code block *)
  (if !cursor < String.length md then
     let remaining = String.sub md !cursor (String.length md - !cursor) in
     let text_id, remaining =
       match strip_leading_cell_id remaining with
       | Some (id, rest) -> (Some id, rest)
       | None -> (None, remaining)
     in
     let remaining = trim_blank_lines remaining in
     if not (is_blank remaining) then
       cells := Quill.Cell.text ?id:text_id remaining :: !cells);
  Quill.Doc.of_cells (List.rev !cells)

(* ───── Rendering ───── *)

let add_content buf s =
  Buffer.add_string buf s;
  if s <> "" && s.[String.length s - 1] <> '\n' then Buffer.add_char buf '\n'

let render_output buf = function
  | Quill.Cell.Stdout s ->
      Buffer.add_string buf "<!-- out:stdout -->\n";
      add_content buf s
  | Quill.Cell.Stderr s ->
      Buffer.add_string buf "<!-- out:stderr -->\n";
      add_content buf s
  | Quill.Cell.Error s ->
      Buffer.add_string buf "<!-- out:error -->\n";
      add_content buf s
  | Quill.Cell.Display { mime; data } ->
      Buffer.add_string buf "<!-- out:display ";
      Buffer.add_string buf mime;
      Buffer.add_string buf " -->\n";
      add_content buf data

let render_cell_id buf id =
  Buffer.add_string buf cell_id_open;
  Buffer.add_string buf id;
  Buffer.add_string buf cell_id_close;
  Buffer.add_char buf '\n'

let render_cell ~with_outputs buf = function
  | Quill.Cell.Text { id; source; _ } ->
      render_cell_id buf id;
      Buffer.add_string buf source;
      Buffer.add_char buf '\n'
  | Quill.Cell.Code { id; source; language; outputs; _ } ->
      render_cell_id buf id;
      Buffer.add_string buf "```";
      Buffer.add_string buf language;
      Buffer.add_char buf '\n';
      Buffer.add_string buf source;
      Buffer.add_char buf '\n';
      Buffer.add_string buf "```";
      if with_outputs && outputs <> [] then (
        Buffer.add_char buf '\n';
        Buffer.add_string buf "<!-- quill:output -->\n";
        List.iter (render_output buf) outputs;
        Buffer.add_string buf "<!-- /quill:output -->")

let render ~with_outputs doc =
  let buf = Buffer.create 4096 in
  let cells = Quill.Doc.cells doc in
  let rec loop = function
    | [] -> ()
    | [ c ] -> render_cell ~with_outputs buf c
    | c :: rest ->
        render_cell ~with_outputs buf c;
        Buffer.add_char buf '\n';
        Buffer.add_char buf '\n';
        loop rest
  in
  loop cells;
  let s = Buffer.contents buf in
  (* Ensure file ends with a newline *)
  if s <> "" && s.[String.length s - 1] <> '\n' then s ^ "\n" else s

let to_string doc = render ~with_outputs:false doc
let to_string_with_outputs doc = render ~with_outputs:true doc

module Edit = Edit
