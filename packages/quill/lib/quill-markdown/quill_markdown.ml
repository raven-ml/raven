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
let comment_close = "-->"
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

let parse_attrs_tokens s =
  let tokens = String.split_on_char ' ' s |> List.filter (fun t -> t <> "") in
  let rec loop (attrs : Quill.Cell.attrs) = function
    | [] -> attrs
    | "collapsed" :: rest -> loop { attrs with collapsed = true } rest
    | "hide-source" :: rest -> loop { attrs with hide_source = true } rest
    | _ :: rest -> loop attrs rest
  in
  loop Quill.Cell.default_attrs tokens

let quote = "\""

let try_parse_cell_id s start =
  match find_substring s cell_id_open start with
  | Some open_pos -> (
      let id_start = open_pos + String.length cell_id_open in
      match find_substring s quote id_start with
      | Some quote_pos -> (
          let id = String.sub s id_start (quote_pos - id_start) in
          match find_substring s comment_close (quote_pos + 1) with
          | Some close_pos ->
              let attrs_str =
                String.sub s (quote_pos + 1) (close_pos - quote_pos - 1)
              in
              let attrs = parse_attrs_tokens attrs_str in
              let comment_end = close_pos + String.length comment_close in
              Some (id, attrs, open_pos, comment_end)
          | None -> None)
      | None -> None)
  | None -> None

let strip_leading_cell_id s =
  let s_trimmed = trim_blank_lines s in
  match try_parse_cell_id s_trimmed 0 with
  | Some (id, attrs, 0, comment_end) ->
      let rest =
        if comment_end < String.length s_trimmed then
          String.sub s_trimmed comment_end
            (String.length s_trimmed - comment_end)
        else ""
      in
      Some (id, attrs, trim_blank_lines rest)
  | _ -> None

let strip_trailing_cell_id s =
  let s_trimmed = trim_blank_lines s in
  let len = String.length s_trimmed in
  (* Minimum length: cell_id_open + closing quote + space + comment_close *)
  let min_len = String.length cell_id_open + String.length "\" -->" in
  if len < min_len then None
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
    | Some (id, attrs, 0, comment_end)
      when comment_end = String.length last_line ->
        let rest =
          if last_line_start > 0 then String.sub s_trimmed 0 last_line_start
          else ""
        in
        Some (id, attrs, trim_blank_lines rest)
    | _ -> None

let out_marker_prefix = "<!-- out:"
let out_marker_suffix = " -->"

let is_image mime =
  String.length mime >= 6 && String.sub mime 0 6 = "image/"

let extension_of_mime mime =
  match mime with
  | "image/png" -> "png"
  | "image/jpeg" -> "jpg"
  | "image/gif" -> "gif"
  | "image/svg+xml" -> "svg"
  | "image/webp" -> "webp"
  | _ ->
      if is_image mime && String.length mime > 6 then
        String.sub mime 6 (String.length mime - 6)
      else "bin"

let base64_decode_table =
  let t = Array.make 256 (-1) in
  String.iteri
    (fun i c ->
      t.(Char.code c) <- i)
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  t

let base64_decode s =
  let len = String.length s in
  (* Count valid base64 characters *)
  let valid = ref 0 in
  for i = 0 to len - 1 do
    let c = Char.code (String.unsafe_get s i) in
    if base64_decode_table.(c) >= 0 then incr valid
  done;
  let out_len = !valid * 3 / 4 in
  let out = Bytes.create out_len in
  let j = ref 0 in
  let acc = ref 0 in
  let bits = ref 0 in
  for i = 0 to len - 1 do
    let c = Char.code (String.unsafe_get s i) in
    let v = base64_decode_table.(c) in
    if v >= 0 then begin
      acc := (!acc lsl 6) lor v;
      bits := !bits + 6;
      if !bits >= 8 then begin
        bits := !bits - 8;
        if !j < out_len then begin
          Bytes.unsafe_set out !j (Char.chr ((!acc lsr !bits) land 0xff));
          incr j
        end
      end
    end
  done;
  Bytes.sub_string out 0 !j

(* Extract src attribute value from an <img> tag *)
let extract_img_src s =
  let src_attr = "src=\"" in
  match find_substring s src_attr 0 with
  | None -> None
  | Some i ->
      let start = i + String.length src_attr in
      let rec find_quote j =
        if j >= String.length s then None
        else if s.[j] = '"' then Some (String.sub s start (j - start))
        else find_quote (j + 1)
      in
      find_quote start

(* Extract base64 data from a data URI: data:mime;base64,DATA *)
let extract_data_uri_base64 src =
  let prefix = "data:" in
  let marker = ";base64," in
  if
    String.length src > String.length prefix
    && String.sub src 0 (String.length prefix) = prefix
  then
    match find_substring src marker 0 with
    | Some i ->
        let data_start = i + String.length marker in
        Some (String.sub src data_start (String.length src - data_start))
    | None -> None
  else None

(* Parse image Display data from <img> tag content *)
let parse_image_display ?base_dir mime content =
  match extract_img_src content with
  | Some src -> begin
      match extract_data_uri_base64 src with
      | Some base64 ->
          (* Inline data URI: extract base64 directly *)
          Quill.Cell.Display { mime; data = base64 }
      | None -> begin
          (* File reference: read and base64-encode *)
          match base_dir with
          | Some dir ->
              let path = Filename.concat dir src in
              let ic = open_in_bin path in
              let raw =
                Fun.protect
                  ~finally:(fun () -> close_in ic)
                  (fun () -> really_input_string ic (in_channel_length ic))
              in
              let data =
                (* Reuse the base64_encode from Hugin's image_util convention *)
                let alphabet =
                  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\
                   0123456789+/"
                in
                let len = String.length raw in
                let out_len = (len + 2) / 3 * 4 in
                let out = Bytes.create out_len in
                let rec loop i j =
                  if i < len then begin
                    let b0 = Char.code (String.unsafe_get raw i) in
                    let b1 =
                      if i + 1 < len then
                        Char.code (String.unsafe_get raw (i + 1))
                      else 0
                    in
                    let b2 =
                      if i + 2 < len then
                        Char.code (String.unsafe_get raw (i + 2))
                      else 0
                    in
                    Bytes.unsafe_set out j
                      (String.unsafe_get alphabet (b0 lsr 2));
                    Bytes.unsafe_set out (j + 1)
                      (String.unsafe_get alphabet
                         (((b0 land 3) lsl 4) lor (b1 lsr 4)));
                    Bytes.unsafe_set out (j + 2)
                      (if i + 1 < len then
                         String.unsafe_get alphabet
                           (((b1 land 0xf) lsl 2) lor (b2 lsr 6))
                       else '=');
                    Bytes.unsafe_set out (j + 3)
                      (if i + 2 < len then
                         String.unsafe_get alphabet (b2 land 0x3f)
                       else '=');
                    loop (i + 3) (j + 4)
                  end
                in
                loop 0 0;
                Bytes.unsafe_to_string out
              in
              Quill.Cell.Display { mime; data }
          | None ->
              (* No base_dir, store src as placeholder *)
              Quill.Cell.Display { mime; data = "" }
        end
    end
  | None ->
      (* No <img> tag — treat as raw data *)
      Quill.Cell.Display { mime; data = content }

let parse_output_sections ?base_dir content =
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
              if is_image mime then
                parse_image_display ?base_dir mime trimmed
              else Quill.Cell.Display { mime; data = trimmed }
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

let parse_outputs ?base_dir md ~after =
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
          let outputs = parse_output_sections ?base_dir content in
          let end_pos = close_pos + String.length output_close in
          Some (outputs, end_pos)
      | None -> None)
  | _ -> None

let of_string ?base_dir md =
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
      let code_attrs = ref Quill.Cell.default_attrs in
      (if !cursor < first then
         let gap = String.sub md !cursor (first - !cursor) in
         (* Extract trailing cell ID for the code block *)
         let gap =
           match strip_trailing_cell_id gap with
           | Some (id, attrs, rest) ->
               code_id := Some id;
               code_attrs := attrs;
               rest
           | None -> gap
         in
         (* Extract leading cell ID for the text cell *)
         let text_id, text_attrs, gap =
           match strip_leading_cell_id gap with
           | Some (id, attrs, rest) -> (Some id, Some attrs, rest)
           | None -> (None, None, gap)
         in
         let gap = trim_blank_lines gap in
         if not (is_blank gap) then
           cells := Quill.Cell.text ?id:text_id ?attrs:text_attrs gap :: !cells);
      (* The code block itself *)
      let cell =
        Quill.Cell.code ?id:!code_id ~attrs:!code_attrs ~language:lang code
      in
      (* Check for output markers immediately after the code block *)
      let cell, end_pos =
        match parse_outputs ?base_dir md ~after:(last + 1) with
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
     let text_id, text_attrs, remaining =
       match strip_leading_cell_id remaining with
       | Some (id, attrs, rest) -> (Some id, Some attrs, rest)
       | None -> (None, None, remaining)
     in
     let remaining = trim_blank_lines remaining in
     if not (is_blank remaining) then
       cells :=
         Quill.Cell.text ?id:text_id ?attrs:text_attrs remaining :: !cells);
  Quill.Doc.of_cells (List.rev !cells)

(* ───── Rendering ───── *)

let add_content buf s =
  Buffer.add_string buf s;
  if s <> "" && s.[String.length s - 1] <> '\n' then Buffer.add_char buf '\n'

let rec mkdir_p dir =
  if Sys.file_exists dir then ()
  else (
    mkdir_p (Filename.dirname dir);
    try Unix.mkdir dir 0o755 with Unix.Unix_error (Unix.EEXIST, _, _) -> ())

let write_figure_file ~path ~data =
  mkdir_p (Filename.dirname path);
  let raw = base64_decode data in
  let oc = open_out_bin path in
  Fun.protect
    ~finally:(fun () -> close_out oc)
    (fun () -> output_string oc raw)

(* Extract cell ID prefix from a figure filename like "c_abc123.png" or
   "c_abc123-2.png" *)
let cell_id_of_figure_name name =
  let base = Filename.remove_extension name in
  (* Strip trailing -N suffix *)
  match String.rindex_opt base '-' with
  | Some i ->
      let suffix = String.sub base (i + 1) (String.length base - i - 1) in
      let all_digits =
        String.length suffix > 0
        && String.to_seq suffix
           |> Seq.for_all (fun c -> c >= '0' && c <= '9')
      in
      if all_digits then String.sub base 0 i else base
  | None -> base

let clean_orphan_figures ~figures_dir ~cell_ids =
  if Sys.file_exists figures_dir && Sys.is_directory figures_dir then
    let entries = Sys.readdir figures_dir in
    Array.iter
      (fun name ->
        let cid = cell_id_of_figure_name name in
        if not (List.mem cid cell_ids) then
          Sys.remove (Filename.concat figures_dir name))
      entries

let render_output ?figures_dir ~cell_id ~img_counter buf = function
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
      if is_image mime then begin
        let ext = extension_of_mime mime in
        match figures_dir with
        | Some dir ->
            (* Disk mode: write file, reference by path *)
            incr img_counter;
            let basename =
              if !img_counter = 1 then cell_id ^ "." ^ ext
              else cell_id ^ "-" ^ string_of_int !img_counter ^ "." ^ ext
            in
            let path = Filename.concat dir basename in
            write_figure_file ~path ~data;
            Buffer.add_string buf "<img src=\"figures/";
            Buffer.add_string buf basename;
            Buffer.add_string buf "\">\n"
        | None ->
            (* Inline mode (default): data URI in <img> tag *)
            Buffer.add_string buf "<img src=\"data:";
            Buffer.add_string buf mime;
            Buffer.add_string buf ";base64,";
            Buffer.add_string buf data;
            Buffer.add_string buf "\">\n"
      end
      else if mime = "text/html" then add_content buf data
      else add_content buf data

let render_cell_id buf id (attrs : Quill.Cell.attrs) =
  Buffer.add_string buf cell_id_open;
  Buffer.add_string buf id;
  Buffer.add_char buf '"';
  if attrs.collapsed then Buffer.add_string buf " collapsed";
  if attrs.hide_source then Buffer.add_string buf " hide-source";
  Buffer.add_string buf " -->\n"

let render_cell ?figures_dir ~with_outputs buf = function
  | Quill.Cell.Text { source; _ } ->
      Buffer.add_string buf source;
      Buffer.add_char buf '\n'
  | Quill.Cell.Code { id; source; language; outputs; attrs; _ } ->
      render_cell_id buf id attrs;
      Buffer.add_string buf "```";
      Buffer.add_string buf language;
      Buffer.add_char buf '\n';
      Buffer.add_string buf source;
      Buffer.add_char buf '\n';
      Buffer.add_string buf "```";
      if with_outputs && outputs <> [] then (
        Buffer.add_char buf '\n';
        Buffer.add_string buf "<!-- quill:output -->\n";
        let img_counter = ref 0 in
        List.iter
          (render_output ?figures_dir ~cell_id:id ~img_counter buf)
          outputs;
        Buffer.add_string buf "<!-- /quill:output -->")

let render ?figures_dir ~with_outputs doc =
  let buf = Buffer.create 4096 in
  let cells = Quill.Doc.cells doc in
  (* Clean orphaned figures before writing new ones *)
  (match figures_dir with
  | Some dir ->
      let cell_ids =
        List.filter_map
          (function
            | Quill.Cell.Code { id; _ } -> Some id
            | Quill.Cell.Text _ -> None)
          cells
      in
      clean_orphan_figures ~figures_dir:dir ~cell_ids
  | None -> ());
  let rec loop = function
    | [] -> ()
    | [ c ] -> render_cell ?figures_dir ~with_outputs buf c
    | c :: rest ->
        render_cell ?figures_dir ~with_outputs buf c;
        Buffer.add_char buf '\n';
        Buffer.add_char buf '\n';
        loop rest
  in
  loop cells;
  let s = Buffer.contents buf in
  (* Ensure file ends with a newline *)
  if s <> "" && s.[String.length s - 1] <> '\n' then s ^ "\n" else s

let to_string doc = render ~with_outputs:false doc
let to_string_with_outputs ?figures_dir doc =
  render ?figures_dir ~with_outputs:true doc

module Edit = Edit
