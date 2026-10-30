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
let is_image mime = String.length mime >= 6 && String.sub mime 0 6 = "image/"

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
    (fun i c -> t.(Char.code c) <- i)
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

let base64_alphabet =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

let base64_encode raw =
  let len = String.length raw in
  let out = Bytes.create ((len + 2) / 3 * 4) in
  let byte i = if i < len then Char.code (String.unsafe_get raw i) else 0 in
  let digit k = String.unsafe_get base64_alphabet k in
  let i = ref 0 and j = ref 0 in
  while !i < len do
    let b0 = byte !i and b1 = byte (!i + 1) and b2 = byte (!i + 2) in
    Bytes.unsafe_set out !j (digit (b0 lsr 2));
    Bytes.unsafe_set out (!j + 1) (digit (((b0 land 3) lsl 4) lor (b1 lsr 4)));
    Bytes.unsafe_set out (!j + 2)
      (if !i + 1 < len then digit (((b1 land 0xf) lsl 2) lor (b2 lsr 6))
       else '=');
    Bytes.unsafe_set out (!j + 3)
      (if !i + 2 < len then digit (b2 land 0x3f) else '=');
    i := !i + 3;
    j := !j + 4
  done;
  Bytes.unsafe_to_string out

(* An image output's <img> tag holds a data URI or, for an image stored as a
   file, a reference that [resolve ~cell_id src] turns into the base64 of the
   image. *)
let parse_image_display ~resolve ~cell_id mime content =
  match extract_img_src content with
  | None -> Quill.Cell.Display { mime; id = None; data = content }
  | Some src ->
      let data =
        match extract_data_uri_base64 src with
        | Some base64 -> base64
        | None -> resolve ~cell_id src
      in
      Quill.Cell.Display { mime; id = None; data }

let parse_output_sections ~resolve ~cell_id content =
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
                parse_image_display ~resolve ~cell_id mime trimmed
              else Quill.Cell.Display { mime; id = None; data = trimmed }
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

let parse_outputs ~resolve ~cell_id md ~after =
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
          let outputs = parse_output_sections ~resolve ~cell_id content in
          let end_pos = close_pos + String.length output_close in
          Some (outputs, end_pos)
      | None -> None)
  | _ -> None

let parse ~resolve md =
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
         let gap = trim_blank_lines gap in
         if not (is_blank gap) then cells := Quill.Cell.text gap :: !cells);
      (* The code block itself *)
      let cell =
        Quill.Cell.code ?id:!code_id ~attrs:!code_attrs ~language:lang code
      in
      (* Check for output markers immediately after the code block *)
      let cell, end_pos =
        match
          parse_outputs ~resolve ~cell_id:(Quill.Cell.id cell) md
            ~after:(last + 1)
        with
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
     let remaining = trim_blank_lines remaining in
     if not (is_blank remaining) then
       cells := Quill.Cell.text remaining :: !cells);
  Quill.Doc.of_cells (List.rev !cells)

(* ───── Rendering ───── *)

let add_content buf s =
  Buffer.add_string buf s;
  if s <> "" && s.[String.length s - 1] <> '\n' then Buffer.add_char buf '\n'

(* [image ~cell_id ~n ~mime ~data] is the [src] of the [<img>] tag that stores
   the [n]th image of the cell [cell_id]. *)
let render_output ~image ~cell_id ~img_counter buf = function
  | Quill.Cell.Stdout s ->
      Buffer.add_string buf "<!-- out:stdout -->\n";
      add_content buf s
  | Quill.Cell.Stderr s ->
      Buffer.add_string buf "<!-- out:stderr -->\n";
      add_content buf s
  | Quill.Cell.Error s ->
      Buffer.add_string buf "<!-- out:error -->\n";
      add_content buf s
  | Quill.Cell.Display { mime; data; _ } ->
      Buffer.add_string buf "<!-- out:display ";
      Buffer.add_string buf mime;
      Buffer.add_string buf " -->\n";
      if is_image mime then (
        incr img_counter;
        Buffer.add_string buf "<img src=\"";
        Buffer.add_string buf (image ~cell_id ~n:!img_counter ~mime ~data);
        Buffer.add_string buf "\">\n")
      else add_content buf data

let render_cell_id buf id (attrs : Quill.Cell.attrs) =
  Buffer.add_string buf cell_id_open;
  Buffer.add_string buf id;
  Buffer.add_char buf '"';
  if attrs.collapsed then Buffer.add_string buf " collapsed";
  if attrs.hide_source then Buffer.add_string buf " hide-source";
  Buffer.add_string buf " -->\n"

(* A backtick fence longer than the longest run of backticks that begins a line
   of [source] after its leading spaces or tabs, and at least three, so no line
   closes it, indented ones included (CommonMark 0.31.2, 4.5). *)
let fence source =
  let longest = ref 2 in
  List.iter
    (fun line ->
      let n = String.length line in
      let i = ref 0 in
      while !i < n && (line.[!i] = ' ' || line.[!i] = '\t') do
        incr i
      done;
      let start = !i in
      while !i < n && line.[!i] = '`' do
        incr i
      done;
      longest := max !longest (!i - start))
    (String.split_on_char '\n' source);
  String.make (!longest + 1) '`'

let render_cell ~image ~with_outputs buf = function
  | Quill.Cell.Text { source; _ } ->
      Buffer.add_string buf source;
      Buffer.add_char buf '\n'
  | Quill.Cell.Code { id; source; language; outputs; attrs; _ } ->
      let fence = fence source in
      render_cell_id buf id attrs;
      Buffer.add_string buf fence;
      Buffer.add_string buf language;
      Buffer.add_char buf '\n';
      Buffer.add_string buf source;
      Buffer.add_char buf '\n';
      Buffer.add_string buf fence;
      if with_outputs && outputs <> [] then (
        Buffer.add_char buf '\n';
        Buffer.add_string buf "<!-- quill:output -->\n";
        let img_counter = ref 0 in
        List.iter (render_output ~image ~cell_id:id ~img_counter buf) outputs;
        Buffer.add_string buf "<!-- /quill:output -->")

let render ~image ~with_outputs doc =
  let buf = Buffer.create 4096 in
  let rec loop = function
    | [] -> ()
    | [ c ] -> render_cell ~image ~with_outputs buf c
    | c :: rest ->
        render_cell ~image ~with_outputs buf c;
        Buffer.add_char buf '\n';
        Buffer.add_char buf '\n';
        loop rest
  in
  loop (Quill.Doc.cells doc);
  let s = Buffer.contents buf in
  (* Ensure file ends with a newline *)
  if s <> "" && s.[String.length s - 1] <> '\n' then s ^ "\n" else s

let data_uri ~cell_id:_ ~n:_ ~mime ~data = "data:" ^ mime ^ ";base64," ^ data
let to_string doc = render ~image:data_uri ~with_outputs:false doc
let to_string_with_outputs doc = render ~image:data_uri ~with_outputs:true doc

(* Files *)

type figures = Inline | Files

(* The first line of a notebook that stores its images as files. *)
let figures_marker = "<!-- quill:figures files -->"
let figures_root = "figures"

(* An error of [load] or [save], which they return. *)
exception Storage_error of string

let error fmt = Printf.ksprintf (fun s -> raise (Storage_error s)) fmt

let read_file ?(bin = false) path =
  let ic = if bin then open_in_bin path else open_in path in
  Fun.protect
    ~finally:(fun () -> close_in ic)
    (fun () -> really_input_string ic (in_channel_length ic))

(* A file's device, inode, size and modification time. *)
type stamp = int * int * int * float

let stamp_of (st : Unix.stats) = (st.st_dev, st.st_ino, st.st_size, st.st_mtime)

(* [sync_dir dir] puts on disk the entries made or renamed in [dir], so that
   they survive a power loss in the order they were made. Windows has no
   directory sync. *)
let sync_dir dir =
  if not Sys.win32 then
    let fd = Unix.openfile dir [ Unix.O_RDONLY; Unix.O_CLOEXEC ] 0 in
    Fun.protect ~finally:(fun () -> Unix.close fd) (fun () -> Unix.fsync fd)

let mkdir dir =
  Unix.mkdir dir 0o777;
  sync_dir (Filename.dirname dir)

(* [write_file path content] replaces the file [path] names, through a link too,
   by one holding [content], and is the stamp of the new file. The content goes
   to a fresh temporary file beside it, created exclusively under a name no
   other file has, written and synced through the descriptor that made it, then
   renamed over the file and its directory synced: a crash leaves the old file
   or the new one, and may leave the temporary file. A new file has the
   permissions the umask allows; a replaced one keeps its own. *)
let write_file path content =
  let target, existing =
    match Unix.realpath path with
    | target -> (target, true)
    | exception Unix.Unix_error (Unix.ENOENT, _, _) -> (path, false)
  in
  let perm =
    if not existing then None
    else (
      (try Unix.access target [ Unix.W_OK ]
       with Unix.Unix_error _ -> error "%s is read-only" path);
      Some (Unix.stat target).Unix.st_perm)
  in
  let dir = Filename.dirname target in
  let tmp, oc =
    Filename.open_temp_file ~mode:[ Open_binary ] ~perms:0o666 ~temp_dir:dir
      ("." ^ Filename.basename target ^ ".")
      ".tmp"
  in
  match
    Fun.protect
      ~finally:(fun () -> close_out_noerr oc)
      (fun () ->
        output_string oc content;
        flush oc;
        let fd = Unix.descr_of_out_channel oc in
        Option.iter (Unix.chmod tmp) perm;
        Unix.fsync fd;
        stamp_of (Unix.fstat fd))
  with
  | stamp ->
      Unix.rename tmp target;
      sync_dir dir;
      stamp
  | exception e ->
      (try Sys.remove tmp with Sys_error _ -> ());
      raise e

(* [real path] is the file [path] names, through links. A notebook's figures
   live beside that file, where a viewer of it looks for them. *)
let real path =
  match Unix.realpath path with
  | real -> real
  | exception Unix.Unix_error (Unix.ENOENT, _, _) -> path

let stamp path =
  match Unix.stat path with
  | st -> Some (stamp_of st)
  | exception Unix.Unix_error ((Unix.ENOENT | Unix.ENOTDIR), _, _) -> None

(* [kind path] follows links: a link is what it points to. *)
let kind path =
  match Unix.stat path with
  | st -> Some st.Unix.st_kind
  | exception Unix.Unix_error ((Unix.ENOENT | Unix.ENOTDIR), _, _) -> None

let is_file_name s =
  s <> "" && s <> "." && s <> ".."
  && (not (String.contains s '/'))
  && not (String.contains s '\\')

(* The images of [nb/x.md] live in [nb/figures/x/]. The stem names a directory,
   so a notebook named [..md], whose stem is [.], has none. *)
let stem path =
  let stem = Filename.remove_extension (Filename.basename path) in
  if is_file_name stem then stem
  else error "%s: no figures directory for this name" path

let figure_error path ~cell_id ~src reason =
  error "%s: cell %s: figure %s: %s" path cell_id src reason

(* [split_marker md] is [Some rest] if the first line of [md] is
   [figures_marker], where [rest] is [md] after that line. *)
let split_marker md =
  let first, rest =
    match String.index_opt md '\n' with
    | Some i ->
        (String.sub md 0 i, String.sub md (i + 1) (String.length md - i - 1))
    | None -> (md, "")
  in
  let first =
    if String.ends_with ~suffix:"\r" first then
      String.sub first 0 (String.length first - 1)
    else first
  in
  if first = figures_marker then Some rest else None

(* A reference is accepted only in the shape [save] writes it, so that a
   notebook cannot make [load] read, and a book publish, a path of its choice: a
   file name in [figures/<stem>/]. The file must be a regular file, so that a
   read cannot block on a FIFO or a device. *)
let read_figure path ~cell_id src =
  let real = real path in
  let stem = stem real in
  let prefix = figures_root ^ "/" ^ stem ^ "/" in
  let name =
    if String.starts_with ~prefix src then
      String.sub src (String.length prefix)
        (String.length src - String.length prefix)
    else ""
  in
  if not (is_file_name name) then
    figure_error path ~cell_id ~src ("not in " ^ prefix);
  let file =
    Filename.concat
      (Filename.concat
         (Filename.concat (Filename.dirname real) figures_root)
         stem)
      name
  in
  match kind file with
  | None -> figure_error path ~cell_id ~src "no such file"
  | Some Unix.S_REG -> (
      try base64_encode (read_file ~bin:true file)
      with Sys_error e -> figure_error path ~cell_id ~src e)
  | Some _ -> figure_error path ~cell_id ~src "not a regular file"

(* [catch f] is [Ok (f ())], or the error [f] raised as [Error]. *)
let catch f =
  match f () with
  | v -> Ok v
  | exception Storage_error e -> Error e
  | exception Sys_error e -> Error e
  | exception Unix.Unix_error (e, _, arg) ->
      Error (arg ^ ": " ^ Unix.error_message e)

let load ?(outputs = true) path =
  catch @@ fun () ->
  let md = read_file path in
  let figures, md =
    match split_marker md with Some md -> (Files, md) | None -> (Inline, md)
  in
  let resolve ~cell_id src =
    if outputs then read_figure path ~cell_id src else ""
  in
  let doc = parse ~resolve md in
  let doc = if outputs then doc else Quill.Doc.clear_all_outputs doc in
  (doc, figures)

let of_string md =
  if split_marker md <> None then
    invalid_arg
      "Quill_markdown.of_string: the notebook stores its figures as files, \
       which Quill_markdown.load reads";
  let resolve ~cell_id src =
    invalid_arg
      (Printf.sprintf
         "Quill_markdown.of_string: cell %s: figure %s is a file, which \
          Quill_markdown.load reads"
         cell_id src)
  in
  parse ~resolve md

let figure_ext mime =
  let ext = extension_of_mime mime in
  let safe = function
    | 'a' .. 'z' | 'A' .. 'Z' | '0' .. '9' | '+' | '-' | '.' | '_' -> true
    | _ -> false
  in
  if String.for_all safe ext then ext else "bin"

let real_dir path dir =
  match kind dir with
  | None | Some Unix.S_DIR -> ()
  | Some _ -> error "%s: %s is not a directory" path dir

(* [rmdir_empty dir] removes [dir] if it is an empty directory and no link. *)
let rmdir_empty dir =
  match Unix.lstat dir with
  | { Unix.st_kind = Unix.S_DIR; _ } when Sys.readdir dir = [||] ->
      Sys.rmdir dir
  | _ -> ()

(* The figures are written before the notebook that references them, and the
   files it no longer references are removed after, so that a failure leaves no
   reference to a missing file. *)
let save_files path doc =
  let real = real path in
  let stem = stem real in
  let dir = Filename.concat (Filename.dirname real) figures_root in
  let stem_dir = Filename.concat dir stem in
  real_dir path dir;
  real_dir path stem_dir;
  let stored = ref [] in
  let image ~cell_id ~n ~mime ~data =
    let ext = figure_ext mime in
    let name =
      if n = 1 then cell_id ^ "." ^ ext
      else cell_id ^ "-" ^ string_of_int n ^ "." ^ ext
    in
    let src = figures_root ^ "/" ^ stem ^ "/" ^ name in
    if not (is_file_name name) then
      figure_error path ~cell_id ~src "the cell id makes no file name";
    (match List.assoc_opt name !stored with
    | Some (other, _) ->
        figure_error path ~cell_id ~src ("also stored by cell " ^ other)
    | None -> ());
    stored := (name, (cell_id, data)) :: !stored;
    src
  in
  let md = figures_marker ^ "\n\n" ^ render ~image ~with_outputs:true doc in
  if !stored <> [] then (
    if kind dir = None then mkdir dir;
    if kind stem_dir = None then mkdir stem_dir);
  List.iter
    (fun (name, (_, data)) ->
      ignore (write_file (Filename.concat stem_dir name) (base64_decode data)))
    !stored;
  let stamp = write_file path md in
  if kind stem_dir = Some Unix.S_DIR then (
    Array.iter
      (fun name ->
        let file = Filename.concat stem_dir name in
        if (not (List.mem_assoc name !stored)) && kind file <> Some Unix.S_DIR
        then Sys.remove file)
      (Sys.readdir stem_dir);
    rmdir_empty stem_dir;
    rmdir_empty dir);
  stamp

let save figures path doc =
  catch @@ fun () ->
  match figures with
  | Inline -> write_file path (to_string_with_outputs doc)
  | Files -> save_files path doc

module Edit = Edit
