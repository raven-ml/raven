(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

let buf_add_escaped buf s =
  String.iter
    (function
      | '&' -> Buffer.add_string buf "&amp;"
      | '<' -> Buffer.add_string buf "&lt;"
      | '>' -> Buffer.add_string buf "&gt;"
      | '"' -> Buffer.add_string buf "&quot;"
      | c -> Buffer.add_char buf c)
    s

let escape_html s =
  let buf = Buffer.create (String.length s) in
  buf_add_escaped buf s;
  Buffer.contents buf

(* ───── Server-side OCaml syntax highlighting ───── *)

let ocaml_keywords =
  let tbl = Hashtbl.create 64 in
  List.iter
    (fun k -> Hashtbl.replace tbl k true)
    [
      "let";
      "in";
      "match";
      "with";
      "if";
      "then";
      "else";
      "fun";
      "function";
      "type";
      "module";
      "struct";
      "sig";
      "end";
      "open";
      "val";
      "rec";
      "and";
      "of";
      "begin";
      "for";
      "do";
      "done";
      "while";
      "to";
      "downto";
      "try";
      "exception";
      "raise";
      "when";
      "as";
      "mutable";
      "include";
      "external";
      "class";
      "object";
      "method";
      "inherit";
      "virtual";
      "private";
      "constraint";
      "assert";
      "lazy";
      "true";
      "false";
    ];
  tbl

let is_ident_start c =
  (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c = '_'

let is_ident_char c = is_ident_start c || (c >= '0' && c <= '9') || c = '\''
let is_digit c = c >= '0' && c <= '9'

let is_operator_char c =
  match c with
  | '!' | '$' | '%' | '&' | '*' | '+' | '-' | '.' | '/' | ':' | '<' | '=' | '>'
  | '?' | '@' | '^' | '|' | '~' ->
      true
  | _ -> false

let highlight_ocaml buf source =
  let len = String.length source in
  let i = ref 0 in
  let span cls text =
    Buffer.add_string buf "<span class=\"hljs-";
    Buffer.add_string buf cls;
    Buffer.add_string buf "\">";
    buf_add_escaped buf text;
    Buffer.add_string buf "</span>"
  in
  while !i < len do
    let c = source.[!i] in
    if c = '(' && !i + 1 < len && source.[!i + 1] = '*' then begin
      (* Nested comment *)
      let start = !i in
      let depth = ref 1 in
      i := !i + 2;
      while !i < len && !depth > 0 do
        if !i + 1 < len && source.[!i] = '(' && source.[!i + 1] = '*' then (
          incr depth;
          i := !i + 2)
        else if !i + 1 < len && source.[!i] = '*' && source.[!i + 1] = ')' then (
          decr depth;
          i := !i + 2)
        else incr i
      done;
      span "comment" (String.sub source start (!i - start))
    end
    else if c = '"' then begin
      (* String literal *)
      let start = !i in
      incr i;
      while !i < len && source.[!i] <> '"' do
        if source.[!i] = '\\' && !i + 1 < len then i := !i + 2 else incr i
      done;
      if !i < len then incr i;
      span "string" (String.sub source start (!i - start))
    end
    else if c = '\'' && !i + 1 < len then begin
      (* Character literal or type variable — check for char literal patterns *)
      if !i + 2 < len && source.[!i + 1] <> '\'' && source.[!i + 2] = '\'' then begin
        (* 'x' *)
        span "string" (String.sub source !i 3);
        i := !i + 3
      end
      else if !i + 3 < len && source.[!i + 1] = '\\' && source.[!i + 3] = '\''
      then begin
        (* '\n' etc *)
        span "string" (String.sub source !i 4);
        i := !i + 4
      end
      else begin
        buf_add_escaped buf (String.make 1 c);
        incr i
      end
    end
    else if is_digit c then begin
      (* Number literal *)
      let start = !i in
      incr i;
      (* Handle 0x, 0o, 0b prefixes *)
      if
        c = '0' && !i < len
        && (source.[!i] = 'x'
           || source.[!i] = 'X'
           || source.[!i] = 'o'
           || source.[!i] = 'O'
           || source.[!i] = 'b'
           || source.[!i] = 'B')
      then incr i;
      while
        !i < len
        && (is_digit source.[!i]
           || source.[!i] = '_'
           || source.[!i] = '.'
           || (source.[!i] >= 'a' && source.[!i] <= 'f')
           || (source.[!i] >= 'A' && source.[!i] <= 'F')
           || source.[!i] = 'e'
           || source.[!i] = 'E')
      do
        incr i
      done;
      span "number" (String.sub source start (!i - start))
    end
    else if is_ident_start c then begin
      (* Identifier or keyword *)
      let start = !i in
      incr i;
      while !i < len && is_ident_char source.[!i] do
        incr i
      done;
      let word = String.sub source start (!i - start) in
      if Hashtbl.mem ocaml_keywords word then span "keyword" word
      else if word.[0] >= 'A' && word.[0] <= 'Z' then span "type" word
      else buf_add_escaped buf word
    end
    else if c = ';' && !i + 1 < len && source.[!i + 1] = ';' then begin
      span "operator" ";;";
      i := !i + 2
    end
    else if c = '-' && !i + 1 < len && source.[!i + 1] = '>' then begin
      span "operator" "->";
      i := !i + 2
    end
    else if c = '|' && !i + 1 < len && source.[!i + 1] = '>' then begin
      span "operator" "|>";
      i := !i + 2
    end
    else if c = '<' && !i + 1 < len && source.[!i + 1] = '-' then begin
      span "operator" "<-";
      i := !i + 2
    end
    else if is_operator_char c then begin
      let start = !i in
      incr i;
      while !i < len && is_operator_char source.[!i] do
        incr i
      done;
      span "operator" (String.sub source start (!i - start))
    end
    else begin
      buf_add_escaped buf (String.make 1 c);
      incr i
    end
  done

(* ───── Markdown to HTML with heading anchors ───── *)

let markdown_to_html source =
  let doc = Cmarkit.Doc.of_string ~strict:false ~heading_auto_ids:true source in
  let module C = Cmarkit_renderer.Context in
  let heading_block c = function
    | Cmarkit.Block.Heading (h, _) ->
        let level = string_of_int (Cmarkit.Block.Heading.level h) in
        C.string c "<h";
        C.string c level;
        (match Cmarkit.Block.Heading.id h with
        | None -> C.byte c '>'
        | Some (`Auto id | `Id id) ->
            C.string c " id=\"";
            Cmarkit_html.html_escaped_string c id;
            C.string c "\">");
        C.inline c (Cmarkit.Block.Heading.inline h);
        (match Cmarkit.Block.Heading.id h with
        | None -> ()
        | Some (`Auto id | `Id id) ->
            C.string c " <a class=\"heading-anchor\" href=\"#";
            Cmarkit_html.pct_encoded_string c id;
            C.string c "\">#</a>");
        C.string c "</h";
        C.string c level;
        C.string c ">\n";
        true
    | _ -> false
  in
  let custom = Cmarkit_renderer.make ~block:heading_block () in
  let default = Cmarkit_html.renderer ~safe:true () in
  let r = Cmarkit_renderer.compose default custom in
  Cmarkit_renderer.doc_to_string r doc

(* ───── Chapter rendering ───── *)

let render_output buf (output : Quill.Cell.output) =
  match output with
  | Stdout s ->
      Buffer.add_string buf {|<pre class="output">|};
      buf_add_escaped buf s;
      Buffer.add_string buf "</pre>\n"
  | Stderr s ->
      Buffer.add_string buf {|<pre class="output stderr">|};
      buf_add_escaped buf s;
      Buffer.add_string buf "</pre>\n"
  | Error s ->
      Buffer.add_string buf {|<pre class="output error">|};
      buf_add_escaped buf s;
      Buffer.add_string buf "</pre>\n"
  | Display { mime; data } ->
      if String.length mime >= 6 && String.sub mime 0 6 = "image/" then (
        Buffer.add_string buf {|<pre class="output">|};
        Buffer.add_string buf {|<img src="data:|};
        Buffer.add_string buf mime;
        Buffer.add_string buf ";base64,";
        Buffer.add_string buf data;
        Buffer.add_string buf {|">|};
        Buffer.add_string buf "</pre>\n")
      else if mime = "text/html" then (
        Buffer.add_string buf {|<div class="output">|};
        Buffer.add_string buf data;
        Buffer.add_string buf "</div>\n")
      else (
        Buffer.add_string buf {|<pre class="output">|};
        buf_add_escaped buf data;
        Buffer.add_string buf "</pre>\n")

let render_code_cell buf ~language ~source ~outputs ~(attrs : Quill.Cell.attrs)
    =
  let collapsed = attrs.collapsed in
  let hide_source = attrs.hide_source in
  if collapsed then (
    Buffer.add_string buf {|<details class="collapsed"><summary>Code</summary>|};
    Buffer.add_string buf "\n");
  Buffer.add_string buf {|<div class="code-cell">|};
  Buffer.add_string buf "\n";
  if not hide_source then (
    Buffer.add_string buf "<pre><code class=\"language-";
    Buffer.add_string buf (escape_html language);
    Buffer.add_string buf "\">";
    if language = "ocaml" || language = "ml" then highlight_ocaml buf source
    else buf_add_escaped buf source;
    Buffer.add_string buf "</code></pre>\n";
    (* Copy button *)
    Buffer.add_string buf {|<button class="copy-btn" aria-label="Copy code">|};
    Buffer.add_string buf
      {|<svg class="copy-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>|};
    Buffer.add_string buf
      {|<svg class="check-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>|};
    Buffer.add_string buf "</button>\n");
  List.iter (render_output buf) outputs;
  Buffer.add_string buf "</div>\n";
  if collapsed then Buffer.add_string buf "</details>\n"

let chapter_html (doc : Quill.Doc.t) =
  let buf = Buffer.create 4096 in
  List.iter
    (fun cell ->
      match cell with
      | Quill.Cell.Text { source; _ } ->
          Buffer.add_string buf (markdown_to_html source)
      | Quill.Cell.Code { language; source; outputs; attrs; _ } ->
          render_code_cell buf ~language ~source ~outputs ~attrs)
    (Quill.Doc.cells doc);
  Buffer.contents buf

(* ───── TOC rendering ───── *)

let notebook_output_path (nb : Quill_project.notebook) =
  (* chapters/01-intro/chapter.md → chapters/01-intro/index.html *)
  Filename.concat (Filename.dirname nb.path) "index.html"

let rec render_toc_items buf ~toc ~(current : Quill_project.notebook) ~root_path
    ~depth items =
  List.iter
    (fun item ->
      match item with
      | Quill_project.Section title ->
          Buffer.add_string buf {|<div class="toc-part">|};
          buf_add_escaped buf title;
          Buffer.add_string buf "</div>\n"
      | Quill_project.Notebook (nb, children) ->
          let number_prefix =
            match Quill_project.number_string (Quill_project.number toc nb) with
            | "" -> ""
            | s -> s ^ ". "
          in
          (if Quill_project.is_placeholder nb then (
             Buffer.add_string buf
               (Printf.sprintf {|<span class="toc-chapter draft depth-%d">|}
                  depth);
             buf_add_escaped buf (number_prefix ^ nb.title);
             Buffer.add_string buf "</span>\n")
           else
             let active = if nb.path = current.path then " active" else "" in
             Buffer.add_string buf
               (Printf.sprintf {|<a class="toc-chapter%s depth-%d" href="|}
                  active depth);
             Buffer.add_string buf
               (escape_html (root_path ^ notebook_output_path nb));
             Buffer.add_string buf {|">|};
             buf_add_escaped buf (number_prefix ^ nb.title);
             Buffer.add_string buf "</a>\n");
          if children <> [] then
            render_toc_items buf ~toc ~current ~root_path ~depth:(depth + 1)
              children
      | Quill_project.Separator ->
          Buffer.add_string buf {|<div class="toc-separator"></div>|};
          Buffer.add_string buf "\n")
    items

let toc_html (project : Quill_project.t) ~(current : Quill_project.notebook)
    ~root_path =
  let buf = Buffer.create 1024 in
  render_toc_items buf ~toc:project.toc ~current ~root_path ~depth:0 project.toc;
  Buffer.contents buf

(* ───── HTML stripping ───── *)

let strip_html_tags s =
  let len = String.length s in
  let buf = Buffer.create len in
  let in_tag = ref false in
  for i = 0 to len - 1 do
    let c = s.[i] in
    if c = '<' then in_tag := true
    else if c = '>' then in_tag := false
    else if not !in_tag then Buffer.add_char buf c
  done;
  (* Collapse whitespace *)
  let raw = Buffer.contents buf in
  let rlen = String.length raw in
  let buf2 = Buffer.create rlen in
  let prev_space = ref true in
  for i = 0 to rlen - 1 do
    let c = raw.[i] in
    if c = ' ' || c = '\n' || c = '\r' || c = '\t' then (
      if not !prev_space then Buffer.add_char buf2 ' ';
      prev_space := true)
    else (
      Buffer.add_char buf2 c;
      prev_space := false)
  done;
  Buffer.contents buf2

(* ───── Page template ───── *)

let replace_all ~pattern ~with_ s =
  let plen = String.length pattern in
  if plen = 0 then s
  else
    let buf = Buffer.create (String.length s) in
    let slen = String.length s in
    let rec loop i =
      if i > slen - plen then (
        Buffer.add_substring buf s i (slen - i);
        Buffer.contents buf)
      else if String.sub s i plen = pattern then (
        Buffer.add_string buf with_;
        loop (i + plen))
      else (
        Buffer.add_char buf s.[i];
        loop (i + 1))
    in
    loop 0

let nav_html ~dir ~url ~title =
  let buf = Buffer.create 128 in
  Buffer.add_string buf {|<a href="|};
  Buffer.add_string buf (escape_html url);
  Buffer.add_string buf {|" class="nav-|};
  Buffer.add_string buf dir;
  Buffer.add_string buf {|"><span class="nav-dir">|};
  Buffer.add_string buf
    (if dir = "prev" then "&larr; Previous" else "Next &rarr;");
  Buffer.add_string buf {|</span><span class="nav-title">|};
  buf_add_escaped buf title;
  Buffer.add_string buf "</span></a>";
  Buffer.contents buf

(* ───── On-page TOC ───── *)

let extract_headings html =
  (* Extract h2 and h3 tags with their id and text content. Scans for <h2
     id="..."> or <h3 id="..."> patterns. *)
  let len = String.length html in
  let headings = ref [] in
  let i = ref 0 in
  while !i < len - 6 do
    if
      html.[!i] = '<'
      && html.[!i + 1] = 'h'
      && (html.[!i + 2] = '2' || html.[!i + 2] = '3')
    then begin
      let level = Char.code html.[!i + 2] - Char.code '0' in
      let tag_start = !i in
      (* Find the end of opening tag *)
      let tag_end = ref (!i + 3) in
      while !tag_end < len && html.[!tag_end] <> '>' do
        incr tag_end
      done;
      if !tag_end < len then begin
        let tag = String.sub html tag_start (!tag_end - tag_start + 1) in
        (* Extract id attribute *)
        let id_prefix = " id=\"" in
        let id_start =
          let rec find j =
            if j + String.length id_prefix > String.length tag then None
            else if String.sub tag j (String.length id_prefix) = id_prefix then
              Some (j + String.length id_prefix)
            else find (j + 1)
          in
          find 0
        in
        match id_start with
        | Some id_s ->
            let id_end = ref id_s in
            while !id_end < String.length tag && tag.[!id_end] <> '"' do
              incr id_end
            done;
            let id = String.sub tag id_s (!id_end - id_s) in
            (* Find closing tag </h2> or </h3> *)
            let close_tag = Printf.sprintf "</h%d>" level in
            let close_len = String.length close_tag in
            let body_start = !tag_end + 1 in
            let close_pos = ref body_start in
            while
              !close_pos + close_len <= len
              && String.sub html !close_pos close_len <> close_tag
            do
              incr close_pos
            done;
            if !close_pos + close_len <= len then begin
              let body = String.sub html body_start (!close_pos - body_start) in
              (* Strip HTML tags from body to get plain text *)
              let text = strip_html_tags body in
              headings := (level, id, text) :: !headings;
              i := !close_pos + close_len
            end
            else i := !tag_end + 1
        | None -> i := !tag_end + 1
      end
      else i := !i + 1
    end
    else incr i
  done;
  List.rev !headings

let page_toc_html headings =
  match headings with
  | [] -> ""
  | _ ->
      let buf = Buffer.create 512 in
      Buffer.add_string buf
        {|<nav class="page-toc"><div class="page-toc-title">On this page</div><ul>|};
      Buffer.add_char buf '\n';
      List.iter
        (fun (level, id, text) ->
          let cls = if level = 3 then {| class="toc-h3"|} else "" in
          Buffer.add_string buf
            (Printf.sprintf {|<li%s><a href="#%s">|} cls (escape_html id));
          buf_add_escaped buf text;
          Buffer.add_string buf "</a></li>\n")
        headings;
      Buffer.add_string buf "</ul></nav>\n";
      Buffer.contents buf

let edit_link_html edit_url =
  match edit_url with
  | None -> ""
  | Some url ->
      Printf.sprintf
        {|<div class="edit-link"><a href="%s">Edit this page</a></div>|}
        (escape_html url)

let page_html ~book_title ~chapter_title ~toc_html ~prev ~next ~root_path
    ~content ~edit_url ~live_reload_script =
  let prev_nav =
    match prev with
    | Some (url, title) -> nav_html ~dir:"prev" ~url ~title
    | None -> ""
  in
  let next_nav =
    match next with
    | Some (url, title) -> nav_html ~dir:"next" ~url ~title
    | None -> ""
  in
  let edit_link = edit_link_html edit_url in
  let page_toc =
    let headings = extract_headings content in
    page_toc_html headings
  in
  Theme.template_html
  |> replace_all ~pattern:"{{book_title}}" ~with_:(escape_html book_title)
  |> replace_all ~pattern:"{{chapter_title}}" ~with_:(escape_html chapter_title)
  |> replace_all ~pattern:"{{root_path}}" ~with_:root_path
  |> replace_all ~pattern:"{{toc}}" ~with_:toc_html
  |> replace_all ~pattern:"{{edit_link}}" ~with_:edit_link
  |> replace_all ~pattern:"{{content}}" ~with_:content
  |> replace_all ~pattern:"{{page_toc}}" ~with_:page_toc
  |> replace_all ~pattern:"{{prev_nav}}" ~with_:prev_nav
  |> replace_all ~pattern:"{{next_nav}}" ~with_:next_nav
  |> replace_all ~pattern:"{{live_reload_script}}" ~with_:live_reload_script

let print_page_html ~book_title ~chapters =
  let buf = Buffer.create 4096 in
  List.iter
    (fun (title, content) ->
      Buffer.add_string buf {|<article class="print-chapter">|};
      Buffer.add_string buf "\n<h1>";
      buf_add_escaped buf title;
      Buffer.add_string buf "</h1>\n";
      Buffer.add_string buf content;
      Buffer.add_string buf "\n</article>\n")
    chapters;
  let chapters_html = Buffer.contents buf in
  Theme.print_template_html
  |> replace_all ~pattern:"{{book_title}}" ~with_:(escape_html book_title)
  |> replace_all ~pattern:"{{chapters}}" ~with_:chapters_html
