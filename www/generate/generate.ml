let site_dir = "site"
let docs_dir = "docs"
let build_dir = "build"
let templates_dir = "templates"
let lib_doc_dir lib_name = Filename.concat (Filename.concat ".." lib_name) "doc"

let lib_examples_dir lib_name =
  Filename.concat (Filename.concat ".." lib_name) "examples"

type library = {
  name : string;
  display : string;
  color : string;
  description : string;
}

let libraries =
  [
    {
      name = "nx";
      display = "nx";
      color = "color-blue";
      description = "N-dimensional arrays with linear algebra operations";
    };
    {
      name = "rune";
      display = "rune";
      color = "color-orange";
      description = "Automatic differentiation and JIT compilation";
    };
    {
      name = "kaun";
      display = {|kaun <span class="rune-symbol">ᚲ</span>|};
      color = "color-red";
      description = "High-level neural network library";
    };
    {
      name = "hugin";
      display = "hugin";
      color = "color-purple";
      description = "Publication-quality 2D and 3D plotting library";
    };
    {
      name = "saga";
      display = "saga";
      color = "color-teal";
      description = "Modern text tokenization and processing for NLP";
    };
    {
      name = "talon";
      display = "talon";
      color = "color-pink";
      description = "Fast and elegant dataframes with type-safe operations";
    };
    {
      name = "quill";
      display = "quill";
      color = "color-green";
      description = "Interactive notebook environment";
    };
    {
      name = "fehu";
      display = {|fehu <span class="rune-symbol">ᚠ</span>|};
      color = "color-teal";
      description = "Reinforcement learning for OCaml";
    };
    {
      name = "sowilo";
      display = {|sowilo <span class="rune-symbol">ᛋ</span>|};
      color = "color-indigo";
      description = "Differentiable computer vision library";
    };
  ]

let find_library name = List.find_opt (fun lib -> lib.name = name) libraries

(* -- String utilities -- *)

let replace pattern replacement s =
  let plen = String.length pattern in
  let slen = String.length s in
  if plen = 0 then s
  else
    let buf = Buffer.create slen in
    let i = ref 0 in
    while !i <= slen - plen do
      if String.sub s !i plen = pattern then (
        Buffer.add_string buf replacement;
        i := !i + plen)
      else (
        Buffer.add_char buf (String.unsafe_get s !i);
        incr i)
    done;
    while !i < slen do
      Buffer.add_char buf (String.unsafe_get s !i);
      incr i
    done;
    Buffer.contents buf

let find_sub ?(start = 0) s sub =
  let slen = String.length s in
  let sublen = String.length sub in
  if sublen = 0 || sublen > slen then None
  else
    let rec loop i =
      if i > slen - sublen then None
      else if String.sub s i sublen = sub then Some i
      else loop (i + 1)
    in
    loop start

let strip_tags s =
  let buf = Buffer.create (String.length s) in
  let in_tag = ref false in
  String.iter
    (fun c ->
      if c = '<' then in_tag := true
      else if c = '>' then in_tag := false
      else if not !in_tag then Buffer.add_char buf c)
    s;
  Buffer.contents buf

let strip_order_prefix s =
  let len = String.length s in
  if
    len >= 3
    && s.[0] >= '0'
    && s.[0] <= '9'
    && s.[1] >= '0'
    && s.[1] <= '9'
    && s.[2] = '-'
  then String.sub s 3 (len - 3)
  else s

let title_case s =
  s |> String.split_on_char '-'
  |> List.map (fun w ->
      if w = "" then w
      else
        String.make 1 (Char.uppercase_ascii w.[0])
        ^ String.sub w 1 (String.length w - 1))
  |> String.concat " "

(* -- File system -- *)

let read_file path = In_channel.with_open_bin path In_channel.input_all

let write_file path content =
  Out_channel.with_open_bin path (fun oc -> output_string oc content)

let rec ensure_dir path =
  if path <> "" && path <> "." && path <> "/" then
    if not (Sys.file_exists path) then (
      ensure_dir (Filename.dirname path);
      Unix.mkdir path 0o755)

let rec walk dir =
  Sys.readdir dir |> Array.to_list
  |> List.concat_map (fun entry ->
      let path = Filename.concat dir entry in
      if Sys.is_directory path then walk path else [ path ])

(* -- HTML utilities -- *)

let extract_h1 html =
  match find_sub html "<h1" with
  | None -> None
  | Some i -> (
      match String.index_from_opt html i '>' with
      | None -> None
      | Some j -> (
          match find_sub ~start:(j + 1) html "</h1>" with
          | None -> None
          | Some k -> Some (strip_tags (String.sub html (j + 1) (k - j - 1)))))

(* -- Paths and URLs -- *)

let strip_prefix ~prefix path =
  let plen = String.length prefix + 1 in
  String.sub path plen (String.length path - plen)

let url_segments path =
  let stem = Filename.chop_extension path in
  let parts = String.split_on_char '/' stem in
  match List.rev parts with "index" :: rest -> List.rev rest | _ -> parts

let dest_path path =
  let stem = Filename.chop_extension path in
  if Filename.basename stem = "index" then
    Filename.concat build_dir (stem ^ ".html")
  else Filename.concat build_dir (Filename.concat stem "index.html")

(* -- Breadcrumbs -- *)

let breadcrumb_sep = {|<span class="breadcrumb-separator">/</span>|}

let make_breadcrumbs segments title =
  match segments with
  | [] | [ _ ] -> ""
  | _ ->
      let ancestors = List.rev (List.tl (List.rev segments)) in
      let links =
        let rec go acc url_path = function
          | [] -> List.rev acc
          | seg :: rest ->
              let url_path = url_path ^ "/" ^ seg in
              let link =
                Printf.sprintf {|<a href="%s/" class="breadcrumb-link">%s</a>|}
                  url_path (title_case seg)
              in
              go (link :: acc) url_path rest
        in
        go [] "" ancestors
      in
      Printf.sprintf
        {|<div id="breadcrumbs" class="breadcrumbs">%s%s<span class="breadcrumb-current">%s</span></div>|}
        (String.concat breadcrumb_sep links)
        breadcrumb_sep title

(* -- Library navigation -- *)

let generate_lib_nav lib_name =
  let dir = lib_doc_dir lib_name in
  if not (Sys.file_exists dir && Sys.is_directory dir) then ""
  else
    let files = Sys.readdir dir |> Array.to_list |> List.sort String.compare in
    let entries =
      files
      |> List.filter_map (fun f ->
          if Filename.extension f = ".md" then
            let stem = Filename.chop_extension f in
            let slug = strip_order_prefix stem in
            let title =
              if slug = "index" then "Overview" else title_case slug
            in
            let url =
              if slug = "index" then Printf.sprintf "/docs/%s/" lib_name
              else Printf.sprintf "/docs/%s/%s/" lib_name slug
            in
            Some (stem, title, url)
          else None)
    in
    let entries =
      List.sort
        (fun (a, _, _) (b, _, _) ->
          match (a, b) with
          | "index", _ -> -1
          | _, "index" -> 1
          | _ -> String.compare a b)
        entries
    in
    entries
    |> List.map (fun (_, title, url) ->
        Printf.sprintf {|          <li><a href="%s">%s</a></li>|} url title)
    |> String.concat "\n"

let generate_lib_examples_nav lib_name =
  let dir = lib_examples_dir lib_name in
  if not (Sys.file_exists dir && Sys.is_directory dir) then ""
  else
    let entries =
      Sys.readdir dir |> Array.to_list
      |> List.filter (fun entry -> Sys.is_directory (Filename.concat dir entry))
      |> List.sort String.compare
      |> List.map (fun entry ->
          let slug = strip_order_prefix entry in
          let title = title_case slug in
          let url = Printf.sprintf "/docs/%s/examples/%s/" lib_name slug in
          (title, url))
    in
    match entries with
    | [] -> ""
    | _ ->
        let items =
          entries
          |> List.map (fun (title, url) ->
              Printf.sprintf {|          <li><a href="%s">%s</a></li>|} url
                title)
          |> String.concat "\n"
        in
        Printf.sprintf
          {|      <div class="nav-section">
        <div class="nav-title">Examples</div>
        <ul class="nav-links">
%s
        </ul>
      </div>|}
          items

(* -- Template -- *)

let select_template path =
  let parts = String.split_on_char '/' path in
  let name =
    match parts with
    | "docs" :: lib :: _ when find_library lib <> None -> "layout_docs_lib.html"
    | "docs" :: _ -> "layout_docs.html"
    | _ -> "main.html"
  in
  Filename.concat templates_dir name

let apply_template ~template ~title ~breadcrumbs ~content ~lib =
  let t =
    template |> replace "{{title}}" title
    |> replace "{{breadcrumbs}}" breadcrumbs
    |> replace "{{content}}" content
  in
  match lib with
  | None -> t
  | Some lib ->
      t
      |> replace "{{lib_name}}" lib.name
      |> replace "{{lib_display}}" lib.display
      |> replace "{{lib_color}}" lib.color
      |> replace "{{lib_description}}" lib.description
      |> replace "{{lib_nav}}" (generate_lib_nav lib.name)
      |> replace "{{lib_examples_nav}}" (generate_lib_examples_nav lib.name)

(* -- Processing -- *)

let render_markdown content =
  content
  |> Cmarkit.Doc.of_string ~heading_auto_ids:true ~strict:false
  |> Hilite_markdown.transform ~skip_unknown_languages:true
  |> Cmarkit_html.of_doc ~safe:false

let process_markdown path content =
  let html = render_markdown content in
  let h1 = extract_h1 html in
  let title = match h1 with Some t -> t ^ " - raven" | None -> "raven" in
  let page_title =
    match h1 with
    | Some t -> t
    | None -> title_case (Filename.chop_extension (Filename.basename path))
  in
  let breadcrumbs = make_breadcrumbs (url_segments path) page_title in
  let lib =
    match String.split_on_char '/' path with
    | "docs" :: lib_name :: _ -> find_library lib_name
    | _ -> None
  in
  let template = read_file (select_template path) in
  apply_template ~template ~title ~breadcrumbs ~content:html ~lib

let highlight_html_code_blocks html =
  let buf = Buffer.create (String.length html) in
  let len = String.length html in
  let i = ref 0 in
  let pre_open = {|<pre class="language-|} in
  let pre_open_len = String.length pre_open in
  let code_close = "</code></pre>" in
  let code_close_len = String.length code_close in
  while !i < len do
    match find_sub ~start:!i html pre_open with
    | None ->
        Buffer.add_string buf (String.sub html !i (len - !i));
        i := len
    | Some pre_start -> (
        Buffer.add_string buf (String.sub html !i (pre_start - !i));
        let lang_start = pre_start + pre_open_len in
        match String.index_from_opt html lang_start '"' with
        | None ->
            Buffer.add_char buf html.[!i];
            i := !i + 1
        | Some lang_end -> (
            let lang = String.sub html lang_start (lang_end - lang_start) in
            let code_tag = {|<code class="language-|} ^ lang ^ {|">|} in
            match find_sub ~start:lang_end html code_tag with
            | None ->
                Buffer.add_char buf html.[!i];
                i := !i + 1
            | Some code_tag_start -> (
                let content_start = code_tag_start + String.length code_tag in
                match find_sub ~start:content_start html code_close with
                | None ->
                    Buffer.add_char buf html.[!i];
                    i := !i + 1
                | Some content_end -> (
                    let raw =
                      String.sub html content_start (content_end - content_start)
                    in
                    let code =
                      raw |> replace "&amp;" "&" |> replace "&lt;" "<"
                      |> replace "&gt;" ">"
                    in
                    match Hilite.src_code_to_html ~lang code with
                    | Ok highlighted ->
                        Buffer.add_string buf highlighted;
                        i := content_end + code_close_len
                    | Error _ ->
                        Buffer.add_string buf
                          (String.sub html pre_start
                             (content_end + code_close_len - pre_start));
                        i := content_end + code_close_len))))
  done;
  Buffer.contents buf

let process_file ~path full_path =
  let ext = Filename.extension path in
  let out, content =
    match ext with
    | ".md" -> (dest_path path, process_markdown path (read_file full_path))
    | ".html" | ".htm" ->
        (dest_path path, highlight_html_code_blocks (read_file full_path))
    | _ -> (Filename.concat build_dir path, read_file full_path)
  in
  ensure_dir (Filename.dirname out);
  write_file out content

let escape_html s =
  let buf = Buffer.create (String.length s) in
  String.iter
    (fun c ->
      match c with
      | '&' -> Buffer.add_string buf "&amp;"
      | '<' -> Buffer.add_string buf "&lt;"
      | '>' -> Buffer.add_string buf "&gt;"
      | _ -> Buffer.add_char buf c)
    s;
  Buffer.contents buf

let process_example ~lib example_dir =
  let entry = Filename.basename example_dir in
  let slug = strip_order_prefix entry in
  let path = Printf.sprintf "docs/%s/examples/%s.md" lib.name slug in
  let readme_path = Filename.concat example_dir "README.md" in
  let prose_html =
    if Sys.file_exists readme_path then render_markdown (read_file readme_path)
    else Printf.sprintf "<h1>%s</h1>" (escape_html (title_case slug))
  in
  let ml_files =
    Sys.readdir example_dir |> Array.to_list
    |> List.filter (fun f -> Filename.extension f = ".ml")
    |> List.sort String.compare
  in
  let multi = List.length ml_files > 1 in
  let code_html =
    ml_files
    |> List.map (fun f ->
        let code = read_file (Filename.concat example_dir f) in
        let header =
          if multi then Printf.sprintf "<h3>%s</h3>\n" (escape_html f) else ""
        in
        let highlighted =
          match Hilite.src_code_to_html ~lang:"ocaml" code with
          | Ok html -> html
          | Error _ ->
              Printf.sprintf "<pre><code>%s</code></pre>" (escape_html code)
        in
        header ^ highlighted)
    |> String.concat "\n"
  in
  let html = prose_html ^ "\n" ^ code_html in
  let h1 = extract_h1 html in
  let title = match h1 with Some t -> t ^ " - raven" | None -> "raven" in
  let page_title = match h1 with Some t -> t | None -> title_case slug in
  let breadcrumbs = make_breadcrumbs (url_segments path) page_title in
  let template = read_file (select_template path) in
  let content =
    apply_template ~template ~title ~breadcrumbs ~content:html ~lib:(Some lib)
  in
  let out = dest_path path in
  ensure_dir (Filename.dirname out);
  write_file out content

let () =
  walk site_dir
  |> List.iter (fun p -> process_file ~path:(strip_prefix ~prefix:site_dir p) p);
  walk docs_dir |> List.iter (fun p -> process_file ~path:p p);
  libraries
  |> List.iter (fun lib ->
      let dir = lib_doc_dir lib.name in
      if Sys.file_exists dir && Sys.is_directory dir then
        walk dir
        |> List.iter (fun full_path ->
            let ext = Filename.extension full_path in
            let base = Filename.basename full_path in
            if base = "dune" || ext = ".mld" then ()
            else
              let rel = strip_prefix ~prefix:dir full_path in
              let rel_base = Filename.basename rel in
              let rel_dir = Filename.dirname rel in
              let clean_base =
                if Filename.extension rel_base = ".md" then
                  strip_order_prefix (Filename.chop_extension rel_base) ^ ".md"
                else rel_base
              in
              let clean_rel =
                if rel_dir = "." then clean_base
                else Filename.concat rel_dir clean_base
              in
              let path =
                Filename.concat (Filename.concat "docs" lib.name) clean_rel
              in
              process_file ~path full_path));
  libraries
  |> List.iter (fun lib ->
      let dir = lib_examples_dir lib.name in
      if Sys.file_exists dir && Sys.is_directory dir then
        Sys.readdir dir |> Array.to_list |> List.sort String.compare
        |> List.iter (fun entry ->
            let full = Filename.concat dir entry in
            if Sys.is_directory full then process_example ~lib full))
