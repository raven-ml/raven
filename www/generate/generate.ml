let site_dir = "site"
let docs_dir = "../doc"
let build_dir = "build"
let templates_dir = "templates"

let lib_doc_dir lib_name =
  Filename.concat
    (Filename.concat (Filename.concat ".." "packages") lib_name)
    "doc"

let lib_examples_dir lib_name =
  Filename.concat
    (Filename.concat (Filename.concat ".." "packages") lib_name)
    "examples"

(* -- Library navigation -- *)

let lib_doc_entries lib_name =
  let dir = lib_doc_dir lib_name in
  if not (Sys.file_exists dir && Sys.is_directory dir) then []
  else
    let files = Sys.readdir dir |> Array.to_list |> List.sort String.compare in
    let entries =
      files
      |> List.filter_map (fun f ->
          if Filename.extension f = ".md" then
            let stem = Filename.chop_extension f in
            let slug = Site.strip_order_prefix stem in
            let title =
              if slug = "index" then "Overview" else Site.title_case slug
            in
            let url =
              if slug = "index" then Printf.sprintf "/docs/%s/" lib_name
              else Printf.sprintf "/docs/%s/%s/" lib_name slug
            in
            Some (stem, title, url)
          else None)
    in
    List.sort
      (fun (a, _, _) (b, _, _) ->
        match (a, b) with
        | "index", _ -> -1
        | _, "index" -> 1
        | _ -> String.compare a b)
      entries

let generate_lib_nav ~current_url lib_name =
  lib_doc_entries lib_name
  |> List.map (fun (_, title, url) ->
      let active = if url = current_url then {| class="active"|} else "" in
      Printf.sprintf {|          <li><a href="%s"%s>%s</a></li>|} url active
        title)
  |> String.concat "\n"

let generate_prev_next ~current_url lib_name =
  let entries = lib_doc_entries lib_name in
  let arr = Array.of_list entries in
  let len = Array.length arr in
  let cur =
    let rec find i =
      if i >= len then -1
      else
        let _, _, url = arr.(i) in
        if url = current_url then i else find (i + 1)
    in
    find 0
  in
  if cur < 0 then ""
  else
    let prev =
      if cur > 0 then
        let _, title, url = arr.(cur - 1) in
        Printf.sprintf {|<a href="%s" class="prev-link">← %s</a>|} url title
      else ""
    in
    let next =
      if cur < len - 1 then
        let _, title, url = arr.(cur + 1) in
        Printf.sprintf {|<a href="%s" class="next-link">%s →</a>|} url title
      else ""
    in
    if prev = "" && next = "" then ""
    else Printf.sprintf {|<nav class="prev-next">%s%s</nav>|} prev next

let generate_lib_examples_nav_items ~current_url lib_name =
  let dir = lib_examples_dir lib_name in
  if not (Sys.file_exists dir && Sys.is_directory dir) then ""
  else
    let entries =
      Sys.readdir dir |> Array.to_list
      |> List.filter (fun entry -> Sys.is_directory (Filename.concat dir entry))
      |> List.sort String.compare
      |> List.map (fun entry ->
          let slug = Site.strip_order_prefix entry in
          let title = Site.title_case slug in
          let url = Printf.sprintf "/docs/%s/examples/%s/" lib_name slug in
          (title, url))
    in
    match entries with
    | [] -> ""
    | _ ->
        entries
        |> List.map (fun (title, url) ->
            let active =
              if url = current_url then {| class="active"|} else ""
            in
            Printf.sprintf {|          <li><a href="%s"%s>%s</a></li>|} url
              active title)
        |> String.concat "\n"

let generate_tab_nav ~current_url lib_name =
  let guides_items = generate_lib_nav ~current_url lib_name in
  let examples_items = generate_lib_examples_nav_items ~current_url lib_name in
  let api_items = Api.nav_items ~current_url lib_name in
  let has_examples = examples_items <> "" in
  let has_api = api_items <> "" in
  let is_examples_page =
    match Site.find_sub current_url "/examples/" with
    | Some _ -> true
    | None -> false
  in
  let is_api_page =
    match Site.find_sub current_url "/api/" with
    | Some _ -> true
    | None -> false
  in
  let active_tab =
    if is_api_page && has_api then "api"
    else if is_examples_page && has_examples then "examples"
    else "guides"
  in
  let checked tab = if tab = active_tab then " checked" else "" in
  let buf = Buffer.create 1024 in
  Buffer.add_string buf {|      <div class="nav-tabs">|};
  Buffer.add_char buf '\n';
  Printf.bprintf buf
    {|        <input type="radio" id="tab-guides" name="nav-tab"%s>|}
    (checked "guides");
  Buffer.add_char buf '\n';
  Buffer.add_string buf {|        <label for="tab-guides">Guides</label>|};
  Buffer.add_char buf '\n';
  if has_examples then (
    Printf.bprintf buf
      {|        <input type="radio" id="tab-examples" name="nav-tab"%s>|}
      (checked "examples");
    Buffer.add_char buf '\n';
    Buffer.add_string buf {|        <label for="tab-examples">Examples</label>|};
    Buffer.add_char buf '\n');
  if has_api then (
    Printf.bprintf buf
      {|        <input type="radio" id="tab-api" name="nav-tab"%s>|}
      (checked "api");
    Buffer.add_char buf '\n';
    Buffer.add_string buf {|        <label for="tab-api">API</label>|};
    Buffer.add_char buf '\n');
  Buffer.add_string buf {|        <div class="tab-panel" id="panel-guides">|};
  Buffer.add_char buf '\n';
  Buffer.add_string buf {|          <ul class="nav-links">|};
  Buffer.add_char buf '\n';
  Buffer.add_string buf guides_items;
  Buffer.add_char buf '\n';
  Buffer.add_string buf {|          </ul>|};
  Buffer.add_char buf '\n';
  Buffer.add_string buf {|        </div>|};
  Buffer.add_char buf '\n';
  if has_examples then (
    Buffer.add_string buf
      {|        <div class="tab-panel" id="panel-examples">|};
    Buffer.add_char buf '\n';
    Buffer.add_string buf {|          <ul class="nav-links">|};
    Buffer.add_char buf '\n';
    Buffer.add_string buf examples_items;
    Buffer.add_char buf '\n';
    Buffer.add_string buf {|          </ul>|};
    Buffer.add_char buf '\n';
    Buffer.add_string buf {|        </div>|};
    Buffer.add_char buf '\n');
  if has_api then (
    Buffer.add_string buf {|        <div class="tab-panel" id="panel-api">|};
    Buffer.add_char buf '\n';
    Buffer.add_string buf {|          <ul class="nav-links">|};
    Buffer.add_char buf '\n';
    Buffer.add_string buf api_items;
    Buffer.add_string buf {|          </ul>|};
    Buffer.add_char buf '\n';
    Buffer.add_string buf {|        </div>|};
    Buffer.add_char buf '\n');
  Buffer.add_string buf {|      </div>|};
  Buffer.contents buf

(* -- Template -- *)

let select_template path =
  let parts = String.split_on_char '/' path in
  let name =
    match parts with
    | "docs" :: lib :: _ when Site.find_library lib <> None ->
        "layout_docs_lib.html"
    | "docs" :: _ -> "layout_docs.html"
    | _ -> "main.html"
  in
  Filename.concat templates_dir name

let apply_template ~template ~title ~breadcrumbs ~content ~lib ~is_lib_index
    ~page_title ~path =
  let current_url = Site.url_of_path path in
  let tab_nav =
    match lib with
    | Some lib -> generate_tab_nav ~current_url lib.Site.name
    | None -> ""
  in
  let prev_next =
    match lib with
    | Some lib -> generate_prev_next ~current_url lib.Site.name
    | None -> ""
  in
  Site.apply_template ~template ~title ~breadcrumbs ~content ~lib ~is_lib_index
    ~page_title ~path ~tab_nav ~prev_next ()

let dest_path path =
  let stem = Filename.chop_extension path in
  if Filename.basename stem = "index" then
    Filename.concat build_dir (stem ^ ".html")
  else Filename.concat build_dir (Filename.concat stem "index.html")

(* -- Processing -- *)

let render_markdown content =
  content
  |> Cmarkit.Doc.of_string ~heading_auto_ids:true ~strict:false
  |> Hilite_markdown.transform ~skip_unknown_languages:true
  |> Cmarkit_html.of_doc ~safe:false

let process_markdown path content =
  let html = render_markdown content |> Site.rewrite_doc_hrefs in
  let h1 = Site.extract_h1 html in
  let title = match h1 with Some t -> t ^ " - raven" | None -> "raven" in
  let page_title =
    match h1 with
    | Some t -> t
    | None -> Site.title_case (Filename.chop_extension (Filename.basename path))
  in
  let breadcrumbs = Site.make_breadcrumbs (Site.url_segments path) page_title in
  let lib =
    match String.split_on_char '/' path with
    | "docs" :: lib_name :: _ -> Site.find_library lib_name
    | _ -> None
  in
  let is_lib_index =
    match String.split_on_char '/' path with
    | "docs" :: lib_name :: rest when Site.find_library lib_name <> None -> (
        match rest with [ "index.md" ] | [] -> true | _ -> false)
    | _ -> false
  in
  let html = if is_lib_index then Site.strip_h1 html else html in
  let template = Site.read_file (select_template path) in
  apply_template ~template ~title ~breadcrumbs ~content:html ~lib ~is_lib_index
    ~page_title ~path

let highlight_html_code_blocks html =
  let buf = Buffer.create (String.length html) in
  let len = String.length html in
  let i = ref 0 in
  let pre_open = {|<pre class="language-|} in
  let pre_open_len = String.length pre_open in
  let code_close = "</code></pre>" in
  let code_close_len = String.length code_close in
  while !i < len do
    match Site.find_sub ~start:!i html pre_open with
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
            match Site.find_sub ~start:lang_end html code_tag with
            | None ->
                Buffer.add_char buf html.[!i];
                i := !i + 1
            | Some code_tag_start -> (
                let content_start = code_tag_start + String.length code_tag in
                match Site.find_sub ~start:content_start html code_close with
                | None ->
                    Buffer.add_char buf html.[!i];
                    i := !i + 1
                | Some content_end -> (
                    let raw =
                      String.sub html content_start (content_end - content_start)
                    in
                    let code =
                      raw |> Site.replace "&amp;" "&" |> Site.replace "&lt;" "<"
                      |> Site.replace "&gt;" ">"
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
    | ".md" -> (dest_path path, process_markdown path (Site.read_file full_path))
    | ".html" | ".htm" ->
        (dest_path path, highlight_html_code_blocks (Site.read_file full_path))
    | _ -> (Filename.concat build_dir path, Site.read_file full_path)
  in
  Site.ensure_dir (Filename.dirname out);
  Site.write_file out content

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
  let slug = Site.strip_order_prefix entry in
  let path = Printf.sprintf "docs/%s/examples/%s.md" lib.Site.name slug in
  let readme_path = Filename.concat example_dir "README.md" in
  let prose_html =
    if Sys.file_exists readme_path then
      render_markdown (Site.read_file readme_path)
    else Printf.sprintf "<h1>%s</h1>" (escape_html (Site.title_case slug))
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
        let code = Site.read_file (Filename.concat example_dir f) in
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
  let h1 = Site.extract_h1 html in
  let title = match h1 with Some t -> t ^ " - raven" | None -> "raven" in
  let page_title = match h1 with Some t -> t | None -> Site.title_case slug in
  let breadcrumbs = Site.make_breadcrumbs (Site.url_segments path) page_title in
  let template = Site.read_file (select_template path) in
  let content =
    apply_template ~template ~title ~breadcrumbs ~content:html ~lib:(Some lib)
      ~is_lib_index:false ~page_title ~path
  in
  let out = dest_path path in
  Site.ensure_dir (Filename.dirname out);
  Site.write_file out content

let () =
  Site.walk site_dir
  |> List.iter (fun p ->
      process_file ~path:(Site.strip_prefix ~prefix:site_dir p) p);
  Site.walk docs_dir
  |> List.iter (fun p ->
      let rel = Site.strip_prefix ~prefix:docs_dir p in
      let path = Filename.concat "docs" rel in
      process_file ~path p);
  Site.libraries
  |> List.iter (fun lib ->
      let dir = lib_doc_dir lib.Site.name in
      if Sys.file_exists dir && Sys.is_directory dir then
        Site.walk dir
        |> List.iter (fun full_path ->
            let ext = Filename.extension full_path in
            let base = Filename.basename full_path in
            if base = "dune" || ext = ".mld" then ()
            else
              let rel = Site.strip_prefix ~prefix:dir full_path in
              let rel_base = Filename.basename rel in
              let rel_dir = Filename.dirname rel in
              let clean_base =
                if Filename.extension rel_base = ".md" then
                  Site.strip_order_prefix (Filename.chop_extension rel_base)
                  ^ ".md"
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
  Site.libraries
  |> List.iter (fun lib ->
      let dir = lib_examples_dir lib.Site.name in
      if Sys.file_exists dir && Sys.is_directory dir then
        Sys.readdir dir |> Array.to_list |> List.sort String.compare
        |> List.iter (fun entry ->
            let full = Filename.concat dir entry in
            if Sys.is_directory full then process_example ~lib full));
  let tab_nav lib_name =
   fun current_url -> generate_tab_nav ~current_url lib_name
  in
  Api.generate ~build_dir ~templates_dir ~tab_nav
