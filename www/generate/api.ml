(* API reference generation from odoc HTML output.

   Extracts content from odoc-generated HTML pages, rewrites internal links to
   match the site URL scheme, and produces pages wrapped in the site
   template. *)

let odoc_dir = Filename.concat ".." "_doc/_html"

(* Libraries to include per package. Each entry is (library_name,
   entry_module_name). The library name is displayed in the sidebar; the module
   name is the odoc directory name. *)
let libraries =
  [
    ( "nx",
      [
        ("nx", "Nx");
        ("nx.backend", "Nx_backend");
        ("nx.buffer", "Nx_buffer");
        ("nx.core", "Nx_core");
        ("nx.effect", "Nx_effect");
        ("nx.io", "Nx_io");
      ] );
    ("rune", [ ("rune", "Rune") ]);
    ( "kaun",
      [
        ("kaun", "Kaun");
        ("kaun.datasets", "Kaun_datasets");
        ("kaun.hf", "Kaun_hf");
      ] );
    ("brot", [ ("brot", "Brot") ]);
    ("talon", [ ("talon", "Talon"); ("talon.csv", "Talon_csv") ]);
    ("hugin", [ ("hugin", "Hugin") ]);
    ("quill", [ ("quill", "Quill") ]);
    ("fehu", [ ("fehu", "Fehu"); ("fehu.envs", "Fehu_envs") ]);
    ("sowilo", [ ("sowilo", "Sowilo") ]);
  ]

(*---------------------------------------------------------------------------
  Module discovery
  ---------------------------------------------------------------------------*)

(* Walk [dir] recursively, collecting all [index.html] files. Returns
   [(rel_path, full_path)] where [rel_path] is relative to [dir]. *)
let rec walk_modules dir rel =
  let full = Filename.concat dir rel in
  let index = Filename.concat full "index.html" in
  let self = if Sys.file_exists index then [ (rel, index) ] else [] in
  let children =
    if not (Sys.is_directory full) then []
    else
      Sys.readdir full |> Array.to_list
      |> List.filter (fun e -> Sys.is_directory (Filename.concat full e))
      |> List.sort String.compare
      |> List.concat_map (fun e -> walk_modules dir (Filename.concat rel e))
  in
  self @ children

(* All module pages for a package, filtered by [libraries]. *)
let package_modules pkg_name =
  let pkg_dir = Filename.concat odoc_dir pkg_name in
  if not (Sys.file_exists pkg_dir && Sys.is_directory pkg_dir) then []
  else
    match List.assoc_opt pkg_name libraries with
    | None -> []
    | Some libs ->
        libs
        |> List.concat_map (fun (_lib_name, mod_name) ->
            walk_modules pkg_dir mod_name)

(* Direct child subdirectories of [dir/mod_name] that have an index.html. *)
let direct_submodules pkg_name mod_name =
  let mod_dir = Filename.concat (Filename.concat odoc_dir pkg_name) mod_name in
  if not (Sys.file_exists mod_dir && Sys.is_directory mod_dir) then []
  else
    Sys.readdir mod_dir |> Array.to_list
    |> List.filter (fun e ->
        let d = Filename.concat mod_dir e in
        Sys.is_directory d && Sys.file_exists (Filename.concat d "index.html"))
    |> List.sort String.compare

(*---------------------------------------------------------------------------
  HTML extraction
  ---------------------------------------------------------------------------*)

(* Extract the preamble and content from an odoc HTML page. Drops <nav>, <head>,
   scripts, and the local TOC. *)
let extract_content html =
  let preamble =
    let tag = {|<header class="odoc-preamble">|} in
    match Site.find_sub html tag with
    | None -> ""
    | Some i -> (
        match Site.find_sub ~start:i html "</header>" with
        | None -> ""
        | Some j -> String.sub html i (j + 9 - i))
  in
  let body =
    let tag = {|<div class="odoc-content">|} in
    match Site.find_sub html tag with
    | None -> ""
    | Some i -> (
        match Site.find_sub html "</body>" with
        | None -> ""
        | Some j -> String.trim (String.sub html i (j - i)))
  in
  preamble ^ "\n" ^ body

(* Extract the local TOC from odoc HTML. Returns the inner <nav> wrapped in a
   right-sidebar container. *)
let extract_toc html =
  let tag = {|<nav class="odoc-toc odoc-local-toc">|} in
  match Site.find_sub html tag with
  | None -> ""
  | Some i -> (
      match Site.find_sub ~start:i html "</nav>" with
      | None -> ""
      | Some j ->
          let nav = String.sub html i (j + 6 - i) in
          Printf.sprintf
            {|    <aside class="toc">
      <div class="toc-inner">
        <div class="toc-title">On this page</div>
%s
      </div>
    </aside>|}
            nav)

(* Extract the module name from the <h1> in the preamble. "Module
   <code><span>Nx</span></code>" -> "Nx" *)
let extract_title html =
  let tag = {|<header class="odoc-preamble">|} in
  match Site.find_sub html tag with
  | None -> None
  | Some start -> (
      match Site.find_sub ~start html "<h1>" with
      | None -> None
      | Some h1 -> (
          match Site.find_sub ~start:h1 html "</h1>" with
          | None -> None
          | Some h1_end ->
              let raw = String.sub html (h1 + 4) (h1_end - h1 - 4) in
              let text = String.trim (Site.strip_tags raw) in
              if String.length text > 7 && String.sub text 0 7 = "Module " then
                Some (String.sub text 7 (String.length text - 7))
              else if
                String.length text > 12 && String.sub text 0 12 = "Module type "
              then Some (String.sub text 12 (String.length text - 12))
              else Some text))

(*---------------------------------------------------------------------------
  Link rewriting
  ---------------------------------------------------------------------------*)

(* Resolve [..] segments in a path. *)
let resolve_relative ~base href =
  let base_parts = List.rev (String.split_on_char '/' base) in
  let href_parts = String.split_on_char '/' href in
  let rec go base = function
    | ".." :: rest ->
        let base' = match base with _ :: tl -> tl | [] -> [] in
        go base' rest
    | rest -> List.rev base @ rest
  in
  String.concat "/" (go base_parts href_parts)

(* Rewrite a single href from odoc-relative to site-absolute. [current_dir]: the
   module's position relative to the odoc root, e.g. "nx/Nx/Infix". *)
let rewrite_href ~current_dir href =
  if String.length href = 0 || href.[0] = '#' then href
  else if String.contains href ':' then href
  else
    let anchor, path =
      match String.index_opt href '#' with
      | Some i ->
          (String.sub href i (String.length href - i), String.sub href 0 i)
      | None -> ("", href)
    in
    let resolved = resolve_relative ~base:current_dir path in
    let parts =
      String.split_on_char '/' resolved
      |> List.filter (fun s -> s <> "" && s <> "index.html")
    in
    match parts with
    | pkg :: rest when Site.find_library pkg <> None ->
        "/docs/" ^ pkg ^ "/api/" ^ String.concat "/" rest ^ "/" ^ anchor
    | _ -> href

(* Rewrite all href="..." in [html]. *)
let rewrite_hrefs ~current_dir html =
  let buf = Buffer.create (String.length html) in
  let attr = {|href="|} in
  let attr_len = String.length attr in
  let len = String.length html in
  let i = ref 0 in
  while !i < len do
    match Site.find_sub ~start:!i html attr with
    | None ->
        Buffer.add_string buf (String.sub html !i (len - !i));
        i := len
    | Some pos -> (
        Buffer.add_string buf (String.sub html !i (pos + attr_len - !i));
        let href_start = pos + attr_len in
        match String.index_from_opt html href_start '"' with
        | None -> i := href_start
        | Some href_end ->
            let href = String.sub html href_start (href_end - href_start) in
            Buffer.add_string buf (rewrite_href ~current_dir href);
            Buffer.add_char buf '"';
            i := href_end + 1)
  done;
  Buffer.contents buf

(*---------------------------------------------------------------------------
  Sidebar
  ---------------------------------------------------------------------------*)

(* Generate the API tab panel contents for a package's sidebar. *)
let nav_items ~current_url pkg_name =
  match List.assoc_opt pkg_name libraries with
  | None -> ""
  | Some libs ->
      let buf = Buffer.create 512 in
      libs
      |> List.iter (fun (lib_name, mod_name) ->
          Printf.bprintf buf {|            <li class="nav-group">%s</li>|}
            lib_name;
          Buffer.add_char buf '\n';
          let url = Printf.sprintf "/docs/%s/api/%s/" pkg_name mod_name in
          let active = if url = current_url then {| class="active"|} else "" in
          Printf.bprintf buf {|            <li><a href="%s"%s>%s</a></li>|} url
            active mod_name;
          Buffer.add_char buf '\n';
          direct_submodules pkg_name mod_name
          |> List.iter (fun sub ->
              let sub_url =
                Printf.sprintf "/docs/%s/api/%s/%s/" pkg_name mod_name sub
              in
              let active =
                if sub_url = current_url then {| class="active"|} else ""
              in
              Printf.bprintf buf
                {|            <li class="nav-sub"><a href="%s"%s>%s</a></li>|}
                sub_url active sub;
              Buffer.add_char buf '\n'));
      Buffer.contents buf

(*---------------------------------------------------------------------------
  Page generation
  ---------------------------------------------------------------------------*)

let process_page ~build_dir ~templates_dir ~tab_nav ~lib ~module_rel full_path =
  let html = Site.read_file full_path in
  let toc = extract_toc html in
  let content = extract_content html in
  let current_dir = Filename.concat lib.Site.name module_rel in
  let content = rewrite_hrefs ~current_dir content in
  let content = Printf.sprintf {|<div class="odoc-api">%s</div>|} content in
  let mod_title =
    match extract_title html with
    | Some t -> t
    | None -> (
        match List.rev (String.split_on_char '/' module_rel) with
        | last :: _ -> last
        | [] -> "API")
  in
  let path = Printf.sprintf "docs/%s/api/%s/index.html" lib.name module_rel in
  let current_url = Site.url_of_path path in
  let title = mod_title ^ " - " ^ lib.name ^ " - raven" in
  let segments =
    "docs" :: lib.name :: "api" :: String.split_on_char '/' module_rel
  in
  let breadcrumbs = Site.make_breadcrumbs segments mod_title in
  let template =
    Site.read_file (Filename.concat templates_dir "layout_docs_lib.html")
  in
  let result =
    Site.apply_template ~template ~title ~breadcrumbs ~content ~lib:(Some lib)
      ~is_lib_index:false ~page_title:mod_title ~path
      ~tab_nav:(tab_nav current_url) ~prev_next:"" ~toc ()
  in
  let out =
    Filename.concat build_dir
      (Printf.sprintf "docs/%s/api/%s/index.html" lib.name module_rel)
  in
  Site.ensure_dir (Filename.dirname out);
  Site.write_file out result

let generate ~build_dir ~templates_dir ~tab_nav =
  Site.libraries
  |> List.iter (fun lib ->
      let modules = package_modules lib.Site.name in
      let tab_nav = tab_nav lib.Site.name in
      modules
      |> List.iter (fun (module_rel, full_path) ->
          process_page ~build_dir ~templates_dir ~tab_nav ~lib ~module_rel
            full_path))
