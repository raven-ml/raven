(* API reference generation from odoc HTML output.

   Extracts content from odoc-generated HTML pages, rewrites internal links to
   match the site URL scheme, and produces pages wrapped in the site
   template. *)

let odoc_dir = Filename.concat ".." "_doc/_html"

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
    if not (Sys.file_exists full && Sys.is_directory full) then []
    else
      Sys.readdir full |> Array.to_list
      |> List.filter (fun e -> Sys.is_directory (Filename.concat full e))
      |> List.sort String.compare
      |> List.concat_map (fun e -> walk_modules dir (Filename.concat rel e))
  in
  self @ children

(* Dune's package index lists the public entry modules of every library. It is
   the same inventory used by odoc, including unwrapped libraries. *)
let entry_modules pkg_name =
  let index =
    Filename.concat (Filename.concat odoc_dir pkg_name) "index.html"
  in
  let html = Site.read_file index in
  let rec collect start acc =
    match Site.find_sub ~start html {|href="|} with
    | None -> List.sort_uniq String.compare acc
    | Some pos ->
        let first = pos + 6 in
        let last = String.index_from html first '"' in
        let href = String.sub html first (last - first) in
        let acc =
          match String.split_on_char '/' href with
          | [ name; "index.html" ] when name <> ".." -> name :: acc
          | _ -> acc
        in
        collect (last + 1) acc
  in
  match collect 0 [] with
  | [] -> failwith ("No public modules in " ^ index)
  | modules -> modules

let package_modules pkg_name =
  let pkg_dir = Filename.concat odoc_dir pkg_name in
  entry_modules pkg_name
  |> List.concat_map (fun name -> walk_modules pkg_dir name)

(* A link to a public interface in Markdown reaches its rendered module.
   Private interfaces have no published target and are rejected by Links.

   An interface is an entry module, or a module of a wrapped library whose
   entry module sits in the same directory and exposes it under its own
   name: [lib/gg/color.mli] beside [lib/gg/hugin_gg.mli] is published at
   [Hugin_gg/Color/] when odoc rendered that page. *)
let register_sources links =
  let module_name src =
    Filename.basename src |> Filename.chop_extension |> String.capitalize_ascii
  in
  Site.libraries
  |> List.iter (fun lib ->
      let modules = entry_modules lib.Site.name in
      let mlis =
        Site.walk
          (Filename.concat
             (Filename.concat (Filename.concat ".." "packages") lib.name)
             "lib")
        |> List.filter (fun src -> Filename.extension src = ".mli")
      in
      let entry_in dir =
        List.find_map
          (fun src ->
            let name = module_name src in
            if Filename.dirname src = dir && List.mem name modules then
              Some name
            else None)
          mlis
      in
      let published rel =
        Sys.file_exists
          (Filename.concat odoc_dir
             (Filename.concat lib.name (Filename.concat rel "index.html")))
      in
      mlis
      |> List.iter (fun src ->
          let name = module_name src in
          let rel =
            if List.mem name modules then Some name
            else
              match entry_in (Filename.dirname src) with
              | Some entry when published (entry ^ "/" ^ name) ->
                  Some (entry ^ "/" ^ name)
              | _ -> None
          in
          Option.iter
            (fun rel ->
              Links.register links ~src
                ~url:(Printf.sprintf "/docs/%s/api/%s/" lib.name rel))
            rel))

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
let extract_content ~path html =
  let find ~start tag =
    match Site.find_sub ~start html tag with
    | Some i -> i
    | None -> failwith (path ^ ": missing odoc region " ^ tag)
  in
  let first = find ~start:0 {|<header class="odoc-preamble">|} in
  let last = find ~start:first "</header>" + 9 in
  let preamble = String.sub html first (last - first) in
  let first = find ~start:last {|<div class="odoc-content">|} in
  let last = find ~start:first "</body>" in
  let body = String.trim (String.sub html first (last - first)) in
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
              if
                String.length text > 12 && String.sub text 0 12 = "Module type "
              then Some (String.sub text 12 (String.length text - 12))
              else if String.length text > 7 && String.sub text 0 7 = "Module "
              then Some (String.sub text 7 (String.length text - 7))
              else Some text))

(*---------------------------------------------------------------------------
  Link rewriting
  ---------------------------------------------------------------------------*)

(* Resolve [..] segments in a path. *)
let resolve_relative ~base href = Links.normalize (Filename.concat base href)

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
  let buf = Buffer.create 512 in
  entry_modules pkg_name
  |> List.iter (fun mod_name ->
      let url = Printf.sprintf "/docs/%s/api/%s/" pkg_name mod_name in
      let active = if url = current_url then {| class="active"|} else "" in
      Printf.bprintf buf {|            <li><a href="%s"%s>%s</a></li>
|} url
        active mod_name;
      direct_submodules pkg_name mod_name
      |> List.iter (fun sub ->
          let sub_url =
            Printf.sprintf "/docs/%s/api/%s/%s/" pkg_name mod_name sub
          in
          let active =
            if sub_url = current_url then {| class="active"|} else ""
          in
          let prefix = "module-type-" in
          let label =
            if String.starts_with ~prefix sub then
              String.sub sub (String.length prefix)
                (String.length sub - String.length prefix)
            else sub
          in
          Printf.bprintf buf
            {|            <li class="nav-sub"><a href="%s"%s>%s</a></li>
|}
            sub_url active label));
  Buffer.contents buf

(*---------------------------------------------------------------------------
  Page generation
  ---------------------------------------------------------------------------*)

let process_page ~build_dir ~templates_dir ~tab_nav ~lib ~module_rel full_path =
  let html = Site.read_file full_path in
  let toc = extract_toc html in
  let content = extract_content ~path:full_path html in
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
