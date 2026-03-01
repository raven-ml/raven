(* Shared types and utilities for the site generator. *)

type library = {
  name : string;
  display : string;
  color : string;
  description : string;
  tagline : string;
  symbol : string;
}

let libraries =
  [
    {
      name = "nx";
      display = "nx";
      color = "color-blue";
      description = "N-dimensional arrays with linear algebra operations";
      tagline = "N-dimensional arrays for OCaml";
      symbol = "";
    };
    {
      name = "rune";
      display = {|<span class="rune-symbol">ᚱ</span> rune|};
      color = "color-orange";
      description = "Automatic differentiation and functional transformations";
      tagline = "Functional transformations for Nx arrays";
      symbol = {|ᚱ|};
    };
    {
      name = "kaun";
      display = {|<span class="rune-symbol">ᚲ</span> kaun|};
      color = "color-red";
      description = "Neural networks and training";
      tagline = "Neural networks for OCaml";
      symbol = {|ᚲ|};
    };
    {
      name = "hugin";
      display = {|<span class="rune-symbol">ᛞ</span> hugin|};
      color = "color-purple";
      description = "Publication-quality plotting";
      tagline = "Plotting for OCaml";
      symbol = {|ᛞ|};
    };
    {
      name = "brot";
      display = {|<span class="rune-symbol">ᚨ</span> brot|};
      color = "color-cyan";
      description =
        "Fast, HuggingFace-compatible tokenization for language models";
      tagline = "Tokenization for OCaml";
      symbol = {|ᚨ|};
    };
    {
      name = "talon";
      display = {|<span class="rune-symbol">ᛃ</span> talon|};
      color = "color-pink";
      description = "Fast and elegant dataframes with type-safe operations";
      tagline = "Dataframes for OCaml";
      symbol = {|ᛃ|};
    };
    {
      name = "quill";
      display = {|<span class="rune-symbol">ᛈ</span> quill|};
      color = "color-green";
      description = "Interactive REPL and markdown notebooks";
      tagline = "Interactive REPL and markdown notebooks";
      symbol = {|ᛈ|};
    };
    {
      name = "fehu";
      display = {|<span class="rune-symbol">ᚠ</span> fehu|};
      color = "color-lime";
      description = "Reinforcement learning for OCaml";
      tagline = "Reinforcement learning for OCaml";
      symbol = {|ᚠ|};
    };
    {
      name = "sowilo";
      display = {|<span class="rune-symbol">ᛋ</span> sowilo|};
      color = "color-indigo";
      description = "Differentiable computer vision";
      tagline = "Differentiable computer vision for OCaml";
      symbol = {|ᛋ|};
    };
  ]

let find_library name = List.find_opt (fun lib -> lib.name = name) libraries

(*---------------------------------------------------------------------------
  String utilities
  ---------------------------------------------------------------------------*)

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

(* Strip order prefixes from each segment of a relative href path.
   "../02-pipeline/" → "../pipeline/", "05-algorithms/" → "algorithms/" *)
let strip_href_order_prefixes href =
  if String.length href = 0 || href.[0] = '/' || String.contains href ':' then
    href
  else
    let anchor, path =
      match String.index_opt href '#' with
      | Some i ->
          (String.sub href i (String.length href - i), String.sub href 0 i)
      | None -> ("", href)
    in
    let parts = String.split_on_char '/' path in
    let cleaned =
      List.map
        (fun seg -> if seg = ".." then seg else strip_order_prefix seg)
        parts
    in
    String.concat "/" cleaned ^ anchor

(* Rewrite all href="..." in [html] to strip order prefixes from relative
   paths. *)
let rewrite_doc_hrefs html =
  let buf = Buffer.create (String.length html) in
  let attr = {|href="|} in
  let attr_len = String.length attr in
  let len = String.length html in
  let i = ref 0 in
  while !i < len do
    match find_sub ~start:!i html attr with
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
            Buffer.add_string buf (strip_href_order_prefixes href);
            Buffer.add_char buf '"';
            i := href_end + 1)
  done;
  Buffer.contents buf

let title_case s =
  s |> String.split_on_char '-'
  |> List.map (fun w ->
      if w = "" then w
      else
        String.make 1 (Char.uppercase_ascii w.[0])
        ^ String.sub w 1 (String.length w - 1))
  |> String.concat " "

(*---------------------------------------------------------------------------
  File system
  ---------------------------------------------------------------------------*)

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

(*---------------------------------------------------------------------------
  HTML utilities
  ---------------------------------------------------------------------------*)

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

let strip_h1 html =
  match find_sub html "<h1" with
  | None -> html
  | Some i -> (
      match find_sub ~start:i html "</h1>" with
      | None -> html
      | Some k ->
          let after = k + 5 in
          String.sub html 0 i
          ^ String.sub html after (String.length html - after))

(*---------------------------------------------------------------------------
  Paths and URLs
  ---------------------------------------------------------------------------*)

let strip_prefix ~prefix path =
  let plen = String.length prefix + 1 in
  String.sub path plen (String.length path - plen)

let url_segments path =
  let stem = Filename.chop_extension path in
  let parts = String.split_on_char '/' stem in
  match List.rev parts with "index" :: rest -> List.rev rest | _ -> parts

let url_of_path path =
  let segments = url_segments path in
  "/" ^ String.concat "/" segments ^ "/"

(*---------------------------------------------------------------------------
  Breadcrumbs
  ---------------------------------------------------------------------------*)

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

(*---------------------------------------------------------------------------
  Template application
  ---------------------------------------------------------------------------*)

let apply_template ~template ~title ~breadcrumbs ~content ~lib ~is_lib_index
    ~page_title ~path ~tab_nav ~prev_next ?(toc = "") () =
  let t =
    template |> replace "{{title}}" title
    |> replace "{{breadcrumbs}}" breadcrumbs
    |> replace "{{content}}" content
    |> replace "{{toc}}" toc
  in
  match lib with
  | None ->
      let docs_breadcrumb =
        let segments = url_segments path in
        match segments with
        | [ "docs" ] | [ "docs"; "index" ] ->
            {|<span class="breadcrumb-text">docs</span>|}
        | "docs" :: _ ->
            Printf.sprintf
              {|<a href="/docs/" class="breadcrumb-link">docs</a>
      <span class="breadcrumb-separator">/</span>
      <span class="breadcrumb-text">%s</span>|}
              page_title
        | _ -> ""
      in
      t |> replace "{{lib_hero}}" ""
      |> replace "{{docs_breadcrumb}}" docs_breadcrumb
  | Some lib ->
      let hero =
        if is_lib_index then
          let symbol_html =
            if lib.symbol = "" then ""
            else
              Printf.sprintf {|<span class="rune-symbol">%s</span> |} lib.symbol
          in
          Printf.sprintf
            {|      <div class="docs-hero" style="--lib-color: var(--color-%s)">
        <h1>%s%s</h1>
        <p class="tagline">%s</p>
      </div>
|}
            lib.name symbol_html lib.name lib.tagline
        else ""
      in
      let lib_breadcrumb =
        if is_lib_index then
          Printf.sprintf {|<span class="breadcrumb-text %s">%s</span>|}
            lib.color lib.display
        else
          Printf.sprintf
            {|<a href="/docs/%s/" class="breadcrumb-text %s">%s</a>
      <span class="breadcrumb-separator">/</span>
      <span class="breadcrumb-text">%s</span>|}
            lib.name lib.color lib.display page_title
      in
      t
      |> replace "{{lib_name}}" lib.name
      |> replace "{{lib_display}}" lib.display
      |> replace "{{lib_color}}" lib.color
      |> replace "{{lib_description}}" lib.description
      |> replace "{{lib_tab_nav}}" tab_nav
      |> replace "{{lib_hero}}" hero
      |> replace "{{lib_breadcrumb}}" lib_breadcrumb
      |> replace "{{prev_next}}" prev_next
