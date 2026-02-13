let site_dir = "site"
let build_dir = "build"
let templates_dir = "templates"

let libraries =
  [ "nx"; "rune"; "kaun"; "hugin"; "saga"; "talon"; "quill"; "fehu"; "sowilo" ]

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

let title_case s =
  s
  |> String.split_on_char '-'
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

(* -- Template -- *)

let select_template path =
  let parts = String.split_on_char '/' path in
  let name =
    match parts with
    | "docs" :: lib :: _ when List.mem lib libraries ->
      "layout_docs_" ^ lib ^ ".html"
    | "docs" :: _ -> "layout_docs.html"
    | _ -> "main.html"
  in
  Filename.concat templates_dir name

let apply_template ~template ~title ~breadcrumbs ~content =
  template
  |> replace "{{title}}" title
  |> replace "{{breadcrumbs}}" breadcrumbs
  |> replace "{{content}}" content

(* -- Processing -- *)

let render_markdown content =
  content
  |> Cmarkit.Doc.of_string ~heading_auto_ids:true ~strict:false
  |> Cmarkit_html.of_doc ~safe:false

let process_markdown path content =
  let html = render_markdown content in
  let h1 = extract_h1 html in
  let title =
    match h1 with Some t -> t ^ " - raven" | None -> "raven"
  in
  let page_title =
    match h1 with
    | Some t -> t
    | None -> title_case (Filename.chop_extension (Filename.basename path))
  in
  let breadcrumbs = make_breadcrumbs (url_segments path) page_title in
  let template = read_file (select_template path) in
  apply_template ~template ~title ~breadcrumbs ~content:html

let process_file full_path =
  let path = strip_prefix ~prefix:site_dir full_path in
  let ext = Filename.extension path in
  let out, content =
    match ext with
    | ".md" -> (dest_path path, process_markdown path (read_file full_path))
    | ".html" | ".htm" -> (dest_path path, read_file full_path)
    | _ -> (Filename.concat build_dir path, read_file full_path)
  in
  ensure_dir (Filename.dirname out);
  write_file out content

let () = walk site_dir |> List.iter process_file
