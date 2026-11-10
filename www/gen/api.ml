(* The API reference: what odoc wrote for [dune build @doc-json], each
   page a JSON object (odoc's "header", "preamble", "content", "toc",
   "breadcrumbs"), in _doc/_html of the build.

   - _doc/_html/<pkg>/<Path>/index.html.json is /docs/<pkg>/api/<Path>/,
     and /docs/<pkg>/api/ lists the package's libraries and their modules.
   - A link between odoc's pages goes to the page on the site; a link to
     a package the site doesn't show keeps its text and loses its link.
   - OCaml in documentation comments is highlighted as everywhere else;
     signatures keep odoc's markup, its links to types, coloured by the
     stylesheet. *)

open Html

let ( / ) = Filename.concat

(* Where odoc's pages are, in the build: ROOT is _build/default. *)
let dir root = root / "_doc" / "_html"

(* The package's API, if it has one: the packages with no public library
   have none. *)
let has_api root pkg = Sys.file_exists (dir root / pkg / "index.html.json")

type page = {
  pkg : string;
  path : string;  (** Below the package, "" for its index: "Jera/Linear". *)
  url : string;
}

let url_of pkg path = if path = "" then sprintf "/docs/%s/api/" pkg else sprintf "/docs/%s/api/%s/" pkg path

(* The package's pages, its index first, then each module in a
   depth-first walk, sorted. *)
let pages root pkg =
  let rec walk rel =
    let here = dir root / pkg / rel in
    Sys.readdir here |> Array.to_list |> List.sort compare
    |> List.concat_map (fun n ->
           let rel' = if rel = "" then n else rel ^ "/" ^ n in
           if Sys.is_directory (here / n) && Sys.file_exists (here / n / "index.html.json") then
             { pkg; path = rel'; url = url_of pkg rel' } :: walk rel'
           else [])
  in
  { pkg; path = ""; url = url_of pkg "" } :: walk ""

let read root p =
  let file = dir root / p.pkg / p.path / "index.html.json" in
  (file, Json.parse ~file (Repo.read file))

(* The page of the module [name] in [pkg]: the shallowest page with that
   name, if only one is. "rune.mli" is the module Rune; an interface
   inside a wrapped library is a module of it. *)
let find_module root pkg name =
  if not (has_api root pkg) then None
  else
    let depth p = List.length (String.split_on_char '/' p.path) in
    let named = List.filter (fun p -> p.path <> "" && Filename.basename p.path = name) (pages root pkg) in
    match List.sort (fun a b -> compare (depth a) (depth b)) named with
    | [ p ] -> Some p.url
    | p :: q :: _ when depth p < depth q -> Some p.url
    | _ -> None

(* HTML text, unescaped: what odoc's <pre> holds. *)
let unescape s =
  let b = Buffer.create (String.length s) in
  let n = String.length s in
  let rec go i =
    if i >= n then ()
    else if s.[i] = '&' then
      match String.index_from_opt s i ';' with
      | Some j when j - i <= 8 ->
          let e = String.sub s (i + 1) (j - i - 1) in
          (match e with
          | "lt" -> Buffer.add_char b '<'
          | "gt" -> Buffer.add_char b '>'
          | "amp" -> Buffer.add_char b '&'
          | "quot" -> Buffer.add_char b '"'
          | "apos" -> Buffer.add_char b '\''
          | _ when String.length e > 1 && e.[0] = '#' -> (
              let code = if e.[1] = 'x' || e.[1] = 'X' then int_of_string_opt ("0x" ^ String.sub e 2 (String.length e - 2)) else int_of_string_opt (String.sub e 1 (String.length e - 1)) in
              match code with Some c -> Buffer.add_utf_8_uchar b (Uchar.of_int c) | None -> Buffer.add_string b ("&" ^ e ^ ";"))
          | _ -> Buffer.add_string b ("&" ^ e ^ ";"));
          go (j + 1)
      | _ ->
          Buffer.add_char b '&';
          go (i + 1)
    else (
      Buffer.add_char b s.[i];
      go (i + 1))
  in
  go 0;
  Buffer.contents b

let strip_tags s =
  let b = Buffer.create (String.length s) in
  let inside = ref false in
  String.iter (fun c -> if c = '<' then inside := true else if c = '>' then inside := false else if not !inside then Buffer.add_char b c) s;
  unescape (Buffer.contents b)

let find = Page.find_from

(* [s] with each [open_ ... close] span, [close] included, replaced by
   [f inner], [inner] what is between them. *)
let replace_spans s ~open_ ~close f =
  let b = Buffer.create (String.length s) in
  let n = String.length s in
  let rec go i =
    match find s i open_ with
    | None -> Buffer.add_string b (String.sub s i (n - i))
    | Some j -> (
        Buffer.add_string b (String.sub s i (j - i));
        let k = j + String.length open_ in
        match find s k close with
        | None -> Buffer.add_string b (String.sub s j (n - j))
        | Some l ->
            Buffer.add_string b (f (String.sub s k (l - k)));
            go (l + String.length close))
  in
  go 0;
  Buffer.contents b

let replace s ~sub ~by = replace_spans s ~open_:sub ~close:"" (fun _ -> by)

(* The site's URL for odoc's relative link [href] from the page [p], or
   None when the site has no such page. *)
let resolve ~exists p href =
  if href = "" || href.[0] = '#' || Markdown.has_scheme href then Some href
  else
    let path, frag = match String.index_opt href '#' with Some i -> (String.sub href 0 i, String.sub href i (String.length href - i)) | None -> (href, "") in
    let from = if p.path = "" then p.pkg else p.pkg ^ "/" ^ p.path in
    let target = Markdown.normalize (from / path) in
    let target = if Filename.basename target = "index.html" then Filename.dirname target else target in
    match String.split_on_char '/' target with
    | pkg :: rest ->
        let path = String.concat "/" rest in
        if exists pkg path then Some (url_of pkg path ^ frag) else None
    | [] -> None

let unlinked = "\000"

(* Each of odoc's empty anchor links, <a href="#ID" class="anchor"></a>,
   with a name, and "#" to show. *)
let name_anchors html =
  let tail = {|" class="anchor"></a>|} and head = {|<a href="#|} in
  let b = Buffer.create (String.length html) in
  let n = String.length html in
  let rec go i =
    match find html i tail with
    | None -> Buffer.add_string b (String.sub html i (n - i))
    | Some j ->
        (* The link's start: the last head before [j], with only an id
           between. *)
        let rec back k = if k < i then None else if String.sub html k (String.length head) = head then Some k else back (k - 1) in
        (match back (j - String.length head) with
        | Some k when not (String.contains (String.sub html (k + String.length head) (j - k - String.length head)) '"') ->
            let id = String.sub html (k + String.length head) (j - k - String.length head) in
            let name = match String.index_opt id '-' with Some x -> String.sub id (x + 1) (String.length id - x - 1) | None -> id in
            Buffer.add_string b (String.sub html i (k - i));
            Buffer.add_string b (sprintf {|<a href="#%s" class="anchor" aria-label="link to %s">#</a>|} id name)
        | _ -> Buffer.add_string b (String.sub html i (j + String.length tail - i)));
        go (j + String.length tail)
  in
  go 0;
  Buffer.contents b

let rewrite ~exists p html =
  (* Links: to the site's pages, or to their text. *)
  let html = Markdown.resolve_html (fun href -> match resolve ~exists p href with Some u -> u | None -> unlinked) html in
  let html =
    replace_spans html ~open_:(sprintf {|<a href="%s"|} unlinked) ~close:"</a>" (fun inner ->
        match String.index_opt inner '>' with Some i -> String.sub inner (i + 1) (String.length inner - i - 1) | None -> inner)
  in
  (* Headings: the docs' classes, and no empty anchor links. *)
  let html =
    List.fold_left
      (fun html (level, cls) ->
        replace_spans html ~open_:(sprintf "<h%d id=\"" level) ~close:"</a>" (fun inner ->
            (* [inner] is the heading's id, then its anchor link, up to its end. *)
            match String.index_opt inner '"' with
            | Some i when String.ends_with ~suffix:{|class="anchor">|} inner && not (String.contains inner '<' && Page.find_from inner 0 "</h" <> None) ->
                sprintf {|<h%d class="%s" id="%s">|} level cls (String.sub inner 0 i)
            | _ -> sprintf "<h%d id=\"%s</a>" level inner))
      html
      [ (2, "h2-doc"); (3, "h3-doc"); (4, "h4-doc") ]
  in
  (* Anchors of specifications: a link with a name. *)
  let html = name_anchors html in
  (* The name a value specification binds, as code binds names. *)
  let html =
    replace_spans html ~open_:{|<span class="keyword">val</span> |} ~close:" :" (fun name ->
        if String.contains name '<' || String.contains name '\n' then sprintf {|<span class="keyword">val</span> %s :|} name
        else sprintf {|<span class="keyword">val</span> <span class="hl-b">%s</span> :|} name)
  in
  (* OCaml in comments, highlighted; other preformatted text, as is. *)
  let html =
    replace_spans html ~open_:{|<pre class="language-ocaml"><code>|} ~close:"</code></pre>" (fun code ->
        sprintf {|<figure class="code"><figcaption><span>ocaml</span></figcaption><pre tabindex="0" role="region" aria-label="OCaml code">%s</pre></figure>|}
          (Highlight.to_html (unescape code)))
  in
  let text t = sprintf {|<figure class="code"><pre tabindex="0" role="region" aria-label="Text">%s</pre></figure>|} t in
  let html = replace_spans html ~open_:"<pre><code>" ~close:"</code></pre>" text in
  replace_spans html ~open_:"<pre>" ~close:"</pre>" text

type rendered = {
  title : string;  (** "Module Jera.Linear". *)
  lead : string option;  (** HTML, the module's synopsis. *)
  summary : string;
  crumbs : (string * string option) list;  (** After the package's. *)
  body : string;
  contents : (string * string * (string * string) list) list;  (** id, text, children. *)
}

let rec toc_entries entries =
  List.filter_map
    (fun e ->
      match Json.member "href" e with
      | Some (Json.String href) when String.length href > 1 && href.[0] = '#' ->
          let children = toc_entries (Json.list_member "children" e) |> List.map (fun (id, t, _) -> (id, t)) in
          Some (String.sub href 1 (String.length href - 1), strip_tags (match Json.member "title" e with Some (Json.String t) -> t | _ -> ""), children)
      | _ -> None)
    entries

(* The package's index: its libraries, each with its modules and their
   synopses. *)
let render_index ~root ~exists p =
  let file, json = read root p in
  let content = Json.string_member ~file "content" json in
  let libraries =
    String.split_on_char '\000' (replace content ~sub:{|<h2 id="library-|} ~by:"\000") |> List.tl
    |> List.map (fun section ->
           let name = String.sub section 0 (String.index section '"') in
           let modules = ref [] in
           ignore
             (replace_spans section ~open_:{|<a href="|} ~close:{|"|} (fun href ->
                  (if Filename.check_suffix href "/index.html" then
                     let m = Filename.dirname href in
                     if exists p.pkg m && not (List.mem m !modules) then modules := m :: !modules);
                  ""));
           (name, List.rev !modules))
  in
  let synopsis m =
    let file, json = read root { p with path = m } in
    let preamble = Json.string_member ~file "preamble" json in
    match find preamble 0 "<p>" with
    | Some 0 -> ( match find preamble 0 "</p>" with Some j -> strip_tags (String.sub preamble 3 (j - 3)) | None -> "")
    | _ -> ""
  in
  let libraries = List.filter (fun (_, ms) -> ms <> []) libraries in
  let row (library, m) =
    sprintf {|<li class="apirow"><span class="caption mono">%s</span><a href="%s" class="mono-ui u">%s</a><span class="small">%s</span></li>|}
      (esc library) (url_of p.pkg m) m (esc (synopsis m))
  in
  {
    title = "API reference";
    lead = None;
    summary = sprintf "The libraries and modules of %s." p.pkg;
    crumbs = [ ("api", None) ];
    body =
      sprintf {|<ul class="rows apirows" aria-label="Modules"><li class="apirow apihead" aria-hidden="true"><span class="caption">library</span><span class="caption">module</span><span class="caption">about</span></li>%s</ul>|}
        (concat_map row (List.concat_map (fun (l, ms) -> List.map (fun m -> (l, m)) ms) libraries));
    contents = [];
  }

let render_module ~root ~exists p =
  let file, json = read root p in
  let field k = Json.string_member ~file k json in
  if Json.member "uses_katex" json = Some (Json.Bool true) then Repo.fail "%s: math needs KaTeX, which the site doesn't load" file;
  let preamble = rewrite ~exists p (field "preamble") in
  let lead, preamble =
    match (find preamble 0 "<p>", find preamble 0 "</p>") with
    | Some 0, Some j -> (Some (String.sub preamble 3 (j - 3)), String.sub preamble (j + 4) (String.length preamble - j - 4))
    | _ -> (None, preamble)
  in
  let crumbs =
    Json.list_member "breadcrumbs" json
    |> List.filter_map (fun c ->
           match (Json.member "kind" c, Json.member "name" c, Json.member "href" c) with
           | Some (Json.String ("page" | "leaf-page")), _, _ -> None
           | _, Some (Json.String name), Some (Json.String href) ->
               Some (esc name, if href = "#" then None else resolve ~exists p href)
           | _, Some (Json.String name), _ -> Some (esc name, None)
           | _ -> None)
  in
  {
    title = strip_tags (field "header");
    lead;
    summary = (match lead with Some l -> strip_tags l | None -> strip_tags (field "header"));
    crumbs = ("api", Some (url_of p.pkg "")) :: crumbs;
    body = preamble ^ rewrite ~exists p (field "content");
    contents = toc_entries (Json.list_member "toc" json);
  }

(* Each page of the package's API, rendered. A link goes to the API of a
   package in [shown], and loses its link otherwise. *)
let render root ~shown pkg =
  let pages = pages root pkg in
  let exists pkg' path =
    if pkg' = pkg then List.exists (fun q -> q.path = path) pages
    else List.mem pkg' shown && has_api root pkg' && Sys.file_exists (dir root / pkg' / path / "index.html.json")
  in
  List.map (fun p -> (p, if p.path = "" then render_index ~root ~exists p else render_module ~root ~exists p)) pages

(* The index: each module, and each value, type, exception and class it
   specifies, with the first sentence of its documentation. *)

type item = {
  name : string;  (** "dense". *)
  qualified : string;  (** "Jera.Linear.dense". *)
  url : string;
  kind : string;  (** "val", "type", "module"… *)
  about : string;
}

let first_sentence s =
  let n = String.length s in
  let rec go i = if i + 1 >= n then s else if s.[i] = '.' && (s.[i + 1] = ' ' || s.[i + 1] = '\n') then String.sub s 0 (i + 1) else go (i + 1) in
  go 0

(* The text of the first paragraph of [html], if it starts one. *)
let first_paragraph html =
  match (find html 0 "<p>", find html 0 "</p>") with
  | Some i, Some j when j > i -> Some (String.trim (strip_tags (String.sub html (i + 3) (j - i - 3))))
  | _ -> None

(* "Rig_edge/module-type-Driver" is "Rig_edge.Driver". *)
let module_path path =
  String.split_on_char '/' path
  |> List.map (fun s -> if String.starts_with ~prefix:"module-type-" s then String.sub s 12 (String.length s - 12) else s)
  |> String.concat "."

let items root pkg =
  List.concat_map
    (fun p ->
      if p.path = "" then []
      else
        let file, json = read root p in
        let content = Json.string_member ~file "content" json in
        let m = module_path p.path in
        let synopsis = match first_paragraph (Json.string_member ~file "preamble" json) with Some s -> first_sentence s | None -> "" in
        let kind_of = function
          | "value" -> Some "val"
          | "type" -> Some "type"
          | "exception" -> Some "exception"
          | "class" -> Some "class"
          | "class-type" -> Some "class type"
          | _ -> None
        in
        (* Each <div class="spec KIND anchored" id="ID">, and the first
           paragraph of the spec-doc after it, before the next spec. *)
        let rec specs i acc =
          match find content i {|<div class="spec |} with
          | None -> List.rev acc
          | Some j -> (
              let k = j + String.length {|<div class="spec |} in
              let cls_end = Option.value (String.index_from_opt content k '"') ~default:k in
              let cls = String.sub content k (cls_end - k) in
              let next = match find content cls_end {|<div class="spec |} with Some x -> x | None -> String.length content in
              match (kind_of (List.hd (String.split_on_char ' ' cls)), find content cls_end {| id="|}) with
              | Some kind, Some idp when idp < next && not (Repo.contains cls "subst") ->
                  let ids = idp + 5 in
                  let id = String.sub content ids (String.index_from content ids '"' - ids) in
                  let name = match String.index_opt id '-' with Some x -> String.sub id (x + 1) (String.length id - x - 1) | None -> id in
                  let about =
                    match find content cls_end {|<div class="spec-doc">|} with
                    | Some d when d < next -> ( match first_paragraph (String.sub content d (next - d)) with Some s -> first_sentence s | None -> "")
                    | _ -> ""
                  in
                  specs next ({ name; qualified = m ^ "." ^ name; url = p.url ^ "#" ^ id; kind; about } :: acc)
              | _ -> specs next acc)
        in
        { name = List.hd (List.rev (String.split_on_char '.' m)); qualified = m; url = p.url; kind = (if Repo.contains p.path "module-type-" then "module type" else "module"); about = synopsis }
        :: specs 0 [])
    (pages root pkg)
