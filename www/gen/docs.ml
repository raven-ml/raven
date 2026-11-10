(* The docs: the manual, and each library's page and guides, rendered
   from the repository's Markdown into the docs layout of A″ v4, inside
   the head, header and footer of pages/docs.html.

   - doc/manual/<name>.md is /docs/manual/<name>/.
   - A library's doc/index.md is /docs/<lib>/; a contrib package without
     one has its README.md there. Its guides, doc/NN-<slug>.md, are
     /docs/<lib>/<slug>/, and its other Markdown, doc/<name>.md, is
     /docs/<lib>/<name>/.
   - The sidebar lists the manual and every library. The current library
     opens to its guides, each named as the library's page links it, or
     else by its title.
   - A page reads, in order, with a pager: the manual, then each library,
     its page then its guides. *)

open Html

let ( / ) = Filename.concat

type lib = {
  name : string;
  dir : string;  (** In the repository: packages/nx, contrib/fehu. *)
  glyph : string;  (** HTML: its rune, or an empty slot of the same width. *)
  rune : bool;  (** Whether it has a rune. *)
  label : string;  (** "foundation · like JAX". *)
  landing : string;  (** The source of its page. *)
  guides : string list;  (** doc/NN-*.md, in order. *)
  others : string list;  (** Its other Markdown. *)
  examples : (string * string option) list;  (** Each example, and what it shows. *)
  api : bool;  (** Whether it has an API reference. *)
  depends : string list;
}

type kind = Manual | Landing of lib | Guide of lib * int | Other of lib
type page = { file : string; url : string; kind : kind }

let is_guide n =
  String.length n > 3 && Repo.is_digit n.[0] && Repo.is_digit n.[1] && n.[2] = '-' && Filename.check_suffix n ".md"

let lib ~root ~name ~dir ~glyph ~rune ~label ~depends =
  let docdir = dir / "doc" in
  let md =
    if Sys.file_exists (root / docdir) then
      Sys.readdir (root / docdir) |> Array.to_list |> List.filter (fun n -> Filename.check_suffix n ".md") |> List.sort compare
    else []
  in
  let landing = if List.mem "index.md" md then Some (docdir / "index.md") else if Sys.file_exists (root / dir / "README.md") then Some (dir / "README.md") else None in
  Option.map
    (fun landing ->
      {
        name;
        dir;
        glyph;
        rune;
        label;
        landing;
        guides = List.filter is_guide md |> List.map (fun n -> docdir / n);
        others = List.filter (fun n -> n <> "index.md" && not (is_guide n)) md |> List.map (fun n -> docdir / n);
        examples = Repo.example_list (root / dir / "examples");
        api = Api.has_api root name;
        depends;
      })
    landing

(* The core in the registry's order, then contrib. *)
let libs root =
  let core = Repo.core root in
  let core_lib (e : Registry.entry) =
    let group = fst (List.find (fun (_, es) -> List.memq e es) Registry.groups) in
    let p = List.find (fun (p : Repo.package) -> p.name = e.name) core in
    lib ~root ~name:e.name ~dir:("packages" / e.name) ~glyph:(glyph e) ~rune:(e.glyph <> None) ~label:(sprintf "%s · like %s" group (esc e.like)) ~depends:p.depends
  in
  let contrib (p : Repo.package) =
    let rune = Option.join (List.assoc_opt p.name Registry.contrib) |> Option.value ~default:"" in
    lib ~root ~name:p.name ~dir:("contrib" / p.name)
      ~glyph:(sprintf {|<span class="glyph" style="color: var(--dim)" aria-hidden="true">%s</span>|} rune)
      ~rune:(rune <> "") ~label:"contrib" ~depends:p.depends
  in
  List.filter_map core_lib Registry.entries @ List.filter_map contrib (Repo.contrib root)

(* "02-derivatives.md" is "derivatives". *)
let slug file =
  let n = Filename.basename file in
  let n = if is_guide n then String.sub n 3 (String.length n - 3) else n in
  Filename.remove_extension n

let pages root =
  let manual = List.map (fun n -> { file = "doc/manual/" ^ n ^ ".md"; url = "/docs/manual/" ^ n ^ "/"; kind = Manual }) Registry.manual in
  let lib l =
    let at file = sprintf "/docs/%s/%s/" l.name (slug file) in
    ({ file = l.landing; url = sprintf "/docs/%s/" l.name; kind = Landing l } :: List.mapi (fun i g -> { file = g; url = at g; kind = Guide (l, i + 1) }) l.guides)
    @ List.map (fun o -> { file = o; url = at o; kind = Other l }) l.others
  in
  let pages = manual @ List.concat_map lib (libs root) in
  List.iter
    (fun p ->
      match List.filter (fun q -> q.url = p.url) pages with
      | [ _ ] -> ()
      | q :: _ -> Repo.fail "%s and %s would both be %s" q.file p.file p.url
      | [] -> ())
    pages;
  pages

(* Every link in the document at [file], as its text and destination. *)
let links ~root file =
  let doc = Cmarkit.Doc.of_string ~strict:false (Repo.read (root / file)) in
  let defs = Cmarkit.Doc.defs doc in
  let inline _ acc = function
    | Cmarkit.Inline.Link (l, _) -> (
        match Cmarkit.Inline.Link.reference_definition defs l with
        | Some (Cmarkit.Link_definition.Def (d, _)) -> (
            match Cmarkit.Link_definition.dest d with
            | Some (dest, _) -> Cmarkit.Folder.ret ((Repo.plain (Cmarkit.Inline.Link.text l), dest) :: acc)
            | None -> Cmarkit.Folder.default)
        | _ -> Cmarkit.Folder.default)
    | _ -> Cmarkit.Folder.default
  in
  List.rev (Cmarkit.Folder.fold_doc (Cmarkit.Folder.make ~inline ()) [] doc)

(* The sidebar's name for each page. *)
let names ~root pages =
  let titles = List.map (fun p -> (p.file, Repo.title (root / p.file))) pages in
  let named = Hashtbl.create 64 in
  List.iter
    (fun p ->
      match p.kind with
      | Landing l ->
          List.iter
            (fun (text, dest) ->
              let target = Markdown.normalize (Filename.dirname l.landing / dest) in
              if List.mem target l.guides && not (Hashtbl.mem named target) then Hashtbl.add named target text)
            (links ~root l.landing)
      | _ -> ())
    pages;
  fun file -> match Hashtbl.find_opt named file with Some n -> n | None -> List.assoc file titles

let source l = sprintf "%s/tree/main/%s" github l.dir

(* The sidebar, on the page at [url] of the library [lib], if any. *)
let sidebar ~aria ~pages ~name ~lib:current_lib ~url =
  let here u = if u = url then {| aria-current="page"|} else "" in
  let manual =
    concat_map (fun p -> match p.kind with Manual -> sprintf {|<a href="%s" style="display: block"%s>%s</a>|} p.url (here p.url) (esc (name p.file)) | _ -> "") pages
  in
  let lib l =
    let link = sprintf {|<a href="/docs/%s/" class="lib"%s>%s%s</a>|} l.name (here (sprintf "/docs/%s/" l.name)) l.glyph l.name in
    if current_lib <> Some l.name then link
    else
      let guides =
        concat_map
          (fun p -> match p.kind with Guide (g, _) when g.name = l.name -> sprintf {|<a href="%s" class="sub"%s>%s</a>|} p.url (here p.url) (esc (name p.file)) | _ -> "")
          pages
      in
      let examples =
        match l.examples with [] -> "" | xs -> sprintf {|<a href="/docs/%s/#examples" class="subm">examples (%d)</a>|} l.name (List.length xs)
      in
      let api =
        if not l.api then ""
        else
          let u = sprintf "/docs/%s/api/" l.name in
          let inside = String.starts_with ~prefix:u url in
          sprintf {|<a href="%s" class="subm"%s>api</a>|} u (if u = url then here u else if inside then {| aria-current="location"|} else "")
      in
      sprintf {|<div class="open">%s%s%s%s<a href="%s" class="subm">source ↗</a></div>|} link guides examples api (source l)
  in
  let libs = List.filter_map (fun p -> match p.kind with Landing l -> Some l | _ -> None) pages in
  let core, contrib = List.partition (fun l -> l.label <> "contrib") libs in
  let group title body = sprintf {|<div class="group"><p class="label" style="margin-bottom: 8px">%s</p>%s</div>|} title body in
  sprintf {|<nav class="sidebar" aria-label="%s">%s%s%s</nav>|} aria (group "manual" manual) (group "libraries" (concat_map lib core))
    (if contrib = [] then "" else group "contrib" (concat_map lib contrib))

let crumbs parts =
  let n = List.length parts in
  let part i (text, url) =
    if i = n - 1 then sprintf {|<span aria-current="page">%s</span>|} text
    else match url with Some u -> sprintf {|<a href="%s">%s</a>|} u text | None -> text
  in
  sprintf {|<nav aria-label="Breadcrumb" class="crumbs">%s</nav>|}
    (String.concat {|<span aria-hidden="true"> / </span>|} (List.mapi part (("docs", Some "/docs/") :: parts)))

(* The contents: (id, text, its children). *)
let toc contents =
  let link cls (id, text) = sprintf {|<a href="#%s"%s>%s</a>|} (esc id) cls (esc text) in
  match contents with
  | [] -> ""
  | cs ->
      sprintf {|<nav class="toc" aria-label="On this page"><p class="label" style="padding-left: 16px; margin-bottom: 8px">on this page</p>%s</nav>|}
        (concat_map (fun (id, text, children) -> link "" (id, text) ^ concat_map (link {| class="sub"|}) children) cs)

(* The library's tabs; [at] is the one the page is under. *)
let ptabs l ~at =
  let tab name href = sprintf {|<a href="%s"%s>%s</a>|} href (if at = Some name then {| aria-current="page"|} else "") name in
  sprintf {|<nav aria-label="%s sections" class="ptabs">%s%s%s<a href="%s">source ↗</a></nav>|} l.name
    (tab "overview" (sprintf "/docs/%s/" l.name))
    (if l.api then tab "api" (sprintf "/docs/%s/api/" l.name) else "")
    (if l.examples = [] then "" else sprintf {|<a href="/docs/%s/#examples">examples</a>|} l.name)
    (source l)

(* A page of the docs: [main] between the sidebar and the rail, in the
   head, header and footer of [layout]. *)
let frame ~layout ~pages ~name ~lib ~url ~title ~description ~crumbs ~main ~rail =
  let html =
    sprintf
      {|<div class="docs">
%s
<main id="content" class="article prose md">
%s
<details class="docmenu"><summary>docs menu</summary>%s</details>
%s
</main>
<aside class="rail" aria-label="On this page">%s</aside>
</div>|}
      (sidebar ~aria:"Documentation" ~pages ~name ~lib ~url)
      crumbs
      (sidebar ~aria:"Documentation, menu" ~pages ~name ~lib ~url)
      main rail
  in
  Page.with_main layout ~title:(title ^ " — raven") ~description (`Around html)

let first_sentence s =
  let n = String.length s in
  let rec go i = if i + 1 >= n then s else if s.[i] = '.' && s.[i + 1] = ' ' then String.sub s 0 (i + 1) else go (i + 1) in
  go 0

let facts ~libs l =
  let on = List.filter (fun d -> List.exists (fun m -> m.name = d) libs) l.depends in
  sprintf {|<dl class="facts"><dt>install</dt><dd>opam install %s</dd>%s</dl>|} l.name
    (match on with
    | [] -> ""
    | on -> sprintf "<dt>built on</dt><dd>%s</dd>" (String.concat " · " (List.map (fun d -> sprintf {|<a href="/docs/%s/">%s</a>|} d d) on)))

let examples_section l =
  let row (name, about) =
    sprintf {|<li class="row meta exrow"><a href="%s/examples/%s" class="mono-ui u">%s ↗</a><span class="small">%s</span></li>|}
      (source l) name name (match about with Some a -> esc (first_sentence a) | None -> "")
  in
  sprintf {|<h2 id="examples" class="h2-doc">Examples</h2>
<ul class="rows">%s</ul>
|} (concat_map row l.examples)

(* Each page as (its URL, its HTML, what Markdown.render found in it). *)
let build ~root ~layout ~pages env =
  let libs = List.filter_map (fun p -> match p.kind with Landing l -> Some l | _ -> None) pages in
  let name = names ~root pages in
  let sequence = List.filter (fun p -> match p.kind with Other _ -> false | _ -> true) pages in
  let rec neighbours prev = function
    | p :: (next :: _ as rest) -> (p.url, (prev, Some next)) :: neighbours (Some p) rest
    | [ p ] -> [ (p.url, (prev, None)) ]
    | [] -> []
  in
  let neighbours = neighbours None sequence in
  let pager_label p = match p.kind with Landing l -> sprintf {|<span class="mono" style="color: var(--text-0)">%s</span>|} l.name | Guide (_, i) -> sprintf {|<span style="color: var(--text-0)">%02d · %s</span>|} i (esc (name p.file)) | _ -> sprintf {|<span style="color: var(--text-0)">%s</span>|} (esc (name p.file)) in
  let pager p =
    match List.assoc_opt p.url neighbours with
    | None | Some (None, None) -> ""
    | Some (prev, next) ->
        let link cls caption = function Some q -> sprintf {|<a href="%s"%s><span class="caption">%s</span>%s</a>|} q.url cls caption (pager_label q) | None -> "<span></span>" in
        sprintf {|<nav aria-label="Pages" class="pager">%s%s</nav>|} (link "" "← previous" prev) (link {| class="next"|} "next →" next)
  in
  let render p =
    let md = Markdown.render ~lead:true env p.file in
    let tested = List.mem (Filename.basename p.file) (Repo.mdx_files (root / Filename.dirname p.file)) in
    let editline =
      sprintf {|<p class="caption editline"><span>%s</span><a class="u" href="%s/edit/main/%s">edit this page ↗</a></p>|}
        (if tested then "this page is tested with mdx" else "") github p.file
    in
    let lead = match md.lead with Some l -> sprintf {|<p class="lead" style="color: var(--text-2)">%s</p>|} l | None -> "" in
    let crumbs, header, extra, contents, rail_extra =
      match p.kind with
      | Manual ->
          ( crumbs [ ("manual", None); (esc md.title, None) ],
            sprintf {|<header class="stack" style="gap: 14px"><h1 id="%s" class="h1-doc">%s</h1>%s</header>|} (esc md.title_id) (esc md.title) lead,
            "", md.contents, "" )
      | Guide (l, _) | Other l ->
          ( crumbs [ (l.name, Some (sprintf "/docs/%s/" l.name)); (esc (name p.file), None) ],
            sprintf {|<header class="stack" style="gap: 14px"><h1 id="%s" class="h1-doc">%s</h1>%s%s</header>|} (esc md.title_id) (esc md.title) lead (ptabs l ~at:None),
            "", md.contents, "" )
      | Landing l ->
          let examples = if l.examples = [] || List.mem "examples" md.ids then "" else examples_section l in
          ( crumbs [ (l.name, None) ],
            sprintf
              {|<header class="stack" style="gap: 20px">
<div class="libhead">%s<div class="stack" style="gap: 2px"><h1 id="%s" class="h1-doc mono">%s</h1><span class="label">%s</span></div></div>
%s%s
</header>
<div class="phone-only">%s</div>|}
              (if l.rune then sprintf {|<span class="libmark">%s</span>|} l.glyph else "") (esc md.title_id) l.name l.label lead (ptabs l ~at:(Some "overview")) (facts ~libs l),
            examples,
            (md.contents @ if examples = "" then [] else [ ("examples", "Examples") ]),
            sprintf {|<div class="card" style="padding: 18px; gap: 10px">%s</div>|} (facts ~libs l) )
    in
    let lib = match p.kind with Manual -> None | Landing l | Guide (l, _) | Other l -> Some l.name in
    let title = match p.kind with Manual -> md.title | Landing l -> l.name | Guide (l, _) | Other l -> sprintf "%s · %s" md.title l.name in
    let contents = List.map (fun (id, text) -> (id, text, [])) contents in
    ( p,
      frame ~layout ~pages ~name ~lib ~url:p.url ~title ~description:md.summary ~crumbs
        ~main:(String.concat "\n" [ header; md.body ^ extra; pager p; editline ])
        ~rail:(toc contents ^ rail_extra),
      md )
  in
  List.map render pages

(* The API reference of each library that has one, as (URL, HTML). *)
let build_api ~root ~layout ~pages =
  let name = names ~root pages in
  let libs = List.filter_map (fun p -> match p.kind with Landing l when l.api -> Some l | _ -> None) pages in
  List.concat_map
    (fun l ->
      List.map
        (fun ((p : Api.page), (r : Api.rendered)) ->
          let lead = match r.lead with Some l -> sprintf {|<p class="lead" style="color: var(--text-2)">%s</p>|} l | None -> "" in
          let header =
            sprintf {|<header class="stack" style="gap: 14px"><h1 class="h1-doc%s">%s</h1>%s%s</header>|}
              (if p.path = "" then "" else " mono api-title") (esc r.title) lead
              (ptabs l ~at:(Some "api"))
          in
          let title = if p.path = "" then sprintf "API · %s" l.name else sprintf "%s · %s" r.title l.name in
          ( p.url,
            frame ~layout ~pages ~name ~lib:(Some l.name) ~url:p.url ~title ~description:r.summary
              ~crumbs:(crumbs ((l.name, Some (sprintf "/docs/%s/" l.name)) :: r.crumbs))
              ~main:(header ^ sprintf {|
<div class="api">%s</div>|} r.body)
              ~rail:(toc r.contents) ))
        (Api.render root ~shown:(List.map (fun l -> l.name) libs) l.name))
    libs

(* The index, A to Z: every page of the docs, every example, and every
   module and item of the API, each with its library's glyph. *)

type entry = {
  name : string;  (** What it is sorted and found by. *)
  qualified : string;  (** What it is shown as. *)
  url : string;
  kind : string;
  lib : lib option;
  about : string;
}

let index ~root =
  let pages = pages root in
  let name = names ~root pages in
  let about file =
    match
      List.find_map
        (function Cmarkit.Block.Paragraph (p, _) -> Some (Repo.plain (Cmarkit.Block.Paragraph.inline p)) | _ -> None)
        (Repo.blocks (root / file))
    with
    | Some s -> Api.first_sentence s
    | None -> ""
  in
  let page p =
    let e kind lib = { name = name p.file; qualified = name p.file; url = p.url; kind; lib; about = about p.file } in
    match p.kind with
    | Manual -> [ e "manual" None ]
    | Guide (l, _) -> [ e "guide" (Some l) ]
    | Other l -> [ e "page" (Some l) ]
    | Landing l ->
        ({ name = l.name; qualified = l.name; url = p.url; kind = "library"; lib = Some l; about = about p.file }
         :: List.map
              (fun (x, a) ->
                { name = x; qualified = x; url = sprintf "%s/examples/%s" (source l) x; kind = "example"; lib = Some l; about = Option.value (Option.map first_sentence a) ~default:"" })
              l.examples)
        @
        if l.api then
          List.map
            (fun (i : Api.item) -> { name = i.name; qualified = i.qualified; url = i.url; kind = i.kind; lib = Some l; about = i.about })
            (Api.items root l.name)
        else []
  in
  List.concat_map page pages
