(* What a page's slots are filled with: the parts of a page that must
   follow the repository rather than be written by hand.

   - [release]: the newest version in CHANGES.md, as "beta 1".
   - [library-count]: how many core libraries there are, in words.
   - [tabs]: the landing page's code tabs, one per library in
     [Registry.tabs], each the first tested block of its doc/index.md, or
     of the section of a guide the registry names.
   - [excerpt FILE NAME]: the top-level definition NAME in FILE, verbatim
     and highlighted.
   - [libraries]: the landing page's cards, the core by group, then
     contrib.
   - [libraries index]: the docs landing's compact index: a row per
     library, with its guides and examples counted.
   - [posts N] and [posts all]: the latest N posts, or all of them.
   - [index]: the index, A to Z, of the docs and the API.
   - [uses LIB...]: the libraries a project uses, each its glyph and name,
     linked to its docs. *)

open Html

let ( / ) = Filename.concat

let number_word n =
  let words =
    [| "zero"; "one"; "two"; "three"; "four"; "five"; "six"; "seven"; "eight"; "nine"; "ten"; "eleven";
       "twelve"; "thirteen"; "fourteen"; "fifteen"; "sixteen"; "seventeen"; "eighteen"; "nineteen"; "twenty" |]
  in
  if n < Array.length words then String.capitalize_ascii words.(n) else string_of_int n

let plural n one many = sprintf "%d %s" n (if n = 1 then one else many)

(* Without JS, the panels show one under another, each under its heading. *)
let tabs ~root =
  let tab (name, what, source) =
    let e = Registry.find name in
    let doc = root / "packages" / name / "doc" in
    let code =
      match source with
      | None -> Repo.tested_code (doc / "index.md")
      | Some (file, section) -> Repo.tested_code ~section (doc / file)
    in
    sprintf
      {|<section class="panel" id="tab-%s">
<h3 class="panel-h">%s%s</h3>
<pre class="panel-code" tabindex="0" role="region" aria-label="%s example">%s</pre>
<div class="panel-foot caption"><span class="hide-sm">%s · like %s</span><a href="/docs/%s/" class="u">%s docs →</a></div>
</section>|}
      name (glyph e) name name (Highlight.to_html code) what (esc e.like) name name
  in
  concat_map tab Registry.tabs

let libraries ~root =
  let card (e : Registry.entry) =
    sprintf
      {|<a href="/docs/%s/" class="card lib"><div class="glyph-row"><span class="lib-glyph">%s</span><span class="caption">like %s</span></div><div class="stack" style="gap: 6px; flex: 1"><span class="mono h4">%s</span><span class="small">%s</span></div></a>|}
      e.name (glyph e) (esc e.like) e.name (esc e.about)
  in
  let group (title, entries) =
    sprintf {|
<div class="group-row"><h3 class="label group-title">%s</h3><div class="grid-auto">%s</div></div>|}
      title (concat_map card entries)
  in
  let contrib (p : Repo.package) =
    let g = match List.assoc_opt p.name Registry.contrib with Some g -> g | None -> Repo.fail "www/gen/registry.ml: no contrib entry for %s" p.name in
    let href =
      if Sys.file_exists (root / "contrib" / p.name / "doc" / "index.md") then sprintf "/docs/%s/" p.name
      else sprintf "%s/tree/main/contrib/%s" github p.name
    in
    sprintf {|<a href="%s" class="contrib"><span class="glyph" style="color: var(--dim); font-size: 20px" aria-hidden="true">%s</span><span class="mono">%s</span><span class="small">%s</span></a>|}
      href (Option.value g ~default:"") p.name (esc p.synopsis)
  in
  sprintf
    {|<div class="stack groups">%s
<div class="group-row"><div class="stack group-title" style="gap: 6px"><h3 class="label">contrib</h3><p class="caption">Built against every change to the core; released on their own schedule.</p></div><div class="grid-auto">%s</div></div>
</div>|}
    (concat_map group Registry.groups) (concat_map contrib (Repo.contrib root))

(* The docs landing's index of libraries: one row each, grouped, with what
   it is, its Python counterpart and how many guides and examples it has,
   so the right library is found without scrolling. Contrib comes last. *)
let library_index ~root =
  let count dir =
    match (Repo.guides (dir / "doc"), Repo.examples (dir / "examples")) with
    | 0, 0 -> "1 page"
    | 0, x -> plural x "example" "examples"
    | g, 0 -> plural g "guide" "guides"
    | g, x -> plural g "guide" "guides" ^ " · " ^ plural x "example" "examples"
  in
  let row ~href ~glyph ~name ~about ~like ~count =
    sprintf {|
<li><a href="%s" class="librow">%s<span class="mono librow-name">%s</span><span class="librow-about">%s</span><span class="caption librow-like">%s</span><span class="caption librow-count">%s</span></a></li>|}
      href glyph name about like count
  in
  let core (e : Registry.entry) =
    row ~href:(sprintf "/docs/%s/" e.name) ~glyph:(glyph e) ~name:e.name ~about:(esc e.about)
      ~like:("like " ^ esc e.like) ~count:(count (root / "packages" / e.name))
  in
  let contrib (p : Repo.package) =
    let g = match List.assoc_opt p.name Registry.contrib with Some g -> g | None -> Repo.fail "www/gen/registry.ml: no contrib entry for %s" p.name in
    let dir = root / "contrib" / p.name in
    let href = if Sys.file_exists (dir / "doc" / "index.md") then sprintf "/docs/%s/" p.name else sprintf "%s/tree/main/contrib/%s" github p.name in
    row ~href ~glyph:(sprintf {|<span class="glyph" style="color: var(--dim)" aria-hidden="true">%s</span>|} (Option.value g ~default:""))
      ~name:p.name ~about:(esc p.synopsis) ~like:"" ~count:(count dir)
  in
  let group title rows = sprintf {|
<div class="libgroup"><h3 class="label">%s</h3><ul>%s
</ul></div>|} title rows in
  sprintf {|<div class="libindex">%s%s
</div>|}
    (concat_map (fun (title, entries) -> group title (concat_map core entries)) Registry.groups)
    (group "contrib" (concat_map contrib (Repo.contrib root)))

(* [Some n]: the latest n, as the landing page lists them; [None]: all of
   them, as the blog lists them, each title a heading. *)
let posts ~root n =
  let posts = Repo.posts root in
  let blurb (p : Repo.post) = match p.blurb with Some b -> sprintf {|<span class="small" style="max-width: 36rem">%s</span>|} (esc b) | None -> "" in
  match n with
  | Some n ->
      let post (p : Repo.post) =
        sprintf {|
<li><a href="/blog/%s/" class="row meta"><span class="label">%s</span><span class="stack" style="gap: 6px"><span class="h3">%s</span>%s</span></a></li>|}
          p.slug p.date (esc p.title) (blurb p)
      in
      sprintf {|<ul class="rows">%s
</ul>|} (concat_map post (List.filteri (fun i _ -> i < n) posts))
  | None ->
      let post (p : Repo.post) =
        sprintf {|
<li><a href="/blog/%s/" class="row meta" style="--meta-w: 120px; padding: 28px 0"><span class="label" style="line-height: 30px">%s</span><span class="stack" style="gap: 6px"><h2 class="h3">%s</h2>%s</span></a></li>|}
          p.slug p.date (esc p.title) (blurb p)
      in
      sprintf {|<ol reversed class="rows">%s
</ol>|} (concat_map post posts)

(* The index, A to Z, of the docs and the API: a group per letter, each
   entry its name, what it is, and its library. The search page filters
   it; without JS it is the whole list. *)
let index ~root =
  let entries = Docs.index ~root in
  let letter (e : Docs.entry) =
    match Char.lowercase_ascii (if e.name = "" then '#' else e.name.[0]) with 'a' .. 'z' as c -> String.make 1 c | _ -> "#"
  in
  let compare_entries (a : Docs.entry) (b : Docs.entry) =
    compare (String.lowercase_ascii a.name, String.lowercase_ascii a.qualified) (String.lowercase_ascii b.name, String.lowercase_ascii b.qualified)
  in
  let entries = List.sort compare_entries entries in
  (* What a library is for, so that "dataframe join" finds talon's joins. *)
  let topic lib =
    match List.find_opt (fun (r : Registry.entry) -> r.name = lib) (List.concat_map snd Registry.groups) with
    | Some r -> r.like ^ " " ^ r.about
    | None -> ""
  in
  let item (e : Docs.entry) =
    let lib = match e.lib with Some l -> l.name | None -> "" in
    let path, last =
      match String.rindex_opt e.qualified '.' with
      | Some i when e.qualified <> e.name && i + 1 < String.length e.qualified -> (String.sub e.qualified 0 (i + 1), String.sub e.qualified (i + 1) (String.length e.qualified - i - 1))
      | _ -> ("", e.qualified)
    in
    sprintf
      {|<li data-n="%s" data-k="%s"><a href="%s" class="azitem"><span class="azglyph">%s</span><span class="stack" style="gap: 2px; min-width: 0"><span class="mono-ui az-name"><span class="az-path">%s</span>%s</span>%s<span class="caption">%s%s</span></span></a></li>|}
      (esc (String.lowercase_ascii e.name))
      (esc (String.lowercase_ascii (String.concat " " [ e.qualified; e.kind; lib; topic lib; e.about ])))
      (esc e.url)
      (match e.lib with Some l -> l.glyph | None -> {|<span class="glyph" aria-hidden="true"></span>|})
      (esc path) (esc last)
      (if e.about = "" then "" else sprintf {|<span class="small">%s</span>|} (esc e.about))
      (esc e.kind) (if lib = "" then "" else " · " ^ esc lib)
  in
  let groups =
    List.fold_left
      (fun acc e -> match acc with (l, es) :: rest when l = letter e -> (l, e :: es) :: rest | _ -> (letter e, [ e ]) :: acc)
      [] entries
    |> List.rev_map (fun (l, es) -> (l, List.rev es))
    |> List.sort (fun (a, _) (b, _) -> compare (a = "#", a) (b = "#", b))
  in
  let group (l, es) =
    let id = if l = "#" then "az-symbols" else "az-" ^ l in
    sprintf {|
<section class="azgroup" aria-labelledby="%s"><h2 id="%s" class="azletter mono">%s</h2><ul class="azlist">%s</ul></section>|} id id l
      (concat_map item es)
  in
  sprintf {|<div class="azindex" data-index data-count="%d">%s
</div>|} (List.length entries) (concat_map group groups)

let uses ~page libs =
  let lib name =
    match List.find_opt (fun (e : Registry.entry) -> e.name = name) Registry.entries with
    | Some e -> sprintf {|<li><a href="/docs/%s/">%s%s</a></li>|} e.name (glyph e) e.name
    | None -> Repo.fail "%s: uses %s, which the registry doesn't list" page name
  in
  sprintf {|<ul class="uses" aria-label="Raven libraries it uses">%s</ul>|} (concat_map lib libs)

let fill ~root ~page name args =
  match (name, args) with
  | "release", [] -> esc (Repo.release root)
  | "index", [] -> index ~root
  | "uses", (_ :: _ as libs) -> uses ~page libs
  | "library-count", [] -> number_word (List.length Registry.entries)
  | "tabs", [] -> tabs ~root
  | "excerpt", [ file; name ] -> Highlight.to_html (Repo.definition (root / file) name)
  | "libraries", [] -> libraries ~root
  | "libraries", [ "index" ] -> library_index ~root
  | "posts", [ "all" ] -> posts ~root None
  | "posts", [ n ] -> (
      match int_of_string_opt n with Some n -> posts ~root (Some n) | None -> Repo.fail "%s: posts takes a number or all" page)
  | _ -> Repo.fail "%s: no slot %s %s" page name (String.concat " " args)
