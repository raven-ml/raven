(* www ROOT SITE OUT builds the site from the repository at ROOT and the
   site's own folder SITE into OUT: each page of SITE/pages, each post,
   the feed, the docs, and SITE/static and doc/assets copied as they are.
   It warns about pages whose sources changed since they were written,
   and fails on a broken link.

   www --check ROOT SITE fails if any page is stale; releases run it. *)

let ( / ) = Filename.concat

let rec mkdir_p dir =
  if not (Sys.file_exists dir) then (
    mkdir_p (Filename.dirname dir);
    Sys.mkdir dir 0o755)

let write path contents =
  mkdir_p (Filename.dirname path);
  Out_channel.with_open_bin path (fun oc -> Out_channel.output_string oc contents)

let rec copy_dir src dst =
  Array.iter
    (fun name ->
      let s = src / name and d = dst / name in
      if Sys.is_directory s then copy_dir s d else write d (Repo.read s))
    (Sys.readdir src)

(* Every package, core or contrib, is listed in the registry or hidden by
   it, and the registry names no package that doesn't exist. *)
let check_registry root =
  let file = "www/gen/registry.ml" in
  let core = List.map (fun (p : Repo.package) -> p.name) (Repo.core root) in
  let contrib = List.map (fun (p : Repo.package) -> p.name) (Repo.contrib root) in
  let listed = List.map (fun (e : Registry.entry) -> e.name) Registry.entries in
  List.iter
    (fun name ->
      if not (List.mem name listed || List.mem name Registry.hidden) then
        Repo.fail "%s: the package %s is neither listed nor hidden" file name)
    core;
  List.iter (fun name -> if not (List.mem name core) then Repo.fail "%s: %s is not a package in dune-project" file name) (listed @ Registry.hidden);
  List.iter (fun name -> if not (List.mem_assoc name Registry.contrib) then Repo.fail "%s: no contrib entry for %s" file name) contrib;
  List.iter (fun (name, _) -> if not (List.mem name contrib) then Repo.fail "%s: %s is not a package under contrib/" file name) Registry.contrib;
  let manual =
    Sys.readdir (root / "doc" / "manual") |> Array.to_list |> List.filter (fun n -> Filename.check_suffix n ".md") |> List.map Filename.remove_extension
  in
  List.iter (fun n -> if not (List.mem n Registry.manual) then Repo.fail "%s: doc/manual/%s.md is not in the manual" file n) manual;
  List.iter (fun n -> if not (List.mem n manual) then Repo.fail "%s: the manual lists %s, but doc/manual/%s.md doesn't exist" file n n) Registry.manual

(* The pages, sorted by file name. *)
let pages site =
  let dir = site / "pages" in
  Sys.readdir dir |> Array.to_list |> List.filter (fun n -> Filename.check_suffix n ".html") |> List.sort compare
  |> List.map (fun n -> (n, Page.parse (dir / n)))

(* pages/index.html is /, pages/docs.html is /docs/. *)
let out_path name = if name = "index.html" then "index.html" else Filename.remove_extension name / "index.html"

let stale_messages ~root pages =
  List.concat_map
    (fun (name, page) ->
      List.map
        (fun (src, recorded, current) ->
          Printf.sprintf "pages/%s is stale: %s changed since it was written (from: %s %s; now %s)" name src src recorded current)
        (Page.stale ~root page))
    pages

(* A file's URL on the site, by its path in the repository: a page of the
   docs, a post, a hand-written page written from it, or an asset. *)
let site_url ~root ~docs ~posts path =
  let after prefix = String.sub path (String.length prefix) (String.length path - String.length prefix) in
  match List.find_opt (fun (p : Docs.page) -> p.file = path) docs with
  | Some p -> Some p.url
  | None -> (
      match path with
      | "doc/index.md" -> Some "/docs/"
      | "doc/roadmap.md" -> Some "/roadmap/"
      | "doc/performance.md" -> Some "/performance/"
      | "doc/support-raven.md" -> Some "/sponsor/"
      | "doc/blog/index.md" -> Some "/blog/"
      | _ when String.starts_with ~prefix:"doc/assets/" path -> Some ("/assets/" ^ after "doc/assets/")
      | _ when Filename.check_suffix path ".mli" -> (
          (* An interface: its module's page in the API reference. *)
          match String.split_on_char '/' path with
          | ("packages" | "contrib") :: pkg :: _ when List.exists (fun (p : Docs.page) -> p.url = "/docs/" ^ pkg ^ "/") docs ->
              Api.find_module root pkg (String.capitalize_ascii (Filename.remove_extension (Filename.basename path)))
          | _ -> None)
      | _ when String.starts_with ~prefix:"doc/blog/" path ->
          let slug = Filename.remove_extension (after "doc/blog/") in
          if List.exists (fun (p : Repo.post) -> p.slug = slug) posts then Some ("/blog/" ^ slug ^ "/") else None
      | _ -> None)

(* Links into rendered Markdown must name one of its headings. *)
let check_links rendered =
  let ids = List.map (fun (file, (md : Markdown.page)) -> (file, md.ids)) rendered in
  let problems =
    List.concat_map
      (fun (file, (md : Markdown.page)) ->
        md.broken
        @ List.filter_map
            (fun (target, id) ->
              match List.assoc_opt target ids with
              | Some known when not (List.mem id known) -> Some (Printf.sprintf "%s: a link to #%s in %s, which has no such heading" file id target)
              | _ -> None)
            md.anchors)
      rendered
  in
  if problems <> [] then (
    List.iter (fun m -> prerr_endline ("www: " ^ m)) problems;
    Repo.fail "%d broken links" (List.length problems))

let build ~root ~site ~out =
  check_registry root;
  let pages = pages site in
  List.iter (fun m -> prerr_endline ("www: warning: " ^ m)) (stale_messages ~root pages);
  List.iter
    (fun (name, (page : Page.t)) ->
      write (out / out_path name) (Page.fill page (Slots.fill ~root)))
    pages;
  let posts = Repo.posts root and docs = Docs.pages root in
  let env = { Markdown.root; site = site_url ~root ~docs ~posts } in
  let blog = Page.parse (site / "pages" / "blog.html") in
  let posts_md =
    List.mapi
      (fun i (p : Repo.post) ->
        let older = List.nth_opt posts (i + 1) and newer = if i = 0 then None else List.nth_opt posts (i - 1) in
        let html, md = Post.page env ~layout:blog ~prev:older ~next:newer p in
        write (out / "blog" / p.slug / "index.html") html;
        ("doc/blog/" ^ p.slug ^ ".md", md))
      posts
  in
  write (out / "blog" / "feed.xml") (Post.feed posts);
  let docs_md =
    List.map
      (fun ((p : Docs.page), html, md) ->
        write (out / p.url / "index.html") html;
        (p.file, md))
      (Docs.build ~root ~layout:(Page.parse (site / "pages" / "docs.html")) ~pages:docs env)
  in
  check_links (posts_md @ docs_md);
  List.iter (fun (url, html) -> write (out / url / "index.html") html) (Docs.build_api ~root ~layout:(Page.parse (site / "pages" / "docs.html")) ~pages:docs);
  copy_dir (root / "doc" / "assets") (out / "assets");
  copy_dir (site / "static") out

let () =
  try
    match Sys.argv with
    | [| _; "--check"; root; site |] -> (
        match stale_messages ~root (pages site) with
        | [] -> ()
        | ms ->
            List.iter (fun m -> prerr_endline ("www: " ^ m)) ms;
            exit 1)
    | [| _; root; site; out |] -> build ~root ~site ~out
    | _ ->
        prerr_endline "usage: www ROOT SITE OUT, or www --check ROOT SITE";
        exit 2
  with Failure msg ->
    prerr_endline ("www: " ^ msg);
    exit 1
