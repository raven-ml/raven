(* Blog posts: each doc/blog/<date>-<slug>.md rendered into the post
   layout at /blog/<slug>/, inside the head, header and footer of
   the blog's page, and the feed of them all. A post is a record of its
   date: a note above it says so. *)

open Html

let months =
  [| "January"; "February"; "March"; "April"; "May"; "June"; "July"; "August"; "September"; "October";
     "November"; "December" |]

(* "2026-07" or "2026-07-14": its month and year, "July 2026". *)
let month_year date =
  let m = int_of_string (String.sub date 5 2) in
  months.(m - 1) ^ " " ^ String.sub date 0 4

(* Minutes at 230 words a minute, at least one. *)
let minutes text =
  let words = List.length (List.filter (( <> ) "") (String.split_on_char ' ' (String.map (function '\n' | '\t' -> ' ' | c -> c) text))) in
  max 1 ((words + 229) / 230)

(* A post's page, and what [Markdown.render] found in it. Its code has no
   copy buttons: a post's code is a record of its date. *)
let page env ~layout ~prev ~next (p : Repo.post) =
  let md = Markdown.render ~copy:false env ("doc/blog/" ^ p.slug ^ ".md") in
  let body = md.body and contents = md.contents in
  let toc = concat_map (fun (id, title) -> sprintf {|<a href="#%s">%s</a>|} (esc id) (esc title)) contents in
  let link label (q : Repo.post option) cls =
    match q with
    | Some q -> sprintf {|<a href="/blog/%s/"%s><span class="caption">%s</span><span style="color: var(--text-0)">%s</span></a>|} q.slug cls label (esc q.title)
    | None -> sprintf {|<a href="/blog/"%s><span class="caption">%s</span><span style="color: var(--text-0)">All posts</span></a>|} cls label
  in
  let main =
    sprintf
      {|<div class="pad post-wrap">
<div class="post-col">
<header class="post-head">
<nav aria-label="Breadcrumb" class="crumbs"><a href="/blog/">blog</a><span aria-hidden="true"> / </span><span aria-current="page">%s</span></nav>
<h1 id="%s" class="h1-article">%s</h1>
<p class="label"><time datetime="%s">%s</time> · %d min read</p>
<p class="card small post-note">This post is a record of %s: its numbers and code are as of then, and the API may have changed since. <a href="/performance/">Current measurements →</a></p>
<details class="phone-only docmenu"><summary>contents</summary><nav class="toc" aria-label="Contents, menu">%s</nav></details>
</header>
<div class="post-grid">
<article class="post-body prose md">
%s
<div class="card post-support">
<div class="stack" style="gap: 4px"><p class="h4">Raven is built in public.</p><p class="small">Sponsorship funds the road to 1.0.</p></div>
<div class="links"><a href="/sponsor/" class="btn btn-1">sponsor raven</a><a href="/blog/feed.xml" class="btn btn-2">rss</a></div>
</div>
<nav aria-label="Posts" class="pager">%s%s</nav>
</article>
<aside class="rail" aria-label="Contents"><nav class="toc" aria-label="Contents"><p class="label">contents</p>%s</nav></aside>
</div>
</div>
</div>|}
      p.date (esc md.title_id) (esc p.title) p.date p.date (minutes body) (month_year p.date) toc body
      (link "← older" prev "") (link "newer →" next {| class="next"|}) toc
  in
  (Page.with_main layout ~title:(p.title ^ " — raven blog") ~description:(Option.value p.blurb ~default:p.title) (`Main main), md)

(* The posts as RSS 2.0. *)
let feed posts =
  let date (p : Repo.post) =
    let day = if String.length p.date = 10 then String.sub p.date 8 2 else "01" in
    sprintf "%s %s %s 00:00:00 +0000" day (String.sub months.(int_of_string (String.sub p.date 5 2) - 1) 0 3) (String.sub p.date 0 4)
  in
  let item (p : Repo.post) =
    sprintf "<item><title>%s</title><link>https://raven-ml.dev/blog/%s/</link><guid>https://raven-ml.dev/blog/%s/</guid><pubDate>%s</pubDate>%s</item>\n"
      (esc p.title) p.slug p.slug (date p)
      (match p.blurb with Some b -> sprintf "<description>%s</description>" (esc b) | None -> "")
  in
  sprintf {|<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel>
<title>raven blog</title><link>https://raven-ml.dev/blog/</link><description>Notes from building Raven.</description>
%s</channel></rss>
|}
    (concat_map item posts)
