# www

raven-ml.dev, built by one small OCaml program. It implements the A″ v4
design.

```sh
dune build www/site          # build; warns about stale pages
dune build @www/check        # fails on a stale page; run before a release
python3 -m http.server -d _build/default/www/site
```

## The design

1. **Pages are HTML.** Each page in `pages/` is a complete document, head,
   header and footer included, written by hand. The build serves it as it
   is, except for its slots. `pages/index.html` is `/`, and
   `pages/roadmap.html` is `/roadmap/`. A change to the header or the footer
   is made in each page; nothing keeps the copies in step.
2. **Markdown leads; pages follow at release.**
   - The repository's Markdown is the source: `doc/roadmap.md`,
     `doc/performance.md` and `doc/support-raven.md`. They are read on GitHub
     and by contributors.
   - A page written from one names it in its head, with the digest of the
     version it matches:

     ```html
     <meta name="source" content="doc/roadmap.md 308199d49c34">
     ```

   - When the file changes, the build warns that the page is stale, and the
     check fails until someone updates the page and its digest. The digest is
     the first 12 hex digits of the file's MD5, and the warning prints the new
     one.
3. **Slots for what must not be copied.** A slot, `<!-- slot: NAME ARGS -->`,
   is filled at build time from the repository (`gen/slots.ml`):
   - `release`: the release the site is for, the newest version in
     `CHANGES.md`, as "beta 1";
   - `library-count`: the core libraries, counted, in words;
   - `tabs`: the landing page's code tabs, each the first block mdx runs in a
     library's `doc/index.md`, or in the section of a guide the registry
     names (rune's shows a compiled training step);
   - `excerpt FILE NAME`: a top-level definition, verbatim;
   - `libraries`: the core's cards by group, then contrib. `libraries index`
     is the docs landing's compact index, with the guides and examples each
     library has;
   - `posts N`: the latest N posts. `posts all`: every post.
4. **Posts and docs are Markdown, rendered** (`gen/markdown.ml`).
   - A link to a file of the repository goes where the file is on the site,
     or to it on GitHub when the site doesn't show it, so the Markdown reads
     the same on GitHub and on the site. A link to a file that doesn't
     exist, or to a heading that doesn't, fails the build.
   - OCaml is highlighted. A block of toplevel phrases shows each phrase and
     its output, in the figure of the code before it. Blocks mdx runs and
     shell commands can be copied.
5. **Posts.** Each `doc/blog/<date>-<slug>.md` becomes `/blog/<slug>/`, inside
   `pages/blog.html`'s head, header and footer, in the post layout: its
   title, date and reading time, a note that it records its date, and
   contents from its second-level headings. Its blurb is its entry in
   `doc/blog/index.md`. `/blog/feed.xml` lists the posts.
6. **Docs** (`gen/docs.ml`), inside `pages/docs.html`'s head, header and
   footer, in the docs layout: a sidebar, the page, and its contents.
   - `doc/manual/<name>.md` is `/docs/manual/<name>/`, in the order of
     `Registry.manual`.
   - A library's `doc/index.md` is `/docs/<lib>/`, with its Python
     counterpart, how to install it, the libraries it is built on (its
     `depends` in `dune-project`) and its examples. A contrib package
     without one has its `README.md` there.
   - Its guides, `doc/NN-<slug>.md`, are `/docs/<lib>/<slug>/`, numbered,
     and named in the sidebar as the library's page links them.
   - A pager reads the manual, then each library, its page and its guides.
   - A page that mdx runs says so.
7. **The API reference is odoc's** (`gen/api.ml`). `dune build @doc-json`
   has odoc write each page of a package's API as JSON, its parts as HTML
   (odoc's format for embedding its pages in another site), into
   `_doc/_html` of the build; the site rule depends on it.
   - `_doc/_html/<pkg>/<Path>/index.html.json` is `/docs/<pkg>/api/<Path>/`,
     in the docs layout, its contents in the rail. `/docs/<pkg>/api/` lists
     the package's libraries, each with its modules and their synopses.
   - Links between odoc's pages go to the site's; one to a package the site
     doesn't show keeps its text. OCaml in comments is highlighted as
     everywhere else; signatures keep odoc's markup and links, and the
     stylesheet colours odoc's classes as the lexer's.
   - A link in the docs to an interface, `.mli`, goes to its module's page.
   - odoc warnings are errors in the repository, so a module with one has
     no page, and the site doesn't build until it is fixed.
8. **HTML first; JS only adds.** Every page is complete without JavaScript;
   `static/site.js` adds the copy buttons and the code tabs.
9. **A missing fact fails the build.** Each of these stops the build:
   - a package the registry neither lists nor hides, or a page of the
     manual it doesn't order;
   - a broken link in Markdown;
   - a tab without a tested block;
   - a missing definition;
   - an unknown slot;
   - a malformed source tag.

## Files

- `dune`: the site rule, and the `check` alias. The site rule depends on
  the files Markdown may link to, so that links can be checked.
- `pages/`: the hand-written pages:
  - `index.html`, `docs.html`, `blog.html`;
  - `roadmap.html`, `performance.html`, `sponsor.html`.
- `gen/`: the generator:
  - `www.ml`: builds every page, post, page of the docs and the feed,
    copies `static/` and `doc/assets/`, and checks;
  - `page.ml`: a page's sources, its staleness, filling its slots, and
    lending its head, header and footer to posts;
  - `slots.ml`: what slots are filled with;
  - `markdown.ml`: the repository's Markdown as HTML, its links resolved;
  - `post.ml`: posts and the feed;
  - `docs.ml`: the manual, each library's page and guides, and the frame
    of the API's pages;
  - `api.ml`: the API reference, from odoc's JSON;
  - `json.ml`: reading JSON;
  - `repo.ml`: reading facts from the repository;
  - `registry.ml`: what the site says about each package (glyph, colour,
    group, Python counterpart, one line), the packages it hides, and the
    order of the manual;
  - `highlight.ml`: OCaml source as highlighted HTML, at build time;
  - `html.ml`: escaping, and a library's glyph.
- `static/`: the stylesheet (the A″ v4 design system, then the pages'
  classes), `site.js`, `raven.svg`, and the fonts. The fonts are Geist and
  Geist Mono, subset to the characters the site uses, and Noto Sans Runic,
  subset to the library glyphs: 24 KB in all, under the SIL Open Font License.

## Decisions

Checked against `doc/dev/design.md`. Citations were read for this
document. Visual decisions are the A″ ledger's, and are not reopened here,
except the code palette (9).

| # | Decision | Why | Reference |
| --- | --- | --- | --- |
| 1 | Pages are complete, hand-written HTML documents; Markdown is the source for roadmap, performance and sponsor, and pages follow it at release, with a stale check. | Generating designed pages from Markdown forced the design into Markdown conventions and the generator to parse prose; two copies with a check keep both useful. | maintainer, 2026-11-10 |
| 2 | Slots fill only what would otherwise be copied by hand: tested code, the packages, the posts, the release. | Code on a page must run; a package list must match the packages. | A″ ledger, decision 4 |
| 3 | The generator writes HTML as OCaml strings with explicit escaping, and has no template language beyond slots. | A slot is a named, closed set of functions; anything more is a second language. | `design.md`, "Argue for the most elegant shape" |
| 4 | cmarkit parses Markdown. | The docs are CommonMark; cmarkit already ships with raven through quill. | `design.md`, "Hard constraints" |
| 5 | Pages work without JS; JS builds the tab list from the panels' headings. | "make your page work with HTML first"; "The JavaScript layer is not fault-tolerant." | GOV.UK Service Manual, "Building a resilient frontend using progressive enhancement", <https://www.gov.uk/service-manual/technology/using-progressive-enhancement> |
| 6 | Tabs follow the APG pattern: `tablist`, `tab`, `tabpanel`, `aria-selected`, `aria-controls`, arrows with wrap, Home, End, activation on focus. | "It is recommended that tabs activate automatically when they receive focus" | WAI-ARIA APG, Tabs pattern, "Keyboard Interaction", <https://www.w3.org/WAI/ARIA/apg/patterns/tabs/> |
| 7 | Every page reflows at 320px. | "without requiring scrolling in two dimensions for … a width equivalent to 320 CSS pixels" | WCAG 2.2, SC 1.4.10 Reflow, <https://www.w3.org/WAI/WCAG22/Understanding/reflow.html> |
| 8 | Fonts are self-hosted subsets; no third-party requests. | No other host stands between a reader and a page. | `design.md`, "Hard constraints" |
| 9 | Code is highlighted at build time by a lexer of our own, following the manual's token classes; no highlighting library at build or run time. | One page of code, and no dependency; a reader's browser runs nothing to colour code. | OCaml manual, "Lexical conventions", <https://ocaml.org/manual/5.3/lex.html>: "Nested comments are handled correctly." |
