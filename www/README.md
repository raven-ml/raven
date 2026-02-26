# Raven Website

Static site for [raven-ml.dev](https://raven-ml.dev). Built with a small OCaml script (`generate/generate.ml`) that converts Markdown to HTML using cmarkit.

## Build and serve

```bash
dune build www/build
python3 -m http.server -d _build/default/www/build
```

## Structure

- `site/` — HTML landing pages and static assets
- `../doc/` — general documentation (installation, roadmap, etc.)
- `templates/` — HTML templates (`main.html`, `layout_docs.html`, `layout_docs_lib.html`)
- `generate/` — site generator
- `process/` — odoc API docs integration (WIP, not part of the build)

Library-specific docs live in each library's `doc/` directory (e.g., `packages/nx/doc/`, `packages/rune/doc/`) where they're tested with mdx. The site generator pulls them in automatically.
