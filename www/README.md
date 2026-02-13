# Raven Website

Static site for [raven-ml.dev](https://raven-ml.dev). Built with a small OCaml script (`generate/generate.ml`) that converts Markdown to HTML using cmarkit.

## Build and serve

```bash
dune build www/build
python3 -m http.server -d _build/default/www/build
```

## Structure

- `site/` — source content (Markdown docs, HTML landing pages, static assets)
- `templates/` — HTML templates (`main.html`, `layout_docs.html`, `layout_docs_lib.html`)
- `generate/` — site generator
- `process/` — odoc API docs integration (WIP, not part of the build)
