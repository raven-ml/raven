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

## Links between doc pages

Write link destinations as paths to the file you mean, relative to the file you are editing:

```markdown
See [Training](03-training.md) for losses and metrics.
See [Nx](../packages/nx/doc/index.md) for arrays.
See the [BERT pipeline](../10-bert-pipeline/) example.
```

Those resolve in an editor and on a repository host. `generate/links.ml` turns them into site URLs (`/docs/kaun/training/`, `/docs/nx/`, `/docs/brot/examples/bert-pipeline/`) when the page is rendered, so the same link works in both places. A link to a directory means the page that directory publishes through its `README.md` or `index.md`.

Site-absolute destinations (`/docs/...`) and destinations with no file behind them fail the build, naming the file and line. Images and data files work the same way: reference them by relative path and the generator publishes them alongside the page.
