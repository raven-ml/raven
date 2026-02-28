# Quill

Interactive notebooks for OCaml.

Quill turns markdown files into interactive OCaml notebooks. It is part
of the Raven ecosystem. Write prose and code in any text editor, execute
code blocks with a terminal UI, web frontend, or batch evaluator, and
store outputs directly in the markdown as HTML comments.

## Features

- Markdown notebooks: notebooks are `.md` files with fenced OCaml code
  blocks — git-friendly, editor-agnostic, zero lock-in
- Terminal UI: full-screen TUI for cell navigation, execution, and
  output display — no browser required
- Web frontend: `quill serve` opens a browser-based notebook with
  CodeMirror 6 editor, real-time execution, autocompletion, and
  diagnostics
- Batch execution: `quill run` executes all code blocks and prints or
  saves results
- Live editing: `quill watch` re-executes on file change for a live
  editing workflow
- Output format: cell outputs stored as HTML comments, invisible in
  rendered markdown
- Raven integrated: Nx, Rune, Kaun, Hugin, Sowilo, Talon, Brot, and
  Fehu are pre-loaded

## Quick Start

<!-- $MDX skip -->
```bash
# Open a notebook in the terminal UI (creates it if new)
quill

# Open in the browser
quill serve

# Execute all cells from the command line
quill run notebook.md

# Live-edit: outputs update on every save
quill watch notebook.md
```

## Contributing

See the [Raven monorepo README](../README.md) for contribution guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
