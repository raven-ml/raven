# Quill

Interactive computing environment for OCaml.

Quill is a REPL and notebook environment for OCaml. Run `quill` for an
interactive toplevel with syntax highlighting, completion, and persistent
history — or open a markdown file for a full notebook experience with a
terminal UI, web frontend, or batch evaluator. Part of the Raven
ecosystem.

## Features

- Interactive REPL: `quill` launches a toplevel with syntax highlighting,
  tab completion with ghost text, persistent history, smart phrase-aware
  submission, and type inspection — no browser or file required
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
# Interactive REPL
quill

# Open a notebook in the terminal UI
quill notebook.md

# Open in the browser
quill serve notebook.md

# Execute all cells from the command line
quill run notebook.md

# Live-edit: outputs update on every save
quill watch notebook.md
```

## Contributing

See the [Raven monorepo README](../README.md) for contribution guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
