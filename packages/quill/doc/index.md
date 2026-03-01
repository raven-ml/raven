# Quill

Quill is a REPL and notebook environment for OCaml. Run `quill` for an
interactive toplevel with syntax highlighting, completion, and persistent
history — or open a markdown file for a full notebook with a terminal UI,
web frontend, or batch evaluator.

## Features

- **Interactive REPL**: `quill` launches a toplevel with syntax highlighting, tab completion with ghost text, persistent history, smart phrase-aware submission, and type inspection
- **Markdown notebooks**: notebooks are `.md` files with fenced OCaml code blocks — git-friendly, editor-agnostic, zero lock-in
- **Terminal UI**: full-screen TUI for cell navigation, execution, and output display
- **Web frontend**: `quill serve` opens a browser-based notebook with CodeMirror 6 editor, real-time execution, autocompletion, and diagnostics
- **Batch execution**: `quill run` executes all code blocks and prints or saves results
- **Watch mode**: `quill watch` re-executes on file change for a live editing workflow
- **Output format**: cell outputs stored as `<!-- quill:output -->` HTML comments, invisible in rendered markdown
- **Raven integrated**: Nx, Rune, Kaun, Hugin, Sowilo, Talon, Brot, and Fehu are pre-loaded

## Quick Start

Launch the REPL:

<!-- $MDX skip -->
```bash
quill
```

Or open a notebook in the terminal UI:

<!-- $MDX skip -->
```bash
quill notebook.md
```

Or in the browser:

<!-- $MDX skip -->
```bash
quill serve notebook.md
```

Or execute from the command line:

<!-- $MDX skip -->
```bash
quill run notebook.md
```

## Next Steps

- [Getting Started](01-getting-started/) — REPL, notebooks, execution modes
- [Notebook Format](02-notebook-format/) — how markdown becomes cells, how outputs are stored
- [Execution Modes](03-execution-modes/) — REPL, TUI, web, run, watch, and clean
