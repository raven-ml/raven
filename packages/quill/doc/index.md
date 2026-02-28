# Quill

Quill turns markdown files into interactive OCaml notebooks. Write prose
and code in any text editor, execute code blocks with a terminal UI, web
frontend, or batch evaluator, and store outputs directly in the markdown.

## Features

- **Markdown notebooks**: notebooks are `.md` files with fenced OCaml code blocks — git-friendly, editor-agnostic, zero lock-in
- **Terminal UI**: full-screen TUI for cell navigation, execution, and output display
- **Web frontend**: `quill serve` opens a browser-based notebook with CodeMirror 6 editor, real-time execution, autocompletion, and diagnostics
- **Batch execution**: `quill run` executes all code blocks and prints or saves results
- **Watch mode**: `quill watch` re-executes on file change for a live editing workflow
- **Output format**: cell outputs stored as `<!-- quill:output -->` HTML comments, invisible in rendered markdown
- **Raven integrated**: Nx, Rune, Kaun, Hugin, Sowilo, Talon, Brot, and Fehu are pre-loaded

## Quick Start

<!-- $MDX skip -->
```bash
quill
```

This creates `notebook.md` with a starter template and opens the
terminal UI. Run each cell with `Enter` to see arrays, plots, and
automatic differentiation.

Or open a notebook in the browser:

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

- [Getting Started](01-getting-started/) — create a notebook, run it, view results
- [Notebook Format](02-notebook-format/) — how markdown becomes cells, how outputs are stored
- [Execution Modes](03-execution-modes/) — TUI, web, run, watch, and clean
