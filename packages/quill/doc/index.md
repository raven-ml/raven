# Quill

Quill turns markdown files into interactive OCaml notebooks. Write prose
and code in any text editor, execute code blocks with a terminal UI, web
frontend, or batch evaluator, and store outputs directly in the markdown.

## Features

- **Markdown notebooks**: notebooks are `.md` files with fenced OCaml code blocks — git-friendly, editor-agnostic, zero lock-in
- **Terminal UI**: full-screen TUI for cell navigation, execution, and output display
- **Web frontend**: `quill serve` opens a browser-based notebook with CodeMirror 6 editor, real-time execution, autocompletion, and diagnostics
- **Batch evaluation**: `quill eval` executes all code blocks and prints or saves results
- **Watch mode**: `quill eval --watch --inplace` re-executes on file change for a live editing workflow
- **Output format**: cell outputs stored as `<!-- quill:output -->` HTML comments, invisible in rendered markdown
- **Raven integrated**: Nx, Rune, Kaun, Hugin, Sowilo, Talon, Brot, and Fehu are pre-loaded

## Quick Start

A Quill notebook is a markdown file where fenced OCaml code blocks are
executable cells. Everything else is text:

    # My Analysis

    Let's compute something:

    ```ocaml
    let x = 2 + 2
    let () = Printf.printf "Result: %d\n" x
    ```

    The result appears below the code block.

Run it from the command line:

<!-- $MDX skip -->
```bash
quill eval notebook.md
```

Or open it in the terminal UI:

<!-- $MDX skip -->
```bash
quill notebook.md
```

Or start the web frontend:

<!-- $MDX skip -->
```bash
quill serve notebook.md
```

Then open `http://127.0.0.1:8888` in your browser.

## Next Steps

- [Getting Started](01-getting-started/) — create a notebook, run it, view results
- [Notebook Format](02-notebook-format/) — how markdown becomes cells, how outputs are stored
- [Execution Modes](03-execution-modes/) — TUI, web, eval, watch, and fmt
