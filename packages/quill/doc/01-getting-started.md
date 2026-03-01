# Getting Started

This guide covers the REPL, creating a notebook, executing it in
different modes, and viewing results.

## Installation

<!-- $MDX skip -->
```bash
opam install quill
```

Or build from source:

<!-- $MDX skip -->
```bash
git clone https://github.com/raven-ml/raven
cd raven && dune build quill
```

## The REPL

The fastest way to try Quill:

<!-- $MDX skip -->
```bash
quill
```

This launches an interactive toplevel. Type OCaml expressions, press
Enter to evaluate. All Raven packages are pre-loaded — try
`Nx.create Float32 [|3|] [|1.; 2.; 3.|]` right away.

| Key | Action |
| --- | --- |
| Enter | Submit (if phrase is complete) |
| Ctrl-Enter | Insert newline |
| Tab | Trigger completion |
| Ctrl-T | Type at cursor |
| Up / Down | History navigation |
| Ctrl-C | Clear input / interrupt |
| Ctrl-D | Quit (on empty input) |

Quill also works in pipes: `echo 'print_endline "hello"' | quill`
executes the input and prints the result.

## Your First Notebook

Create a notebook with a starter template:

<!-- $MDX skip -->
```bash
quill new notebook.md
quill notebook.md
```

This opens the terminal UI. Run each cell with `Enter` to see arrays,
plots, and automatic differentiation in action.

You can also create a named notebook:

<!-- $MDX skip -->
```bash
quill new analysis.md
quill analysis.md
```

Or open an existing notebook:

<!-- $MDX skip -->
```bash
quill notebook.md
```

## Creating a Notebook

Any `.md` file with fenced OCaml code blocks is a Quill notebook. Create
a file `notebook.md`:

    # Statistics

    We'll compute some basic statistics.

    ```ocaml
    open Nx

    let data = of_list float32 [| 1.0; 2.0; 3.0; 4.0; 5.0 |] [| 5 |]
    let () = Printf.printf "Data: %s\n" (to_string data)
    ```

    Now the mean:

    ```ocaml
    let m = mean data
    let () = Printf.printf "Mean: %s\n" (to_string m)
    ```

Code blocks share state: variables defined in one block are available in
all subsequent blocks.

## Running with `quill run`

Batch-execute all code blocks:

<!-- $MDX skip -->
```bash
quill run notebook.md
```

This prints the complete notebook with outputs to stdout. The original
file is not modified. Useful for quick checks and CI.

### Saving outputs in-place

<!-- $MDX skip -->
```bash
quill run --inplace notebook.md
```

Executes all code blocks and writes outputs back into the file as HTML
comments. The file now contains `<!-- quill:output -->` sections below
each code block. The notebook remains valid, readable markdown.

## Watch Mode

<!-- $MDX skip -->
```bash
quill watch notebook.md
```

Watches the file for changes (polling every second). On each save,
re-executes all cells and writes outputs back. This enables a live
editing workflow: edit in your favorite editor in one terminal, see
results update in the file.

## Running with the TUI

The default command opens the terminal UI:

<!-- $MDX skip -->
```bash
quill notebook.md
```

The TUI displays a full-screen interface with:

- **Header**: filename, cell count, running indicator
- **Cells**: code cells in numbered bordered boxes with syntax
  highlighting, text cells as rendered markdown
- **Footer**: keybinding hints and error messages

### Keybindings

| Key | Action |
| --- | --- |
| j / k | Navigate cells |
| J / K | Move cell down / up |
| Up / Down | Navigate cells |
| Enter | Execute focused cell |
| Ctrl-A | Execute all cells |
| a | Insert code cell below |
| t | Insert text cell below |
| d | Delete focused cell |
| m | Toggle cell kind (code / text) |
| c | Clear focused cell outputs |
| Ctrl-L | Clear all outputs |
| s / Ctrl-S | Save |
| Ctrl-C | Interrupt execution |
| q | Quit |

The TUI watches the file for external changes. If you edit the notebook
in another editor, the TUI reloads automatically.

Quitting with unsaved changes requires pressing `q` twice, or `s` to
save first.

## Running with the Web UI

Start the web frontend:

<!-- $MDX skip -->
```bash
quill serve notebook.md
```

This starts an HTTP server at `http://127.0.0.1:8888` and opens the
notebook in your browser. The web UI provides:

- **CodeMirror 6 editor** with OCaml syntax highlighting
- **Real-time execution** via WebSocket — outputs appear as cells run
- **Autocompletion** and **type-at-position** for OCaml code
- **Diagnostics** — errors and warnings shown inline
- **Keyboard shortcuts** — `j`/`k` navigation, `Ctrl+Enter` to execute

Use `--port` (or `-p`) to change the port:

<!-- $MDX skip -->
```bash
quill serve --port 9000 notebook.md
```

The web UI shares the same markdown notebook format and Raven kernel as
the TUI and batch evaluator.

## Stripping Outputs

Remove all outputs from a notebook:

<!-- $MDX skip -->
```bash
quill clean notebook.md            # print clean markdown to stdout
quill clean --inplace notebook.md  # strip outputs from the file
```

Useful before committing to git for clean diffs, or to get a fresh start
before re-execution.

## Persistent State

Code cells execute sequentially in a shared OCaml toplevel. Variables
and functions defined in one cell are available in all subsequent cells:

    ```ocaml
    let greet name = Printf.printf "Hello, %s!\n" name
    ```

    ```ocaml
    let () = greet "world"
    (* prints: Hello, world! *)
    ```

This mirrors the behavior of the OCaml toplevel (`ocaml` REPL).

## Raven Packages

All Raven packages are pre-loaded automatically. Your first code cell
can immediately use `open Nx`, `open Rune`, `open Hugin`, etc. without
any setup. Pretty-printers for Nx and Rune tensors are installed
automatically.

## Next Steps

- [Notebook Format](02-notebook-format/) — how markdown maps to cells,
  how outputs are serialized
- [Execution Modes](03-execution-modes/) — TUI, web frontend, live
  editing workflow, batch execution
