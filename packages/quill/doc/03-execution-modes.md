# Execution Modes

Quill provides five ways to work with notebooks: the terminal UI for
interactive exploration, a web frontend for browser-based editing, batch
execution for automation, watch mode for a live editing workflow, and
clean for stripping outputs.

## Terminal UI

The default command opens the TUI:

<!-- $MDX skip -->
```bash
quill notebook.md
```

If the file doesn't exist, Quill creates it with a starter template.
With no arguments, `quill` defaults to `notebook.md` in the current
directory:

<!-- $MDX skip -->
```bash
quill
```

### Layout

The TUI displays three areas:

- **Header**: the filename, total cell count (or a running indicator
  with spinner when cells are executing), and an unsaved-changes dot.
- **Cell list**: a scrollable view of all cells. Code cells appear in
  numbered bordered boxes with syntax highlighting. Text cells appear as
  rendered markdown.
- **Footer**: keybinding hints and error messages.

### Navigating and Executing

Navigate between cells with `j`/`k` or the arrow keys. The focused cell
is highlighted with a distinct background and border.

Press `Enter` to execute the focused code cell. Press `Ctrl-A` to
execute all code cells top-to-bottom. During execution, a spinner and
"evaluating" label appear. Outputs display inline below the code.

Pressing `Enter` on a text cell shows an error — only code cells are
executable.

### Cell Management

| Key | Action |
| --- | --- |
| a | Insert a code cell below the focused cell |
| t | Insert a text cell below the focused cell |
| d | Delete the focused cell |
| m | Toggle the focused cell between code and text |
| J | Move the focused cell down |
| K | Move the focused cell up |
| c | Clear the focused cell's outputs |
| Ctrl-L | Clear all outputs |

### File Watching

The TUI checks the file for external modifications every second. If the
file changes on disk (e.g., you edit it in vim or another editor), the
TUI reloads automatically. This means you can keep the TUI open while
editing the notebook externally.

### Saving

Press `s` (or `Ctrl-S`) to save. The notebook is written with all
current outputs. An unsaved-changes indicator (a dot in the header)
appears when the document has been modified since the last save.

Quitting with unsaved changes requires pressing `q` twice. The error
bar shows: "Unsaved changes. Press q again to quit, s to save."

### Interrupting

Press `Ctrl-C` to interrupt a running execution. This sends an
interrupt signal to the kernel.

## Web Frontend

Start the web notebook server:

<!-- $MDX skip -->
```bash
quill serve notebook.md
```

This starts an HTTP server at `http://127.0.0.1:8888` and opens the
notebook in your browser. Use `--port` (or `-p`) to change the port:

<!-- $MDX skip -->
```bash
quill serve --port 9000 notebook.md
```

With no file argument, `quill serve` defaults to `notebook.md` in the
current directory, creating it if needed.

### Features

The web UI provides a full notebook interface in the browser:

- **CodeMirror 6 editor** with OCaml syntax highlighting and theming
- **Real-time execution** — cell outputs stream via WebSocket as code runs
- **Autocompletion** — context-aware completions for OCaml code
- **Type information** — hover over identifiers to see their types
- **Diagnostics** — errors and warnings shown inline in the editor
- **Undo / redo** — checkpoint-based history
- **Cell management** — insert, delete, move, and toggle cells between
  code and text

### Keyboard Shortcuts

| Key | Action |
| --- | --- |
| j / k | Navigate cells |
| Enter | Edit focused cell |
| Ctrl-Enter | Execute focused cell |
| Ctrl-Shift-Enter | Execute all cells |
| a | Insert code cell below |
| t | Insert text cell below |
| d | Delete focused cell |
| Ctrl-S | Save |
| Ctrl-C | Interrupt execution |

### Connection Status

The web UI automatically reconnects if the server restarts or the
connection drops. A banner appears during disconnection with the
reconnection status. Reconnection uses exponential backoff (up to 30
seconds).

## Batch Execution

Non-interactive execution of all code cells:

<!-- $MDX skip -->
```bash
quill run notebook.md
```

Executes every code cell in order and prints the complete notebook with
outputs to stdout. The original file is not modified.

Use cases:
- Quick review of notebook outputs
- Piping output to other tools
- CI/CD validation

### In-place updates

<!-- $MDX skip -->
```bash
quill run --inplace notebook.md
```

Same as above, but writes outputs back into the file. After running,
the file contains `<!-- quill:output -->` sections below each code
block.

## Watch Mode

<!-- $MDX skip -->
```bash
quill watch notebook.md
```

Watches the file for changes and re-executes all cells on every save,
writing outputs back into the file. A timestamp is printed on each
re-evaluation:

    [14:32:05] File changed, re-evaluating...

Watch mode runs until interrupted with `Ctrl-C`.

### The Live Editing Workflow

`quill watch` creates a live notebook experience using your own editor:

1. Open two terminals (or splits in tmux / zellij)
2. Terminal 1: open the notebook in your editor (`vim notebook.md`)
3. Terminal 2: `quill watch notebook.md`
4. Edit and save in terminal 1. Terminal 2 detects the change,
   re-executes all cells, and writes outputs back into the file.
5. Your editor picks up the file change (vim with `:set autoread`,
   VS Code automatically, etc.)

This gives you the "edit in your editor, see results update" workflow
without a browser or notebook server.

## Cleaning

Strip all outputs from a notebook:

<!-- $MDX skip -->
```bash
quill clean notebook.md            # print clean markdown to stdout
quill clean --inplace notebook.md  # strip outputs from the file
```

Cell IDs are preserved. Only `<!-- quill:output -->` sections are
removed.

Use cases:
- **Clean diffs**: strip outputs before committing, regenerate with
  `quill run --inplace` in CI
- **Fresh start**: remove stale outputs before a full re-run
- **Sharing**: send a clean notebook without outputs

## Creating Notebooks

Create a new notebook from a starter template:

<!-- $MDX skip -->
```bash
quill new                   # creates notebook.md
quill new analysis.md       # creates analysis.md
```

The starter template includes working examples of Nx arrays, Hugin
plotting, and Rune automatic differentiation.

## Raven Packages

All execution modes (TUI, web, run, and watch) use the Raven kernel,
which pre-loads these packages automatically:

- **Nx** — n-dimensional arrays
- **Rune** — tensor computation with autodiff
- **Kaun** — neural networks and training
- **Hugin** — visualization and plotting
- **Sowilo** — image processing
- **Talon** — dataframes
- **Brot** — tokenization
- **Fehu** — reinforcement learning

Pretty-printers for Nx and Rune tensors are installed automatically.
Your first code cell can use `open Nx` or any other Raven module without
setup.
