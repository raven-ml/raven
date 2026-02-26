# Notebook Format

A Quill notebook is a CommonMark markdown file. Fenced code blocks with
a language tag become executable code cells. Everything else becomes text
cells. This page explains the mapping and the output serialization
format.

## Cell Types

A notebook contains two kinds of cells:

- **Code cells**: fenced code blocks with a language info string (e.g.,
  ` ```ocaml `). The language tag identifies the execution kernel.
- **Text cells**: all other markdown content between code blocks.
  Adjacent paragraphs, headings, lists, and other block elements form a
  single text cell.

For example, this notebook has three cells:

    # My Notebook          ← text cell

    Some explanation.

    ```ocaml               ← code cell
    let x = 42
    ```

    More text here.        ← text cell

    ```ocaml               ← code cell
    let y = x + 1
    ```

## Cell IDs

Quill assigns each cell a stable identifier stored as an HTML comment
before the cell:

    <!-- quill:cell id="c_a1b2c3d4e5f6" -->
    ```ocaml
    let x = 42
    ```

Cell IDs are generated automatically for cells that lack them. They
enable the TUI and session to track cells across file reloads and edits.

Users do not need to manage cell IDs. They are preserved by `quill fmt`
and `quill eval --inplace`. Deleting them is harmless — fresh IDs are
assigned on the next load.

## Output Format

After executing a code cell, outputs are stored between marker comments:

    ```ocaml
    let x = 42
    let () = Printf.printf "x = %d\n" x
    ```
    <!-- quill:output -->
    <!-- out:stdout -->
    x = 42
    val x : int = 42
    <!-- /quill:output -->

Each output section is tagged with its type:

- `<!-- out:stdout -->` — captured standard output and toplevel value
  printing
- `<!-- out:stderr -->` — warnings and standard error
- `<!-- out:error -->` — execution errors (syntax errors, type errors,
  runtime exceptions)
- `<!-- out:display MIME -->` — rich output with a MIME type (e.g.,
  `<!-- out:display text/html -->` or `<!-- out:display image/png -->`)

A single code cell can produce multiple output sections. For example, a
cell that prints to both stdout and stderr:

    ```ocaml
    let () = Printf.printf "result: 42\n"
    let () = Printf.eprintf "warning: something\n"
    ```
    <!-- quill:output -->
    <!-- out:stdout -->
    result: 42
    <!-- out:stderr -->
    warning: something
    <!-- /quill:output -->

## Why HTML Comments?

The output format uses HTML comments for several reasons:

1. **Invisible in rendered markdown.** GitHub, editors with preview, and
   documentation tools render the notebook without showing outputs. The
   document reads cleanly whether outputs are present or not.
2. **Valid markdown.** HTML comments are part of the CommonMark
   specification. No custom syntax, no extensions, no preprocessing.
3. **Single file.** Outputs live in the notebook itself. No sidecar
   files, no `.ipynb_checkpoints`, no separate output directories.
4. **Clean stripping.** `quill fmt` removes all output sections in one
   pass. `quill eval --inplace` regenerates them. This makes it easy to
   commit clean notebooks and regenerate outputs in CI.

## Non-OCaml Code Blocks

Code blocks without a language tag, or with a language other than
`ocaml`, are not executed. They pass through unchanged as code cells:

    ```bash
    # This is not executed — it's documentation
    quill eval notebook.md
    ```

    ```json
    { "this": "is also not executed" }
    ```

This lets you include shell commands, JSON examples, and other snippets
in your notebook as documentation without affecting execution.

## Roundtrip Guarantees

Parsing a markdown file with `Quill_markdown.of_string` and rendering it
back with `Quill_markdown.to_string` or `to_string_with_outputs`
preserves:

- Cell content and ordering
- Cell IDs
- Output content and types
- Text cell markdown (headings, lists, links, etc.)

The rendering normalizes some whitespace (consistent blank lines between
cells), but the semantic content is preserved exactly.
