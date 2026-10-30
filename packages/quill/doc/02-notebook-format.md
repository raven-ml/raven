# Notebook Format

A Quill notebook is a CommonMark markdown file. Fenced code blocks marked
`{.run}` after their language become code cells. Everything else becomes
text cells. This page explains the mapping and the output serialization
format.

## Cell Types

A notebook contains two kinds of cells:

- **Code cells**: fenced code blocks at the top level of the notebook
  whose info string is a language, then `{.run}`, as in the opening
  fence ```` ```ocaml {.run} ````. The language selects the kernel that
  runs the cell.
- **Text cells**: all other markdown content between code cells.
  Adjacent paragraphs, headings, lists, code blocks without the mark and
  other block elements form a single text cell.

For example, this notebook has four cells:

    # My Notebook          ← text cell

    Some explanation.

    ```ocaml {.run}        ← code cell
    let x = 42
    ```

    More text here.        ← text cell

    ```ocaml {.run}        ← code cell
    let y = x + 1
    ```

## Cell IDs

Quill assigns each cell a stable identifier stored as an HTML comment
before the cell:

    <!-- quill:cell id="c_a1b2c3d4e5f6" -->
    ```ocaml {.run}
    let x = 42
    ```

Cell IDs are generated automatically for cells that lack them. They
enable the TUI and session to track cells across file reloads and edits.

Users do not need to manage cell IDs. They are preserved by `quill clean`
and `quill run --inplace`. Deleting them is harmless — fresh IDs are
assigned on the next load.

## Output Format

After executing a code cell, outputs are stored between marker comments:

    ```ocaml {.run}
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

    ```ocaml {.run}
    let () = Printf.printf "result: 42\n"
    let () = Printf.eprintf "warning: something\n"
    ```
    <!-- quill:output -->
    <!-- out:stdout -->
    result: 42
    <!-- out:stderr -->
    warning: something
    <!-- /quill:output -->

## Images

An image output is an `<img>` tag. By default its source is a data URI,
so the notebook stays a single file:

    <!-- out:display image/svg+xml -->
    <img src="data:image/svg+xml;base64,PHN2Zy8+">

`quill run --inplace --figures` stores each image as a file instead, in
`figures/<stem>/` beside the notebook, where `<stem>` is the notebook's
file name without `.md`. The notebook's first line records the choice,
and each image references its file by path:

    <!-- quill:figures files -->

    ...
    <!-- out:display image/svg+xml -->
    <img src="figures/notebook/c_a1b2c3d4e5f6.svg">

A notebook keeps the storage its first line records: `quill run` without
`--figures`, `quill clean`, `quill serve` and the terminal notebook save
its images the way they found them. `figures/<stem>/` belongs to a
notebook that stores its images as files. Saving it removes the files of
images it no longer has, so `quill clean` empties and removes the
directory. Other files, such as images that text cells show from
`figures/`, are left alone.

To store images inline again, delete the first line, then delete
`figures/<stem>/`: a notebook that stores its images inline never
touches that directory, and `quill build` would keep copying it.

Quill reads image files only from the notebook's own `figures/<stem>/`.
A reference to another path, or to a missing file, is an error that names
the notebook and the cell. The rules are in the Files section of
[`quill_markdown.mli`](../lib/markdown/quill_markdown.mli).

## Why HTML Comments?

The output format uses HTML comments for several reasons:

1. **Invisible in rendered markdown.** GitHub, editors with preview, and
   documentation tools render the notebook without showing outputs. The
   document reads cleanly whether outputs are present or not.
2. **Valid markdown.** HTML comments are part of the CommonMark
   specification. No custom syntax, no extensions, no preprocessing.
3. **Single file.** Outputs live in the notebook itself, images too
   unless stored as files. No `.ipynb_checkpoints`, no separate output
   directories.
4. **Clean stripping.** `quill clean` removes all output sections in one
   pass. `quill run --inplace` regenerates them. This makes it easy to
   commit clean notebooks and regenerate outputs in CI.

## Code Blocks That Do Not Run

A code block without the `{.run}` mark is part of the text and never
runs, whatever its language:

    ```bash
    # This is not executed — it's documentation
    quill run notebook.md
    ```

    ```ocaml
    (* Shown, and not executed *)
    let wrong = 1 + "one"
    ```

This lets you include shell commands, JSON examples, and code you only
want to show without affecting execution.

The mark has one spelling. A block that asks to run in another shape, such
as ```` ```{.ocaml .run} ```` or ```` ```ocaml {.run .numberLines} ````, is
an error that names its line, and so is a marked block inside a list or a
quote. The rules are in
[`quill_markdown.mli`](../lib/markdown/quill_markdown.mli).

A cell in a language that no kernel runs, such as ```` ```bash {.run} ````
in a notebook whose kernel runs OCaml, gets an error output when it is
run, and the cells after it still run.

## Migrating Older Notebooks

Notebooks written before the `{.run}` mark ran every ```` ```ocaml ````
block, and older Quill wrote a `<!-- quill:cell -->` comment above each
cell. Such a notebook does not load. The error names the line of the
first block in that shape and prints the command that marks the
notebook's OCaml cells:

<!-- $MDX skip -->
```bash
sed -E -i.bak 's/^( {0,3})(`{3,}|~{3,})[[:blank:]]*ocaml[[:blank:]]*$/\1\2ocaml {.run}/' notebook.md
```

It marks a fence of backticks or tildes of any length, indented up to
three spaces, with blanks around `ocaml`, and keeps the old file as
`notebook.md.bak`. It also rewrites such a line inside another code
block, and a fence with more words after `ocaml` must be edited by hand.
A block in another language that older Quill ran as a cell, such as a
`bash` block under a cell comment, is left as it is. Delete its comment
to keep it as text. The migrated cells keep their ids and outputs.

## Roundtrip Guarantees

Loading a notebook with `Quill_markdown.load` and saving it back with
`Quill_markdown.save`, in the storage `load` reports, preserves:

- Cell content and ordering
- Cell IDs
- Output content and types, images included
- Text cell markdown (headings, lists, links, etc.)

The rendering normalizes some whitespace (consistent blank lines between
cells), but the semantic content is preserved exactly.
