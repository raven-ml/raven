# Quill

Interactive notebooks for OCaml.

Quill turns markdown files into interactive OCaml notebooks. It is part
of the Raven ecosystem. Write prose and code in any text editor, execute
code blocks with a terminal UI or batch evaluator, and store outputs
directly in the markdown as HTML comments.

## Features

- Markdown notebooks: notebooks are `.md` files with fenced OCaml code
  blocks — git-friendly, editor-agnostic, zero lock-in
- Terminal UI: full-screen TUI for cell navigation, execution, and
  output display — no browser required
- Batch evaluation: `quill eval` executes all code blocks and prints or
  saves results
- Watch mode: `quill eval --watch --inplace` re-executes on file change
  for a live editing workflow
- Output format: cell outputs stored as HTML comments, invisible in
  rendered markdown
- Raven integrated: Nx, Rune, Kaun, Hugin, Sowilo, Talon, Brot, and
  Fehu are pre-loaded

## Quick Start

Create a file `notebook.md`:

    # My Notebook

    ```ocaml
    let x = 2 + 2
    let () = Printf.printf "Result: %d\n" x
    ```

Run it:

<!-- $MDX skip -->
```bash
# Open in the terminal UI
quill notebook.md

# Or evaluate from the command line
quill eval notebook.md

# Or live-edit: outputs update on every save
quill eval --watch --inplace notebook.md
```

## Contributing

See the [Raven monorepo README](../README.md) for contribution guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
