# Installation

## Prerequisites

Raven requires **OCaml 5.2** or later and **opam**.

If you don't have opam installed, follow the [official instructions](https://opam.ocaml.org/doc/Install.html). Then create a switch:

```bash
opam switch create raven 5.2.0
eval $(opam env)
```

## Installing from opam

Install the entire ecosystem:

```bash
opam install raven
```

Or install individual libraries:

```bash
opam install nx          # just arrays
opam install rune        # arrays + autodiff
opam install kaun        # arrays + autodiff + neural networks
opam install brot        # tokenization
opam install talon       # dataframes
```

## Building from Source

```bash
git clone https://github.com/raven-ml/raven
cd raven
dune pkg lock && dune build
```

To build a specific library:

```bash
dune build packages/nx            # just nx
dune build packages/kaun          # kaun + its dependencies
```

## System Dependencies

Most Raven libraries have no system dependencies beyond OCaml. The exceptions:

| Library | Requires | macOS | Ubuntu/Debian |
|---------|----------|-------|---------------|
| **hugin** | Cairo, SDL2 | `brew install cairo sdl2` | `apt install libcairo2-dev libsdl2-dev` |

## Using Raven in Your Project

Add libraries to your `dune-project`:

```dune
(lang dune 3.0)
(package
 (name my_project)
 (depends
  ocaml
  dune
  nx
  rune))
```

And your `dune` file:

```dune
(executable
 (name main)
 (libraries nx rune))
```

## Verify Your Installation

Create a file `main.ml`:

```ocaml
let () =
  let open Nx in
  let x = linspace Float32 0. 1. 5 in
  print_data x
```

Build and run:

```bash
dune exec ./main.exe
```

You should see five evenly-spaced values printed.

## Editor Setup

For the best development experience, use an editor with OCaml LSP support:

- **VS Code**: Install the [OCaml Platform extension](https://marketplace.visualstudio.com/items?itemName=ocamllabs.ocaml-platform)
- **Emacs**: Use [ocaml-eglot](https://github.com/tarides/ocaml-eglot)
- **Vim/Neovim**: Use [ocaml-lsp](https://github.com/ocaml/ocaml-lsp) with your LSP client

## Troubleshooting

**Missing system libraries**: If Hugin fails to build, ensure Cairo and SDL2 development headers are installed.

**Opam switch issues**: Run `eval $(opam env)` after creating or switching opam switches.

**Build failures**: Check your OCaml version with `ocaml --version`. Raven requires 5.2.0 or later.

**Getting help**: Report issues at [github.com/raven-ml/raven/issues](https://github.com/raven-ml/raven/issues).
