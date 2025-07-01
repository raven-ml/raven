# Installation

This guide covers installing the Raven ecosystem.

## System Dependencies

Raven requires OCaml and opam. Individual libraries may have additional dependencies:

### Core Dependencies

All platforms need:
```bash
# OCaml package manager
opam
```

### Additional Dependencies for Hugin (Plotting)

If you plan to use Hugin for visualization, you'll also need Cairo and SDL2:

#### macOS
```bash
brew install cairo sdl2
```

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install libcairo2-dev libsdl2-dev
```

### Other Platforms

Raven is developed primarily on macOS and Linux. Windows support requires WSL2.

## OCaml and Opam Setup

Initialize opam and create a switch with the required OCaml version:

```bash
# Initialize opam (if not already done)
opam init

# Create a new switch for Raven
opam switch create raven 5.2.0
eval $(opam env)
```

## Installing Raven

### From Opam

Once released, you'll be able to install Raven directly from opam:

```bash
# Install the entire ecosystem
opam install raven

# Or install individual libraries
opam install nx hugin rune kaun sowilo quill
```

### Using Raven in Your Project

After installation, add Raven libraries to your Dune project:

```dune
; dune-project
(lang dune 3.0)
(package
 (name my_project)
 (depends
  ocaml
  dune
  nx
  hugin
  rune))
```

In your `dune` files:

```dune
; lib/dune
(library
 (name my_lib)
 (libraries nx hugin rune))

; bin/dune  
(executable
 (public_name my_app)
 (libraries my_lib nx))
```

## Setting up Your Editor

For the best development experience, we recommend using VS Code with the OCaml Platform extension.

### VS Code Setup

1. Install [Visual Studio Code](https://code.visualstudio.com/)
2. Install the [OCaml Platform extension](https://marketplace.visualstudio.com/items?itemName=ocamllabs.ocaml-platform)
3. The extension will automatically detect your opam switch

### Other Editors

- **Emacs**: Use [ocaml-eglot](https://github.com/tarides/ocaml-eglot) for modern LSP support with Eglot
- **Vim/Neovim**: Use [ocaml-lsp](https://github.com/ocaml/ocaml-lsp) with your LSP client

## Troubleshooting

### Common Issues

**Missing system libraries**: Ensure Cairo and SDL2 are installed with development headers.

**Opam switch issues**: Always run `eval $(opam env)` after creating or switching opam switches.

**Build failures**: Check that you're using OCaml 5.2.0 or later with `ocaml --version` (or `dune exec -- ocaml --version` with Dune).

### Getting Help

- Report issues at [github.com/raven-ml/raven/issues](https://github.com/raven-ml/raven/issues)
