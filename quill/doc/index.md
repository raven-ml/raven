# Quill

Interactive notebooks for OCaml with a markdown-first philosophy.

## Overview

Quill reimagines the notebook experience. Instead of complex JSON formats and cell-based interfaces, Quill notebooks are just markdown files with executable code blocks. This means:

- **Version control friendly**: Plain text files work perfectly with git
- **Editor agnostic**: Write in your favorite editor, execute anywhere
- **Writing-first**: Code flows naturally within your narrative
- **Zero lock-in**: Your notebooks remain readable markdown forever

## Key Features

### Markdown as Notebooks

Your notebooks are `.md` files where OCaml code blocks become executable cells:

    # My Analysis

    Let's analyze some data:

    ```ocaml
    open Nx
    let data = zeros float32 [|3; 3|]
    ```

    The output appears below the code block.

### Integrated with Raven

All Raven libraries are pre-loaded and ready to use:

- Create tensors with Nx
- Plot with Hugin  
- Build models with Rune
- Process images with Sowilo

### Multiple Execution Modes

**CLI Mode**: Execute notebooks from the command line
<!-- $MDX skip -->
```bash
quill eval notebook.md
```

**Server Mode**: Interactive web interface for editing and execution
<!-- $MDX skip -->
```bash
quill serve notebook.md
```

**Watch Mode**: Auto-execute on file changes
<!-- $MDX skip -->
```bash
quill eval --watch notebook.md
```

### Rich Output Display

- Tensors display as formatted arrays
- Plots render inline as images
- Errors are captured and displayed clearly
- Support for both value printing and stdout

## Philosophy

Traditional notebooks force an awkward separation between narrative and code. You write text in markdown cells, then interrupt the flow to add code cells. The result feels more like a REPL transcript than a document.

Quill takes a different approach. Your notebook is a markdown document first, with code that enhances the narrative rather than interrupting it. This makes notebooks that are:

1. **Natural to write**: Focus on explaining your work, not managing cells
2. **Pleasant to read**: Even on GitHub or in a text editor
3. **Simple to maintain**: No hidden state or complex metadata

## Getting Started

1. Create a notebook: Any `.md` file with OCaml code blocks
2. Run it: `quill eval my_notebook.md`
3. Share it: Commit to git, share on GitHub, publish anywhere

For interactive development:
<!-- $MDX skip -->
```bash
quill serve my_notebook.md
```

Then open http://localhost:8080 in your browser.

## Comparison with Jupyter

| Feature         | Quill              | Jupyter               |
| --------------- | ------------------ | --------------------- |
| File format     | Plain markdown     | JSON                  |
| Version control | Native git support | Requires nbdiff tools |
| Editor support  | Any text editor    | Specialized tools     |
| Cell structure  | Natural flow       | Rigid cells           |
| Language        | OCaml-native       | Multi-language        |

## Learn More

- [Getting Started](/docs/quill/getting-started/) - Create your first notebook
