# Tooling and Workflow

Modern ML demands rapid iteration, reproducibility, and sharable narratives.  
This chapter establishes a workflow that combines OCaml tooling with Raven’s notebooks and documentation stack.

## Opam Switch and Dependencies

Create an isolated opam switch for the book’s code:

```bash skip
opam switch create raven-book ocaml-base-compiler.5.1.0
eval $(opam env --switch=raven-book)
opam install dune mdx nx rune talon brot sowilo fehu quill
```

`mdx` executes code blocks and promotes outputs. You will run it as part of the build to guarantee that every snippet stays correct.

## Dune Projects

We scaffold examples with dune so they build like production code.  
Here is a minimal `dune` file for a chapter workspace:

```lisp
(executable
 (name train_mlp)
 (libraries nx rune kaun))
```

Run `dune exec bin/train_mlp.exe` to launch experiments reproducibly.

## Quill Notebooks

Quill treats Markdown as the canonical notebook format.

```bash skip
quill serve book/notebooks/vision_transformer.md
```

This opens an interactive environment where Raven libraries are preloaded.  
Use Quill when you want rapid feedback, visualizations via Hugin, or polished narratives for teammates.

## mdBook + mdx

The book itself is rendered with `mdBook`. To keep embedded code honest:

```bash skip
mdx test book/src
mdbook build book
```

`mdx` first executes every OCaml block, updates the Markdown if outputs changed, and fails the build when code no longer compiles.  
`mdbook` then transforms the verified Markdown into HTML documentation.

## Recommended Editor Setup

- **VS Code** with `ocamllabs.ocaml-platform` extension gives Merlin-powered completions.  
- Enable `mdx` language support (e.g., via `realworldocaml.mdx`) for syntax highlighting in `.md` files.
- Install OCamlformat to keep code style consistent; many Raven packages already include configurations.

With our environment in place we are ready to explore the tensor and autodiff layers that power modern ML in Raven.

