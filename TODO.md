# todo

goalpost: mnist notebook in quill with rune + hugin

## current

- wip: improve nx documentation (go through each function, check the implementation, check if there's invariant to document, check if useful to add examples)

- fix nx tests: getting Command got signal SEGV. at randint
- fix plot3d with hugin
- rune (or nx) build system for metal backend on non macOS systems

refactor nx
- allocate buffers outside of the backend?
- not sure we need non-polymorphic functions for perf of where, we should benchmark

## alpha release

nx
- OK

nx-dataset
- OK

nx-io
- OK

hugin
- OK

quill
- editor bugs:
  - minor:rendering an image opens a window?
  - minor: select end of paragraph and delete removes the line break
  - major: cannot create a new paragraph from a an empty paragraph
  - major: delete the whole document preserves hidden elements
  - major: typing ``` creates a new code block with the rest of the document
- save changes to file
- restore removing signature from code blocks (make it optional)
- syntax highlighting for ocaml
- cover more markdown
  - Block_quote
  - Html_block
  - Link_reference_definition
  - List
  - Thematic_break
- make execution output streamable
- make it work on serverless
- run button
- run all

rune
- train mnist
- metal kernels.
- cuda kernels.

docs
- examples
- end-to-end example with quill+rune+hugin
- website

## next

roadmap
- jit support in rune
  - placeholder? not in tinygrad
  - test for kernelize (see kernelize.md)
  - clang jit backend
  - cuda jit backend
- symbolic shapes for jit
- memory planner during lowering (scheduling)
- pmap support in rune

new libs
- dataframe library
