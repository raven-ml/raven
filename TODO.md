# todo

## alpha release

ndarray
- OK

ndarray-cv
- OK

ndarray-dataset
- OK

ndarray-io
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
- support multiple devices

docs
- examples
- end-to-end example with quill+rune+hugin
- website?

## next

roadmap
- jit support in rune
- pmap support in rune

refactorings
- consider having descriptor in Ndarray_core.t instead of backend to simplify code. can we remove descriptor from backend entirely (unlikely)
- extract automatic device dispatching from run to an ndarray library

improvements
- implement buffer pool for the metal bindings
- better slicing function api
- infix operators
- add ?out argument to all functions? replace inplace functions?

optim
- add support for scalar ops in run - review operations with scalar in rune and replace with scalar variants

new libs
- neural network library on top of rune?
- dataframe library
