# todo

goalpost: mnist notebook in quill with rune + hugin

## current

- kaun: run mnist 
  - EXTREMELY slow: normalization (Rune.div) takes 12s for 60k 28x28 images (should be <1s)
  - Training is 12-18x slower than PyTorch CPU:
    - PyTorch: ~17ms per batch, 2.7 min for 10 epochs (source: PyImageSearch benchmark)
    - Our impl: ~200-300ms per batch, ~62 min for 10 epochs
    - Conv ops are 3-5x slower than PyTorch (15ms vs 3-5ms)
    - Forward+backward pass: ~150ms (should be ~15ms)
  - Transpose creates non-contiguous tensors as expected (zero-copy operation)
  - dataset: API is fine but implementation needs work (fixed loading twice)

## alpha release

nx
- fix intermittent Command got signal ABRT in convolution tests (dev/conv2d)
- fix intermittent Command got signal SEGV. at randint

hugin
- fix plot3d

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

docs/website
- examples
- improve mlis (go through each function, apply docs guidelines, check the implementation, check if there's invariant to document, check if useful to add examples)
- generate docs with odoc3 + dream-style html rewriting to integrate in www/
- more user documentation
- favicon
- end-to-end example with quill+rune+hugin

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

refactorings
- (?) allocate buffers outside of the backend?
- (?) not sure we need non-polymorphic functions for perf of where, we should benchmark

notes
- add no_grad and detach
- we can make jit composable by re raising all the effects (but what does it mean to write grad(jit(f)))?? What are the semantics in jax?

new libs
- dataframe library
- nx cuda backend
- (?) nx blas backend
