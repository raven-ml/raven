# todo

goalpost: mnist notebook in quill with rune + hugin

## current

- improve mlis (go through each function, apply docs guidelines, check the implementation, check if there's invariant to document, check if useful to add examples)
- generate docs with odoc3 + dream-style html rewriting to integrate in www/
- website examples generated with quill

## alpha release

nx
- ? intermittent abort at test slice different steps.[1] c backend (can't reproduce)
- slow test zeros max size native backend
- intermittent segfault at unsafe_fold native backend

hugin
- fix plot3d
- fix contour

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
- make execution output streamable
- make it work on serverless
- run button
- run all

docs/website
- examples
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

feature requests:
- add bf16, fp8, bool data types
- complete linear algebra suite
- forward mode ad (jvps)
- cuda backend
- fft for soundml

notes
- use upstream metal library when insulated from camlkit
- (?) not sure we need non-polymorphic functions for perf of where, we should benchmark
- add no_grad and detach
- we can make jit composable by re raising all the effects (but what does it mean to write grad(jit(f)))?? What are the semantics in jax?
- can we use
- think of using effects for prngs, does it simplify ux?
- bad smell: dummy input for init in kaun
- bad smell: the fix for gc collection of tensors during blas operations is hacky, best would be to pass the tensor to the c code, so it can mark it as being used.

new libs
- dataframe library
