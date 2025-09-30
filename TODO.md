# todo

## alpha1 (gpt2 on cpu)

- pocketfft integration

## beta (jit)

goalpost: jit-compiled gpt2 matching pytorch performance

feature requests:
- vmap (that compose with jvp!)

fix:
- near-zero formatting issues on some platform (opam-ci)
- hang/deadlock during opam-ci (need to test commited fix with opam-ci)
- rune debug handler causing malloc in opam-ci

## v1 (devex+docs)

goalpost: mnist notebook in quill with rune + hugin

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
- generate docs with odoc3 + dream-style html rewriting to integrate in www/
- favicon
- more tutorials

## notes

- use upstream metal library when insulated from camlkit
- (?) not sure we need non-polymorphic functions for perf of where, we should benchmark
- add no_grad and detach
- we can make jit composable by re raising all the effects (but what does it mean to write grad(jit(f)))?? What are the semantics in jax?
- (?) remove bigarray_ext (what's the best way to implement our C backend?)
- (?) think of using effects for prngs, does it simplify ux?
