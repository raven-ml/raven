# todo

goalpost: mnist notebook in quill with rune + hugin

## simplify

- remove bigarray_ext (what's the best way to implement our C backend?)
- remove device type from rune interface

## current

- gemma kaun example 
- add bf16, fp8, bool data types
  - add actual support in operations
  - replace mask and cond operations with bool tensors
- fft
  - update backend ops to support given dtype
  - fix tests
- linear algebra functions  
  - fix tests
- jit
- vmap
- jvp

## alpha1

docs
- website examples tested with mdx
- review website+docs examples systematically.

feature requests:
- make nx c backend usable with nx (global backend?)
- complete linear algebra suite (cholesky, einsum, qr, svd, inv, solve, norm, eig, eigh, etc.)
- forward mode ad (jvp)
- vmap (that compose with jvp!)
- fft for soundml

fix:
- near-zero formatting issues on some platform (opam-ci)
- hang/deadlock during opam-ci (need to test commited fix with opam-ci)
- rune debug handler causing malloc in opam-ci

docs
- generate docs with odoc3 + dream-style html rewriting to integrate in www/

notes
- think of using effects for prngs, does it simplify ux?

## post-alpha1

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

docs/website
- favicon
- more tutorials

notes
- use upstream metal library when insulated from camlkit
- (?) not sure we need non-polymorphic functions for perf of where, we should benchmark
- add no_grad and detach
- we can make jit composable by re raising all the effects (but what does it mean to write grad(jit(f)))?? What are the semantics in jax?
