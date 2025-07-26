# todo

goalpost: mnist notebook in quill with rune + hugin

## current

- fft
- linear algebra functions
- jit with xla
- vmap
- jvp

## alpha1

docs
- website examples tested with mdx
- review website+docs examples systematically.

feature requests:
- complete linear algebra suite (cholesky, einsum, qr, svd, inv, solve, norm, eig, eigh, etc.)
- forward mode ad (jvp)
- vmap (that compose with jvp!)
- nx cuda backend
- add bf16, fp8, bool data types
- fft for soundml

fix:
- near-zero formatting issues on some platform (opam-ci)
- hang/deadlock during opam-ci (need to test commited fix with opam-ci)
- rune debug handler causing malloc in opam-ci

docs
- generate docs with odoc3 + dream-style html rewriting to integrate in www/

notes
- think of using effects for prngs, does it simplify ux?
- bad smell: dummy input for init in kaun
- bad smell: the fix for gc collection of tensors during blas operations is hacky, best would be to pass the tensor to the c code, so it can mark it as being used.

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

new libs
- dataframe library
