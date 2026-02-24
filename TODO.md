# todo

## alpha3

- quill integration with merlin for completion (and others?)
- unhappy with how bloated the kaun api is. make a design pass.
- bring in talon changes
- consider rename talon to talf

## alphaX

- close rune grad performances gap (within <2x of pytorch)
- close nx performance gaps (within <2x of numpy)

## beta (jit)

goalpost: jit-compiled gpt2 matching pytorch performance

feature requests:
- vmap (that compose with jvp!)

## v1 (devex+docs)

goalpost: mnist notebook in quill with rune + hugin

hugin
- fix plot3d
- fix contour

quill
- support images (upstream)

docs/website
- generate docs with odoc3 + dream-style html rewriting to integrate in www/
- favicon
- more tutorials

## notes

- (?) not sure we need non-polymorphic functions for perf of where, we should benchmark
- we can make jit composable by re raising all the effects (but what does it mean to write grad(jit(f)))?? What are the semantics in jax?
- (?) think of using effects for prngs, does it simplify ux?
