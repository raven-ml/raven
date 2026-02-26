# todo

## alpha3

- bring in talon changes
- consider rename talon to talf
- remove rng module from nx frontend? inline functions? Then Rng submodule in Nx and Rune (but not in the frontend functor) can just be alias to Nx_core.Rng

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

