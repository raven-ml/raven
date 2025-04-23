# todo

## alpha release

ndarray
- use Random.State for random functions
- reshape should make a copy
- fix tests marked with todo
- tests for all api

ndarray-cv
- morphology and threshold broken

ndarray-dataset
- OK

ndarray-io
- OK

hugin
- OK

quill
- cover more markdown
- make execution output streamable
- execute code blocks
- native support for ndarray/rune/hugin
- support for visualization with hugin
- fix bugs, enter is broken now
- fix empty output
- make output non codeblock
- printer for hugin figures
- restore removing signature from code blocks (make it optional)

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
