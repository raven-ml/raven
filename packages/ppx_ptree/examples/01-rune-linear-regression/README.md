# Rune linear regression

This example derives the walk of a parameter record with
`[@@deriving ptree]` and trains it with `Rune.grad` under `Rune.jit`. The
record `('a, _) Params.t` holds its weights at the payload `'a` and no fixed
array, so its set is anonymous. `Nx.Ptree.instantiate (module Params)`
describes it at arrays, and the gradient, the compiled step and the
donation at each call take that description.

Run it from the repository root:

```sh
dune exec packages/ppx_ptree/examples/01-rune-linear-regression/main.exe
```

The loss should converge toward zero and the learned parameters toward
`w = [2.0; -1.0; 0.5]` and `b = 0.3`.
