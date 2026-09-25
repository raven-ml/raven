# Rune linear regression

This example derives the structure of a parameter record with
`[@@deriving ptree]` and trains it with `Rune.grad` under `Rune.jit`. The
record has no type parameter, so the deriver also generates its structure,
`Params.ptree`, which the gradient, the compiled step and the update take.

Run it from the repository root:

```sh
dune exec packages/ppx_ptree/examples/01-rune-linear-regression/main.exe
```

The loss should converge toward zero and the learned parameters toward
`w = [2.0; -1.0; 0.5]` and `b = 0.3`.
