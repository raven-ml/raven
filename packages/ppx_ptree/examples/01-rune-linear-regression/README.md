# Rune linear regression

This example derives an `Nx.Ptree.S` implementation for a parameter record and
uses it directly with `Rune.grad` and `Rune.jit2`. The complete gradient and
parameter update are compiled together.

Run it from the repository root:

```sh
dune exec packages/ppx_ptree/examples/01-rune-linear-regression/main.exe
```

The loss should converge toward zero and the learned parameters toward
`w = [2.0; -1.0; 0.5]` and `b = 0.3`.
