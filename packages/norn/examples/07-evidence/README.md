# `07-evidence`

The evidence is the probability of the data under the model. This example
estimates it with nested sampling and tempered SMC on a model whose evidence is
known, and resamples the weighted posterior draws.

```bash
dune exec packages/norn/examples/07-evidence/main.exe
```

## What You'll Learn

- `Nested.run` and `Smc.run` with Hamiltonian and slice moves
- `Evidence.pp`: ln Z, its error and the information H
- Weighted draws: `Weighted.ess` and `Weighted.resample`

## Key Functions

| Function                                             | Purpose                  |
| ---------------------------------------------------- | ------------------------ |
| `Nested.run u ~budget ~prior ~likelihood k live`     | Nested sampling          |
| `Smc.run u ?move ~budget ~prior ~likelihood k start` | Tempered SMC             |
| `Evidence.sample z`                                  | Weighted posterior draws |
| `Weighted.resample u k ~n w`                         | Equal-weight draws       |

## Next Steps

Continue to [08-laplace](../08-laplace/).
