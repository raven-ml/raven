# `05-hmc-and-ensemble`

Two other samplers on a correlated Gaussian: HMC, which tunes one step size,
trajectory length and geometry for all chains, and the ensemble stretch move,
which needs no gradient and tunes nothing.

```bash
dune exec packages/norn/examples/05-hmc-and-ensemble/main.exe
```

## What You'll Learn

- `Hmc.warmup` and the tuned step size and length
- `Ensemble` walkers as chains of the draws
- Effective sample sizes with `Diag.ess_bulk`

## Key Functions

| Function                            | Purpose                 |
| ----------------------------------- | ----------------------- |
| `Hmc.init`, `warmup`, `sample`      | Hamiltonian Monte Carlo |
| `Ensemble.init`, `warmup`, `sample` | The stretch move        |
| `Diag.ess_bulk u d`                 | Effective sample size   |

## Next Steps

Continue to [06-diagnostics](../06-diagnostics/).
