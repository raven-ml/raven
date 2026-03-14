# `03-diagnostics`

Multi-chain convergence diagnostics. Runs 4 independent NUTS chains on a 3D
Gaussian target, then computes ESS and split R-hat to verify that the chains
have converged and mixed.

```bash
dune exec packages/norn/examples/03-diagnostics/main.exe
```

## What You'll Learn

- Running multiple chains with different seeds
- Computing effective sample size (ESS) per chain
- Computing split R-hat across chains for convergence assessment
- Interpreting diagnostic thresholds (ESS > 100, R-hat < 1.01)

## Key Functions

| Function | Purpose |
| -------- | ------- |
| `nuts`   | Draw samples with automatic adaptation |
| `ess`    | Effective sample size via initial monotone sequence estimator |
| `rhat`   | Split R-hat convergence diagnostic across chains |

## How It Works

1. Define a 3D Gaussian target with different scales per dimension (var = 1, 4, 0.25).
2. Run 4 independent NUTS chains, each with a different random seed.
3. Report per-chain acceptance rate, step size, and divergence count.
4. Compute ESS for each chain and R-hat across chains.
5. Pool all chains for a final posterior summary.

## Interpreting the Output

- **ESS**: The number of effectively independent samples. If ESS is much lower
  than the actual sample count, the chain has high autocorrelation.
- **R-hat**: Measures between-chain vs within-chain variance. Values close to
  1.0 mean the chains agree. Above 1.01 suggests incomplete mixing.

## Try It

1. Reduce `n_samples` to 50 and observe R-hat increase above 1.01.
2. Use a highly correlated target and see ESS drop.
3. Try `Norn.hmc` instead of `Norn.nuts` and compare ESS.

## Further Reading

- [Sampling Basics](../01-sampling-basics/) -- single-chain sampling
- [Bayesian Regression](../02-bayesian-regression/) -- a real inference problem
