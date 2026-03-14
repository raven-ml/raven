# `01-sampling-basics`

Your first sampler. This example draws 1000 samples from a 2D correlated
Gaussian using NUTS and prints summary statistics to verify the chain recovered
the true distribution.

```bash
dune exec packages/norn/examples/01-sampling-basics/main.exe
```

## What You'll Learn

- Defining a log-density function for MCMC
- One-line sampling with `Norn.nuts`
- Computing sample mean, variance, and covariance from the output
- Reading basic diagnostics: ESS, acceptance rate, step size, divergences

## Key Functions

| Function | Purpose |
| -------- | ------- |
| `nuts`   | Draw samples using the No-U-Turn Sampler with automatic adaptation |
| `ess`    | Effective sample size per parameter via autocorrelation |

## How It Works

The target is a 2D Gaussian with mean `[2, -1]` and covariance `[[1, 0.8], [0.8, 2]]`.
We define `log_prob` as the unnormalized log-density (the Mahalanobis form), then
call `Norn.nuts ~n:1000 log_prob init`. NUTS handles warmup adaptation (step size
and mass matrix) automatically.

## Try It

1. Increase `~n` to 5000 and observe ESS and variance estimates improve.
2. Start from a bad initial point like `[100.0; 100.0]` -- warmup should still converge.
3. Replace `Norn.nuts` with `Norn.hmc` and compare acceptance rates.

## Next Steps

Continue to [02-bayesian-regression](../02-bayesian-regression/) to see MCMC applied
to a real inference problem.
