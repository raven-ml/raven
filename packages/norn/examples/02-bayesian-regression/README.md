# `02-bayesian-regression`

Bayesian linear regression on synthetic data. Generates noisy observations from
`y = 2x + 1`, defines a Gaussian likelihood with normal priors, and uses NUTS
to infer the posterior over slope and intercept.

```bash
dune exec packages/norn/examples/02-bayesian-regression/main.exe
```

## What You'll Learn

- Building a log-posterior from likelihood and prior
- Interpreting posterior means and 95% credible intervals
- Using the configurable `Norn.sample` API with `Norn.nuts_kernel`

## Key Functions

| Function      | Purpose                                           |
| ------------- | ------------------------------------------------- |
| `nuts`        | One-line NUTS sampling with automatic adaptation  |
| `sample`      | Configurable sampling with a user-provided kernel |
| `nuts_kernel` | Construct a NUTS kernel with explicit parameters  |
| `ess`         | Effective sample size per parameter               |

## How It Works

1. Generate 50 data points from `y = 2x + 1 + N(0, 0.5)`.
2. Define `log_posterior` as Gaussian log-likelihood plus `N(0, 10)` priors.
3. Run `Norn.nuts ~n:2000` to draw posterior samples.
4. Compute posterior means and 95% credible intervals from the samples.
5. Re-run with `Norn.sample` + `Norn.nuts_kernel` to show the configurable API.

## Try It

1. Reduce `n_data` to 10 and observe wider credible intervals.
2. Use a tighter prior `N(0, 1)` and see how it biases the posterior toward zero.
3. Replace `Norn.nuts` with `Norn.hmc ~num_leapfrog:30` and compare.

## Next Steps

Continue to [03-diagnostics](../03-diagnostics/) to learn about multi-chain
convergence analysis with ESS and R-hat.
