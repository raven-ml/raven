# Norn Examples

Learn Norn through progressively complex examples. Start with `01-sampling-basics`
and work through the numbered examples in order.

## Examples

| Example | Concept | Key Functions |
|---------|---------|---------------|
| [`01-sampling-basics`](./01-sampling-basics/) | Sample from a correlated Gaussian with NUTS | `nuts`, `ess` |
| [`02-bayesian-regression`](./02-bayesian-regression/) | Bayesian linear regression with posterior inference | `nuts`, `sample`, `nuts_kernel` |
| [`03-diagnostics`](./03-diagnostics/) | Multi-chain convergence diagnostics | `nuts`, `ess`, `rhat` |

## Running Examples

All examples can be run with:

```bash
dune exec packages/norn/examples/<name>/main.exe
```

For example:

```bash
dune exec packages/norn/examples/01-sampling-basics/main.exe
```

## Quick Reference

### One-Line Sampling

```ocaml
let result =
  Nx.Rng.run ~seed:42 @@ fun () ->
  Norn.nuts ~n:1000 log_prob (Nx.zeros Nx.float64 [| dim |])
```

### Configurable Sampling

```ocaml
let result =
  Norn.sample ~n:1000 ~num_warmup:500 log_prob init
    (fun ~step_size ~metric ->
      Norn.nuts_kernel ~step_size ~metric ())
```

### Convergence Diagnostics

```ocaml
let ess = Norn.ess result.samples in
let rhat = Norn.rhat [| chain1.samples; chain2.samples |]
```
