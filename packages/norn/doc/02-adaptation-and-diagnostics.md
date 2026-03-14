# Adaptation and Diagnostics

Norn uses Stan-style window adaptation during warmup to tune step size and
mass matrix automatically. After sampling, diagnostics like ESS and R-hat
help you assess whether the chain has converged.

## Window Adaptation

When you call `Norn.nuts`, `Norn.hmc`, or `Norn.sample`, the first
`num_warmup` iterations (default `n / 2`) are discarded as warmup. During
warmup, Norn adapts two quantities:

1. **Step size** -- via dual averaging (Nesterov 2009)
2. **Mass matrix** -- via regularized Welford variance estimation

Warmup is divided into three phases following Stan's scheme:

| Phase | What adapts | Description |
|-------|-------------|-------------|
| Initial fast | Step size only | Short burn-in to find a reasonable step size |
| Slow windows | Step size + mass matrix | Doubling windows that collect samples for Welford estimation. At the end of each window, the mass matrix is updated and step size is reset |
| Final fast | Step size only | Short phase to re-tune step size for the final metric |

The slow windows double in length (e.g., 25, 50, 100, ...) so more samples
contribute to later mass matrix estimates, which are more reliable as the
chain moves closer to the typical set.

## Step Size Adaptation

Step size is tuned via dual averaging to reach a target acceptance rate.
The default targets are:

- HMC: `target_accept = 0.65`
- NUTS: `target_accept = 0.80`

You can override the target:

<!-- $MDX skip -->
```ocaml
let result = Norn.nuts ~n:1000 ~target_accept:0.90 log_prob init
```

Higher target acceptance rates produce smaller step sizes, which give more
accurate trajectories at the cost of more computation. Values between 0.6
and 0.9 work well for most problems.

You can also set the initial step size:

<!-- $MDX skip -->
```ocaml
let result = Norn.nuts ~n:1000 ~step_size:0.1 log_prob init
```

The final adapted step size is available in `result.stats.step_size`.

## Mass Matrix Adaptation

The mass matrix (inverse metric) controls the shape of the momentum
distribution. A well-chosen mass matrix makes the sampler's kinetic energy
match the target's geometry, improving mixing.

During slow windows, Norn collects position samples and estimates the
inverse mass matrix as a diagonal covariance using the Welford online
algorithm with shrinkage regularization. At the end of each slow window:

1. The diagonal inverse mass matrix is computed from accumulated statistics
2. A `diagonal_metric` is constructed from the estimate
3. The Welford accumulator is reset for the next window
4. Step size dual averaging is reset to re-tune for the new metric

After warmup, the metric is frozen and used for all sampling iterations.

## Controlling Warmup

Set `num_warmup` explicitly when the default (`n / 2`) is too few or too many:

<!-- $MDX skip -->
```ocaml
(* More warmup for a difficult posterior *)
let result = Norn.nuts ~n:1000 ~num_warmup:2000 log_prob init

(* Less warmup when the posterior is simple *)
let result = Norn.nuts ~n:1000 ~num_warmup:100 log_prob init
```

## Effective Sample Size

Autocorrelated MCMC samples contain less information than independent samples.
The effective sample size (ESS) estimates how many independent samples the
chain is worth:

<!-- $MDX skip -->
```ocaml
let result = Norn.nuts ~n:2000 log_prob init in
let n_eff = Norn.ess result.samples in
Printf.printf "ESS: %s\n" (Nx.data_to_string n_eff)
```

`ess` takes a matrix of shape `[n; dim]` and returns a vector of shape `[dim]`
with the ESS for each parameter. It uses autocorrelation with the initial
monotone sequence estimator.

Rules of thumb:
- ESS > 100 per parameter is often sufficient for posterior means
- ESS > 400 is preferred for tail quantiles
- Low ESS relative to `n` suggests poor mixing -- consider reparameterization
  or a different metric

## Split R-hat

R-hat measures convergence by comparing within-chain and between-chain variance.
It requires multiple chains:

<!-- $MDX skip -->
```ocaml
open Nx

let () =
  Rng.run ~seed:42 @@ fun () ->
  let f64 = Nx.float64 in
  let log_prob x = Nx.mul_s (Nx.sum (Nx.square x)) (-0.5) in

  (* Run 4 chains from different starting points *)
  let chains =
    Array.init 4 (fun i ->
        let init = Nx.mul_s (Nx.ones f64 [| 3 |]) (Float.of_int (i - 2)) in
        let result = Norn.nuts ~n:500 log_prob init in
        result.samples)
  in

  let r = Norn.rhat chains in
  Printf.printf "R-hat: %s\n" (Nx.data_to_string r)
```

`rhat` takes an array of chains, each of shape `[n; dim]`, and returns
shape `[dim]`. It uses the split R-hat variant (each chain is split in half
before comparison).

Interpretation:
- R-hat close to 1.0 indicates convergence
- R-hat > 1.01 suggests the chains have not mixed
- R-hat > 1.1 is a strong signal of non-convergence

## Checking Convergence

A practical convergence check combines multiple diagnostics:

<!-- $MDX skip -->
```ocaml
let check_convergence (results : Norn.result array) =
  let chains = Array.map (fun r -> r.Norn.samples) results in
  let r = Norn.rhat chains in

  (* Check R-hat for all parameters *)
  let max_rhat = Nx.item [] (Nx.max r) in
  if max_rhat > 1.01 then
    Printf.printf "WARNING: max R-hat = %.3f (chains have not converged)\n"
      max_rhat;

  (* Check ESS for each chain *)
  Array.iteri
    (fun i result ->
      let n_eff = Norn.ess result.Norn.samples in
      let min_ess = Nx.item [] (Nx.min n_eff) in
      Printf.printf "chain %d: min ESS = %.0f, divergences = %d\n" i min_ess
        result.stats.num_divergent)
    results;

  (* Check divergences *)
  let total_div =
    Array.fold_left (fun acc r -> acc + r.Norn.stats.num_divergent) 0 results
  in
  if total_div > 0 then
    Printf.printf "WARNING: %d divergent transitions (consider reparameterization)\n"
      total_div
```

## Next Steps

- [Advanced Usage](../03-advanced-usage/) -- custom integrators, metrics, and monitoring
- [Getting Started](../01-getting-started/) -- basic usage and the kernel API
- [PyMC Comparison](../04-pymc-comparison/) -- mapping from Python's PyMC/BlackJAX to Norn
