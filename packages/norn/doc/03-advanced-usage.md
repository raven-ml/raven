# Advanced Usage

This guide covers custom integrators, metrics, kernel composition via
`Norn.sample`, and monitoring sampling progress.

## Integrators

The integrator controls how Hamiltonian dynamics are approximated. Norn
provides three symplectic integrators:

| Integrator | Order | Grad evals/step | Best for |
|-----------|-------|-----------------|----------|
| `leapfrog` | 2nd | 1 | General use (default) |
| `mclachlan` | 2nd | 2 | Higher acceptance on stiff problems |
| `yoshida` | 4th | 3 | High accuracy with fewer steps |

### Leapfrog (default)

The standard velocity Verlet integrator. One gradient evaluation per step,
good balance of accuracy and cost:

<!-- $MDX skip -->
```ocaml
let result =
  Norn.sample ~n:1000 log_prob init (fun ~step_size ~metric ->
      Norn.nuts_kernel ~integrator:Norn.leapfrog ~step_size ~metric ())
```

### McLachlan

McLachlan's two-stage integrator achieves higher acceptance rates than
leapfrog on challenging posteriors at the cost of two gradient evaluations
per step:

<!-- $MDX skip -->
```ocaml
let result =
  Norn.sample ~n:1000 log_prob init (fun ~step_size ~metric ->
      Norn.nuts_kernel ~integrator:Norn.mclachlan ~step_size ~metric ())
```

Use McLachlan when leapfrog produces too many divergences or low acceptance
rates despite adaptation.

### Yoshida

Yoshida's fourth-order integrator is more accurate than leapfrog, allowing
larger step sizes or fewer integration steps. Three gradient evaluations
per step:

<!-- $MDX skip -->
```ocaml
let result =
  Norn.sample ~n:1000 log_prob init (fun ~step_size ~metric ->
      Norn.hmc_kernel ~integrator:Norn.yoshida ~num_leapfrog:10
        ~step_size ~metric ())
```

Yoshida is most useful with HMC where the trajectory length is fixed -- the
higher accuracy lets you use fewer steps for the same trajectory quality.

## Metrics

The metric defines the mass matrix, which shapes the momentum distribution
to match the target geometry. A good metric improves mixing by making the
sampler's kinetic energy reflect the posterior's covariance structure.

### Unit Metric

Identity mass matrix. Momentum sampled from `N(0, I)`. This is the starting
point for adaptation:

<!-- $MDX skip -->
```ocaml
let m = Norn.unit_metric dim
```

### Diagonal Metric

Diagonal mass matrix estimated from the inverse variance of each parameter.
This is what window adaptation produces automatically:

<!-- $MDX skip -->
```ocaml
let f64 = Nx.float64 in
let inv_mass_diag = Nx.create f64 [| 2 |] [| 1.0; 0.01 |] in
let m = Norn.diagonal_metric inv_mass_diag
```

Use a diagonal metric when parameters have very different scales. Adaptation
estimates this automatically, but you can provide your own if you know the
posterior variances.

### Dense Metric

Full inverse mass matrix. Uses Cholesky decomposition for momentum sampling.
Captures correlations between parameters:

<!-- $MDX skip -->
```ocaml
let f64 = Nx.float64 in
let inv_mass =
  Nx.create f64 [| 2; 2 |] [| 1.0; 0.8; 0.8; 1.0 |]
in
let m = Norn.dense_metric inv_mass
```

Dense metrics help with strongly correlated posteriors but are expensive for
high-dimensional problems (`O(dim^2)` storage, `O(dim^3)` Cholesky).

## Composing Kernels with sample

`Norn.sample` is the configurable entry point. The `make_kernel` function
receives the current adapted step size and metric, returning a kernel:

<!-- $MDX skip -->
```ocaml
let result =
  Norn.sample ~n:2000 ~num_warmup:1000 ~target_accept:0.85
    log_prob init (fun ~step_size ~metric ->
      Norn.nuts_kernel ~integrator:Norn.mclachlan ~max_depth:8
        ~step_size ~metric ())
```

This gives you full control over:
- The sampler algorithm (HMC vs NUTS)
- The integrator (leapfrog, mclachlan, yoshida)
- Algorithm-specific parameters (`num_leapfrog`, `max_depth`)
- Step size and metric are provided by adaptation

The `make_kernel` function is called at every warmup step (with updated
adaptation values) and once more with the final values before sampling begins.

## Monitoring with report

The `~report` callback lets you monitor sampling progress. It is called after
each step with the current step number, state, and diagnostics:

<!-- $MDX skip -->
```ocaml
let report ~step state info =
  if step mod 100 = 0 then
    Printf.printf "step %4d  log_p = %.2f  accept = %.3f  steps = %d%s\n"
      step state.Norn.log_density info.Norn.acceptance_rate
      info.num_integration_steps
      (if info.is_divergent then "  DIVERGENT" else "")

let result =
  Norn.sample ~n:1000 ~report log_prob init (fun ~step_size ~metric ->
      Norn.nuts_kernel ~step_size ~metric ())
```

Step numbers are negative during warmup (counting down to zero) and
non-negative during sampling. This makes it easy to distinguish the two
phases:

<!-- $MDX skip -->
```ocaml
let report ~step _state info =
  if step < 0 then
    Printf.printf "warmup %4d  accept = %.3f\n" step info.Norn.acceptance_rate
  else if step mod 100 = 0 then
    Printf.printf "sample %4d  accept = %.3f\n" step info.acceptance_rate
```

## Providing a Known Metric

If you know the posterior covariance from a previous run or analytic
calculation, skip the adaptation overhead by providing the metric directly:

<!-- $MDX skip -->
```ocaml
open Nx

let () =
  Rng.run ~seed:42 @@ fun () ->
  let f64 = Nx.float64 in
  let log_prob x = Nx.mul_s (Nx.sum (Nx.square x)) (-0.5) in
  let init = Nx.zeros f64 [| 2 |] in

  (* Use a known diagonal inverse mass *)
  let inv_mass_diag = Nx.create f64 [| 2 |] [| 1.0; 1.0 |] in
  let metric = Norn.diagonal_metric inv_mass_diag in

  let result =
    Norn.sample ~n:1000 ~num_warmup:200 log_prob init (fun ~step_size ~metric:_ ->
        Norn.nuts_kernel ~step_size ~metric ())
  in
  Printf.printf "accept rate: %.3f\n" result.stats.accept_rate
```

Note that `~metric:_` ignores the adapted metric and uses the fixed one.
Step size is still adapted during warmup.

## Next Steps

- [Getting Started](../01-getting-started/) -- basic usage and the kernel API
- [Adaptation and Diagnostics](../02-adaptation-and-diagnostics/) -- warmup windows, ESS, R-hat
- [PyMC Comparison](../04-pymc-comparison/) -- mapping from Python's PyMC/BlackJAX to Norn
