# Norn

MCMC sampling with automatic gradients for OCaml, powered by [Rune](../rune/)

Norn provides Hamiltonian Monte Carlo and NUTS samplers that leverage Rune's
automatic differentiation. You supply an unnormalized log-density function and
an initial position; Norn handles gradient computation, trajectory integration,
and Stan-style window adaptation of step size and mass matrix. Common workflows
are one-line calls, while the kernel API gives full control over the sampling
pipeline.

## Quick Start

Sample from a 2D Gaussian with NUTS:

```ocaml
open Nx

let () =
  Rng.run ~seed:42 @@ fun () ->
  let f64 = Nx.float64 in

  (* Target: N([3; -1], [[1, 0.8]; [0.8, 1]]) *)
  let mu = Nx.create f64 [| 2 |] [| 3.0; -1.0 |] in
  let prec = Nx.create f64 [| 2; 2 |] [| 5.0; -4.0; -4.0; 5.0 |] in
  let log_prob x =
    let d = Nx.sub x mu in
    let d_col = Nx.reshape [| 2; 1 |] d in
    Nx.mul_s (Nx.squeeze (Nx.matmul (Nx.matrix_transpose d_col) (Nx.matmul prec d_col))) (-0.5)
  in

  let init = Nx.zeros f64 [| 2 |] in
  let result = Norn.nuts ~n:1000 log_prob init in

  let mean = Nx.mean ~axes:[ 0 ] result.samples in
  Printf.printf "posterior mean: %s\n" (Nx.data_to_string mean);
  Printf.printf "accept rate:   %.2f\n" result.stats.accept_rate;
  Printf.printf "ESS:           %s\n" (Nx.data_to_string (Norn.ess result.samples))
```

## Features

- **One-line sampling**: `Norn.hmc` and `Norn.nuts` for common workflows
- **Configurable API**: `Norn.sample` with custom kernels via `make_kernel`
- **Automatic gradients**: log-density gradients computed by Rune -- no manual derivatives
- **Symplectic integrators**: `leapfrog`, `mclachlan`, `yoshida`
- **Mass matrix metrics**: `unit_metric`, `diagonal_metric`, `dense_metric`
- **Stan-style adaptation**: dual averaging for step size, Welford estimation for mass matrix
- **Diagnostics**: effective sample size (`ess`) and split R-hat (`rhat`)

## Examples

- **01-sampling-basics** -- Sample from a correlated 2D Gaussian with NUTS
- **02-bayesian-regression** -- Bayesian linear regression with credible intervals
- **03-diagnostics** -- Multi-chain convergence checking with ESS and R-hat

## Contributing

See the [Raven monorepo README](../README.md) for guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
