# Norn

Norn provides MCMC sampling with automatic gradients for OCaml. You supply an unnormalized log-density function and an initial position; Norn handles gradient computation via Rune, trajectory integration, and Stan-style window adaptation. One-line convenience functions cover common workflows, while the kernel API gives full control over integrators, metrics, and adaptation.

## Features

- **One-line sampling** -- `Norn.hmc` and `Norn.nuts` with automatic adaptation
- **Configurable API** -- `Norn.sample` with custom kernels via `make_kernel`
- **Automatic gradients** -- log-density gradients computed by Rune
- **Symplectic integrators** -- `leapfrog`, `mclachlan`, `yoshida`
- **Mass matrix metrics** -- `unit_metric`, `diagonal_metric`, `dense_metric`
- **Stan-style adaptation** -- dual averaging for step size, Welford estimation for mass matrix
- **Diagnostics** -- effective sample size (`ess`) and split R-hat (`rhat`)

## Quick Start

<!-- $MDX skip -->
```ocaml
open Nx

let () =
  Rng.run ~seed:42 @@ fun () ->
  let f64 = Nx.float64 in

  (* Target: N([3; -1], I) *)
  let mu = Nx.create f64 [| 2 |] [| 3.0; -1.0 |] in
  let log_prob x =
    let d = Nx.sub x mu in
    Nx.mul_s (Nx.sum (Nx.square d)) (-0.5)
  in

  let init = Nx.zeros f64 [| 2 |] in
  let result = Norn.nuts ~n:1000 log_prob init in

  let mean = Nx.mean ~axes:[ 0 ] result.samples in
  Printf.printf "posterior mean: %s\n" (Nx.data_to_string mean);
  Printf.printf "accept rate:   %.2f\n" result.stats.accept_rate;
  Printf.printf "ESS:           %s\n" (Nx.data_to_string (Norn.ess result.samples))
```

## Next Steps

- [Getting Started](01-getting-started/) -- installation, first sampler, the kernel API
- [Adaptation and Diagnostics](02-adaptation-and-diagnostics/) -- warmup windows, ESS, R-hat
- [Advanced Usage](03-advanced-usage/) -- custom integrators, metrics, and monitoring
- [PyMC Comparison](04-pymc-comparison/) -- mapping from Python's PyMC/BlackJAX to Norn
