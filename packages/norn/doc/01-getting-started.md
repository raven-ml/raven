# Getting Started

This guide shows you how to sample from a target distribution using Norn's
MCMC samplers.

## Installation

<!-- $MDX skip -->
```bash
opam install norn
```

Or build from source:

<!-- $MDX skip -->
```bash
git clone https://github.com/raven-ml/raven
cd raven && dune build norn
```

Add to your `dune` file:

<!-- $MDX skip -->
```dune
(executable
 (name main)
 (libraries norn nx rune))
```

## Your First Sampler

Norn samplers take three things: a sample count, an unnormalized log-density
function, and an initial position. Here we sample from a 2D standard Gaussian
using NUTS:

<!-- $MDX skip -->
```ocaml
open Nx

let () =
  Rng.run ~seed:42 @@ fun () ->
  let f64 = Nx.float64 in

  (* log p(x) = -0.5 * ||x||^2  (standard Gaussian) *)
  let log_prob x = Nx.mul_s (Nx.sum (Nx.square x)) (-0.5) in

  let init = Nx.zeros f64 [| 2 |] in
  let result = Norn.nuts ~n:1000 log_prob init in

  Printf.printf "samples shape: %s\n"
    (String.concat "x" (List.map string_of_int
       (Array.to_list (Nx.shape result.samples))));
  Printf.printf "accept rate:   %.3f\n" result.stats.accept_rate;
  Printf.printf "divergences:   %d\n" result.stats.num_divergent
```

Key points:
- `log_prob` returns a scalar `Nx.float64_t` (not a float) -- Rune differentiates it automatically
- `init` is the starting position, shape `[dim]`
- `result.samples` has shape `[n; dim]` -- one row per sample
- NUTS adapts trajectory length automatically via U-turn detection

## Understanding the Result

`Norn.nuts` and `Norn.hmc` return a `result` record:

<!-- $MDX skip -->
```ocaml
type result = {
  samples : Nx.float64_t;       (* shape [n; dim] *)
  log_densities : Nx.float64_t; (* shape [n] *)
  stats : stats;
}

type stats = {
  accept_rate : float;    (* mean acceptance rate during sampling *)
  step_size : float;      (* final adapted step size *)
  num_divergent : int;    (* number of divergent transitions *)
}
```

Compute posterior summaries from `result.samples`:

<!-- $MDX skip -->
```ocaml
let mean = Nx.mean ~axes:[ 0 ] result.samples in
let std = Nx.std ~axes:[ 0 ] result.samples in
Printf.printf "mean: %s\n" (Nx.data_to_string mean);
Printf.printf "std:  %s\n" (Nx.data_to_string std)
```

## Using HMC

HMC requires a fixed number of leapfrog steps per transition. It is simpler
than NUTS but requires tuning `num_leapfrog`:

<!-- $MDX skip -->
```ocaml
let result =
  Norn.hmc ~n:1000 ~num_leapfrog:30 log_prob init
```

Default values: `step_size = 0.01`, `target_accept = 0.65`, `num_leapfrog = 20`.
Step size and mass matrix are adapted during warmup regardless of the sampler.

## The Kernel API

For more control, use `Norn.sample` with a kernel constructor. The `make_kernel`
function receives adapted step size and metric at each warmup step:

<!-- $MDX skip -->
```ocaml
let result =
  Norn.sample ~n:1000 log_prob init (fun ~step_size ~metric ->
      Norn.nuts_kernel ~step_size ~metric ())
```

This is equivalent to `Norn.nuts ~n:1000 log_prob init`, but you can customize
the kernel:

<!-- $MDX skip -->
```ocaml
let result =
  Norn.sample ~n:1000 log_prob init (fun ~step_size ~metric ->
      Norn.nuts_kernel ~integrator:Norn.mclachlan ~max_depth:8
        ~step_size ~metric ())
```

The `make_kernel` signature is `step_size:float -> metric:metric -> kernel`.
During warmup, `sample` calls `make_kernel` each step with the latest adapted
values. After warmup, it freezes the final step size and metric for all
sampling steps.

## HMC vs NUTS

| Aspect | HMC | NUTS |
|--------|-----|------|
| Trajectory length | Fixed (`num_leapfrog` steps) | Automatic (U-turn detection) |
| Tuning parameters | `step_size`, `num_leapfrog` | `step_size`, `max_depth` |
| Default target accept | 0.65 | 0.80 |
| Gradient evaluations | `num_leapfrog` per step | Variable, up to `2^max_depth` |
| Best for | Simple, well-conditioned posteriors | General use |

NUTS is the recommended default. Use HMC when you know the optimal trajectory
length or need predictable cost per step.

## Next Steps

- [Adaptation and Diagnostics](../02-adaptation-and-diagnostics/) -- warmup windows, ESS, R-hat
- [Advanced Usage](../03-advanced-usage/) -- custom integrators, metrics, and monitoring
- [PyMC Comparison](../04-pymc-comparison/) -- mapping from Python's PyMC/BlackJAX to Norn
