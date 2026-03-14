# PyMC Comparison

This page maps [PyMC](https://www.pymc.io/) and
[BlackJAX](https://github.com/blackjax-devs/blackjax) concepts to their
Norn equivalents. Norn's design is closest to BlackJAX: both provide
functional kernel APIs where the sampler state is explicit and the log-density
function is passed at each step.

## One-Line Sampling

**PyMC:**

```python
import pymc as pm

with pm.Model():
    x = pm.Normal("x", mu=0, sigma=1, shape=2)
    trace = pm.sample(1000, tune=500)
```

**Norn:**

<!-- $MDX skip -->
```ocaml
let log_prob x = Nx.mul_s (Nx.sum (Nx.square x)) (-0.5) in
let init = Nx.zeros Nx.float64 [| 2 |] in
let result = Norn.nuts ~n:1000 ~num_warmup:500 log_prob init
```

PyMC builds a probabilistic model and derives the log-density automatically.
Norn takes the log-density function directly -- you write it yourself or
build it from your model. Rune handles the gradient.

## BlackJAX Kernel API

**BlackJAX:**

```python
import blackjax
import jax

kernel = blackjax.nuts(log_prob, step_size=0.5)
state = kernel.init(jax.numpy.zeros(2))

for _ in range(1000):
    key, subkey = jax.random.split(key)
    state, info = kernel.step(subkey, state)
```

**Norn:**

<!-- $MDX skip -->
```ocaml
let metric = Norn.unit_metric 2 in
let kernel = Norn.nuts_kernel ~step_size:0.5 ~metric () in
let state = ref (kernel.init (Nx.zeros Nx.float64 [| 2 |]) log_prob) in

for _ = 1 to 1000 do
  let new_state, _info = kernel.step !state log_prob in
  state := new_state
done
```

Both use a `{init; step}` pattern. The key difference: BlackJAX threads a
PRNG key explicitly, while Norn uses Nx's RNG context (`Rng.run`).

## Adaptation

**BlackJAX:**

```python
warmup = blackjax.window_adaptation(blackjax.nuts, log_prob)
state, kernel, _ = warmup.run(key, jax.numpy.zeros(2), 1000)
```

**Norn:**

<!-- $MDX skip -->
```ocaml
(* Adaptation is built into sample/nuts/hmc *)
let result = Norn.nuts ~n:1000 ~num_warmup:500 log_prob init

(* Or use sample for control over the kernel *)
let result =
  Norn.sample ~n:1000 ~num_warmup:500 log_prob init (fun ~step_size ~metric ->
      Norn.nuts_kernel ~step_size ~metric ())
```

In BlackJAX, adaptation is a separate step that returns a tuned kernel. In
Norn, adaptation is integrated into `sample` -- it adapts step size and mass
matrix during warmup, then freezes them for sampling.

## Samplers

| PyMC / BlackJAX | Norn | Notes |
|-----------------|------|-------|
| `pm.sample()` (NUTS) | `Norn.nuts ~n log_prob init` | NUTS with adaptation |
| `blackjax.nuts(log_prob, step_size)` | `Norn.nuts_kernel ~step_size ~metric ()` | NUTS kernel |
| `blackjax.hmc(log_prob, step_size, ...)` | `Norn.hmc_kernel ~step_size ~metric ()` | HMC kernel |
| `pm.sample(step=pm.HamiltonianMC(...))` | `Norn.hmc ~n log_prob init` | HMC with adaptation |

## Integrators

| BlackJAX | Norn | Notes |
|----------|------|-------|
| `blackjax.mcmc.integrators.velocity_verlet` | `Norn.leapfrog` | Default, 1 grad eval/step |
| `blackjax.mcmc.integrators.mclachlan` | `Norn.mclachlan` | 2 grad evals/step |
| `blackjax.mcmc.integrators.yoshida` | `Norn.yoshida` | 3 grad evals/step |

Usage comparison:

```python
# BlackJAX
kernel = blackjax.nuts(log_prob, step_size=0.5,
                       integrator=blackjax.mcmc.integrators.mclachlan)
```

<!-- $MDX skip -->
```ocaml
(* Norn *)
let kernel =
  Norn.nuts_kernel ~integrator:Norn.mclachlan ~step_size:0.5 ~metric ()
```

## Metrics (Mass Matrix)

| BlackJAX | Norn | Notes |
|----------|------|-------|
| `blackjax.mcmc.metrics.default_metric(jnp.ones(d))` | `Norn.unit_metric d` | Identity |
| `blackjax.mcmc.metrics.default_metric(inv_mass_diag)` | `Norn.diagonal_metric inv_mass_diag` | Diagonal |
| Dense metric via Cholesky | `Norn.dense_metric inv_mass_matrix` | Full matrix |

## Diagnostics

| PyMC / ArviZ | Norn | Notes |
|--------------|------|-------|
| `az.ess(trace)` | `Norn.ess samples` | Effective sample size |
| `az.rhat(trace)` | `Norn.rhat chains` | Split R-hat |
| `trace.sample_stats["diverging"]` | `result.stats.num_divergent` | Divergence count |
| `trace.sample_stats["accept"]` | `result.stats.accept_rate` | Mean acceptance rate |
| `trace.sample_stats["step_size"]` | `result.stats.step_size` | Final step size |

## State and Info

**BlackJAX state:**

```python
state.position      # current sample
state.logdensity    # log p(x)
state.logdensity_grad  # grad log p(x)
```

**Norn state:**

<!-- $MDX skip -->
```ocaml
state.position         (* Nx.float64_t, shape [dim] *)
state.log_density      (* float *)
state.grad_log_density (* Nx.float64_t, shape [dim] *)
```

**BlackJAX info:**

```python
info.acceptance_rate
info.is_divergent
info.energy
info.num_integration_steps
```

**Norn info:**

<!-- $MDX skip -->
```ocaml
info.acceptance_rate        (* float in [0, 1] *)
info.is_divergent           (* bool *)
info.energy                 (* float *)
info.num_integration_steps  (* int *)
```

## Key Differences

| Aspect | PyMC / BlackJAX | Norn |
|--------|-----------------|------|
| Language | Python / JAX | OCaml / Rune |
| Model definition | Declarative (PyMC) or functional (BlackJAX) | Functional -- write `log_prob` directly |
| Gradients | JAX autodiff | Rune autodiff |
| PRNG | Explicit key splitting (JAX) | Scoped via `Nx.Rng.run` |
| Adaptation | Separate step (BlackJAX) or automatic (PyMC) | Integrated into `sample` |
| Mass matrix output | Diagonal or dense | `metric` record with `sample_momentum`, `kinetic_energy`, `scale` |
| Multi-chain | Built-in (`chains` parameter) | Run multiple calls, combine with `rhat` |
| Trace format | ArviZ InferenceData | `result` record with `samples` matrix |
| Probabilistic DSL | Yes (PyMC) | No -- bring your own log-density |
