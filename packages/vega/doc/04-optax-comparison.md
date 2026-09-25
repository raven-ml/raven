# Optax Comparison

This page maps [Optax](https://github.com/google-deepmind/optax) concepts
and API to their Vega equivalents. In both libraries an optimizer's state is a
tree of arrays shaped like the parameters. Optax builds optimizers by chaining
gradient transformations into one `GradientTransformation`; Vega gives each
optimizer as a step function, and a training step composes them by calling
one after the other.

## Creating Optimizers

Each Optax optimizer is a Vega `*_init` and `*_step` pair, and its
hyperparameters are the step's optional arguments. The learning rate moves to
the step, as `~lr`.

| Optax (Python) | Vega (OCaml) |
|----------------|--------------|
| `optax.sgd(0.1)` | `Vega.sgd_step p ~lr:(Vega.lr 0.1)` |
| `optax.sgd(0.1, momentum=0.9)` | `Vega.sgd_step p ~lr:(Vega.lr 0.1) ~momentum:0.9` |
| `optax.adam(1e-3)` | `Vega.adam_step p ~lr:(Vega.lr 1e-3)` |
| `optax.adamw(1e-3, weight_decay=0.01)` | `Vega.adamw_step p ~lr:(Vega.lr 1e-3) ~weight_decay:0.01` |
| `optax.radam(1e-3)` | `Vega.radam_step p ~lr:(Vega.lr 1e-3)` |
| `optax.lamb(1e-3)` | `Vega.lamb_step p ~lr:(Vega.lr 1e-3)` |
| `optax.lars(0.1)` | `Vega.lars_step p ~lr:(Vega.lr 1e-4)` |
| `optax.rmsprop(1e-3)` | `Vega.rmsprop_step p ~lr:(Vega.lr 1e-3)` |
| `optax.adagrad(0.01)` | `Vega.adagrad_step p ~lr:(Vega.lr 0.01)` |
| `optax.adan(1e-3)` | `Vega.adan_step p ~lr:(Vega.lr 1e-3)` |
| `optax.lion(1e-4)` | `Vega.lion_step p ~lr:(Vega.lr 1e-4)` |
| `optax.adafactor(1e-3)` | `Vega.adafactor_step p ~lr:(Vega.lr 1e-3)` |

Defaults can differ. Vega's `lars_step` and `lamb_step` default to a weight
decay of `0.01` and its `adan_step` to `0.02`, where Optax's default to none.
Vega's `lars_step` has no trust coefficient: the paper's η (Optax's
`trust_coefficient`, default `0.001`) is folded into `~lr`, hence the rate
above. `optax.lamb` defaults to `eps=1e-6` where `lamb_step` uses `1e-8`.
Vega's `adafactor_step` factors every leaf of two or more axes, whatever their
sizes, and does not scale updates by the parameters' root mean square.

## Init and Update

**Optax:**

```python
import optax

tx = optax.adam(1e-3)
state = tx.init(params)
updates, state = tx.update(grads, state, params)
params = optax.apply_updates(params, updates)
```

**Vega:**

<!-- $MDX skip -->
```ocaml
let state = Vega.adam_init model params in
let params, state =
  Vega.adam_step model ~lr:(Vega.lr 1e-3) state ~params ~grads
```

`model` is the parameters' structure, an `Nx.Ptree.t`, the counterpart of the
pytree structure JAX infers from the value. A step returns the new parameters
directly, with no separate updates to apply.

## Chaining Transforms

**Optax:**

```python
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(1e-3, weight_decay=0.01),
)
```

**Vega:**

<!-- $MDX skip -->
```ocaml
let step (params, st) grads =
  let grads = Vega.clip_by_global_norm model ~max_norm:1.0 grads in
  Vega.adamw_step model ~lr:(Vega.lr 1e-3) ~weight_decay:0.01 st ~params ~grads
```

Transformations of the gradients are functions applied before the step.

## Gradient Transformations

| Optax | Vega |
|-------|------|
| `clip_by_global_norm(max)` | `clip_by_global_norm p ~max_norm g` |
| `clip(delta)` | `clip_by_value p ~max:delta g` |
| `global_norm(g)` | `global_norm p g` (a host float) |
| `apply_if_finite(tx)` | `Loss_scale.grads_finite p g`, then select with `Nx.where` |
| `scale_by_schedule(fn)`, `scale_by_learning_rate(lr)` | `~lr:(sched st.step)` |

## Schedules

| Optax | Vega |
|-------|------|
| `constant_schedule(lr)` | `Schedule.constant lr` |
| `linear_schedule(init, end, steps)` | `Schedule.linear ~init_value ~end_value ~steps` |
| `cosine_decay_schedule(init, steps)` | `Schedule.cosine_decay ~init_value ~decay_steps ()` |
| `exponential_decay(init, steps, rate)` | `Schedule.exponential_decay ~init_value ~decay_rate ~decay_steps` |
| `polynomial_schedule(init, end, power, steps)` | `Schedule.polynomial_decay ~init_value ~end_value ~decay_steps ~power ()` |
| `warmup_cosine_decay_schedule(...)` | `Schedule.warmup_cosine_decay ~init_value ~peak_value ~warmup_steps ~decay_steps ()` |
| `sgdr_schedule(...)` | `Schedule.cosine_decay_restarts ~init_value ~decay_steps ()` |
| `piecewise_constant_schedule(...)` | `Schedule.piecewise_constant ~boundaries ~values` |
| `join_schedules(...)` | `Schedule.join segments` |

## Key Differences

| Aspect | Optax | Vega |
|--------|-------|------|
| Language | Python/JAX | OCaml/Nx |
| Optimizer | A `GradientTransformation` of `init` and `update` | An `*_init` and a `*_step` function |
| Composition | `optax.chain` | Function application |
| State type | Pytree of arrays | A record per optimizer (`adam_state`, ...) with its structure (`adam_ptree`, ...) |
| Learning rate | Float or schedule | Scalar tensor `~lr`: `Vega.lr v`, or a schedule at `st.step` |
| Parameter trees | Built-in (JAX pytrees) | Any structure, an `Nx.Ptree.t` |
