# Optax Comparison

This page maps [Optax](https://github.com/google-deepmind/optax) concepts
and API to their Vega equivalents. Both libraries share the same core idea:
optimizers are composable gradient transformations.

## Creating Optimizers

| Optax (Python) | Vega (OCaml) |
|----------------|--------------|
| `optax.sgd(0.1)` | `Vega.sgd (Schedule.constant 0.1)` |
| `optax.sgd(0.1, momentum=0.9)` | `Vega.sgd ~momentum:0.9 (Schedule.constant 0.1)` |
| `optax.adam(1e-3)` | `Vega.adam (Schedule.constant 1e-3)` |
| `optax.adamw(1e-3, weight_decay=0.01)` | `Vega.adamw ~weight_decay:0.01 (Schedule.constant 1e-3)` |
| `optax.rmsprop(1e-3)` | `Vega.rmsprop (Schedule.constant 1e-3)` |
| `optax.adagrad(0.01)` | `Vega.adagrad (Schedule.constant 0.01)` |
| `optax.lamb(1e-3)` | `Vega.lamb (Schedule.constant 1e-3)` |
| `optax.lion(1e-4)` | `Vega.lion (Schedule.constant 1e-4)` |
| `optax.radam(1e-3)` | `Vega.radam (Schedule.constant 1e-3)` |
| `optax.adafactor()` | `Vega.adafactor ()` |

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
let tx = Vega.adam (Vega.Schedule.constant 1e-3) in
let state = Vega.init tx param in
let updates, state = Vega.update state ~grad ~param in
let param = Vega.apply_updates ~param ~updates

(* Or use the convenience function: *)
let param, state = Vega.step state ~grad ~param
```

The key difference: Optax passes `(grads, state, params)` to `tx.update`,
while Vega passes `state ~grad ~param` — the optimizer is baked into the
state at `init` time.

## Chaining Transforms

**Optax:**

```python
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.scale_by_adam(),
    optax.add_decayed_weights(0.01),
    optax.scale_by_learning_rate(1e-3),
)
```

**Vega:**

<!-- $MDX skip -->
```ocaml
let tx =
  Vega.chain [
    Vega.clip_by_norm 1.0;
    Vega.scale_by_adam ();
    Vega.add_decayed_weights ~rate:(Vega.Schedule.constant 0.01) ();
    Vega.scale_by_learning_rate (Vega.Schedule.constant 1e-3);
  ]
```

## Primitives

| Optax | Vega | Notes |
|-------|------|-------|
| `scale(s)` | `scale s` | |
| `scale_by_adam()` | `scale_by_adam ()` | Supports `~nesterov`, `~amsgrad` |
| `scale_by_rms()` | `scale_by_rms ()` | |
| `scale_by_lion()` | `scale_by_lion ()` | |
| `scale_by_radam()` | `scale_by_radam ()` | |
| `scale_by_trust_ratio()` | `scale_by_trust_ratio ()` | |
| `scale_by_factored_rms()` | `scale_by_adafactor ()` | Different name |
| `trace(decay)` | `trace ~decay ()` | |
| `add_decayed_weights(wd)` | `add_decayed_weights ~rate:(Schedule.constant wd) ()` | Vega uses a schedule |
| `clip_by_global_norm(max)` | `clip_by_norm max` | Per-tensor, not global |
| `clip(delta)` | `clip_by_value delta` | |
| `centralize()` | `centralize` | Value, not function |
| `add_noise(eta, gamma)` | `add_noise ~eta ~gamma ()` | `eta` is a schedule in Vega |
| `apply_if_finite(tx)` | `apply_if_finite tx` | |
| `scale_by_learning_rate(lr)` | `scale_by_learning_rate (Schedule.constant lr)` | Vega uses a schedule |
| `scale_by_schedule(fn)` | `scale_by_schedule fn` | |

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
| State type | PyTree of arrays | Typed `('a, 'b) state` |
| Learning rate | Float or schedule | Always `Schedule.t` (`int -> float`) |
| Weight decay rate | Float | `Schedule.t` (dynamic decay) |
| Noise eta | Float | `Schedule.t` (dynamic noise) |
| Gradient clipping | Global norm across all params | Per-tensor norm |
| Parameter trees | Built-in (JAX pytrees) | Handled by Kaun's `Ptree.t` |
| `centralize` | Function call `centralize()` | Value `centralize` (no arguments) |
