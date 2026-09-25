# Optimizers

Vega has no optimizer object. A training step is a function of the parameters
and the optimizer state: compute the gradients, transform them, step. Each
stage is an ordinary function call, so composing optimizers is writing that
function.

## A Training Step

<!-- $MDX skip -->
```ocaml
let sched =
  Vega.Schedule.warmup_cosine_decay ~init_value:0.0 ~peak_value:1e-3
    ~warmup_steps:1000 ~decay_steps:9000 ()

let step (params, st) =
  let loss, grads = Rune.value_and_grad model objective params in
  let grads = Vega.clip_by_global_norm model ~max_norm:1.0 grads in
  let params, st =
    Vega.adamw_step model ~lr:(sched st.step) ~weight_decay:0.01 st ~params
      ~grads
  in
  (loss, (params, st))
```

The gradients flow through each stage in order:

1. `Rune.value_and_grad` differentiates the loss, giving gradients with the
   parameters' structure
2. `clip_by_global_norm` rescales all gradients together so that their joint
   L2 norm is at most 1
3. `adamw_step` updates Adam's moments, derives the rate from the schedule at
   the state's step counter, and applies the update with decoupled weight
   decay

Everything in `step` is tensor arithmetic, so the same function runs eagerly
and compiles under `Rune.jit` ([States Are Structures](#states-are-structures)).

## Gradient Transformations

A gradient transformation maps a gradient structure to another, and goes
between the backward pass and the step:

| Function | Description |
|----------|-------------|
| `clip_by_global_norm p ~max_norm g` | Rescale all leaves together so their joint L2 norm is at most `max_norm` |
| `clip_by_value p ~max g` | Clamp every element to `[-max, max]` |
| `global_norm p g` | The joint L2 norm, read on the host for logging |
| `Loss_scale.unscale p ls g` | Divide by a float16 loss scale |

Your own transformation is a function over the structure, usually one
`Nx.Ptree.map`. To centralize the gradients of every matrix, for instance:

<!-- $MDX skip -->
```ocaml
let centralize p grads =
  Nx.Ptree.map p
    (fun _ g ->
      if Nx.ndim g < 2 then g
      else
        let axes = List.init (Nx.ndim g - 1) (fun i -> i + 1) in
        Nx.sub g (Nx.mean ~axes ~keepdims:true g))
    grads
```

## Choosing an Optimizer

| Situation | Optimizer |
|-----------|-----------|
| A default for deep networks | `adamw_step` |
| Unstable early training, no warmup | `radam_step` |
| Very large batches | `lamb_step` (adaptive), `lars_step` (momentum) |
| Optimizer memory is the limit | `adafactor_step` (no first moment, factored second), `lion_step` (one average) |
| Sparse or rarely-seen features | `adagrad_step` |
| Recurrent networks, non-stationary objectives | `rmsprop_step` |
| Full-batch, deterministic objectives | `lbfgs_step`, `minimize` |

Weight decay in `adamw_step`, `lamb_step`, `lars_step` and `adan_step` is
decoupled: it is added to the update, where the adaptive scaling does not
touch it, and applies to every float leaf.

## States Are Structures

An optimizer state is a record whose parameter-shaped fields have the
parameters' structure, plus a `step` counter as a scalar `int32` tensor. Each
state type has its structure over the parameters' one: `sgd_ptree`,
`adam_ptree`, `rmsprop_ptree`, `adagrad_ptree`, `adan_ptree`, `lion_ptree`,
`adafactor_ptree` and `lbfgs_ptree`. A state's leaves are named by the field
and the parameter's path:

<!-- $MDX skip -->
```ocaml
Nx.Ptree.visits (Vega.adam_ptree model) st
(* mu.w, mu.b, nu.w, nu.b, step *)
```

Those names are the ones a checkpoint stores, and the structure is what a
compiled step takes. With the state an argument the step consumes, a whole
training step is one program that replays on every call:

<!-- $MDX skip -->
```ocaml
let state = Nx.Ptree.pair model (Vega.adam_ptree model)

let step =
  Rune.jit
    Nx.Ptree.(tensor @-> tensor @-> consumes state @@ returns (pair tensor state))
    (fun inputs targets (params, opt) ->
      let loss, grads =
        Rune.value_and_grad model (objective inputs targets) params
      in
      let grads = Vega.clip_by_global_norm model ~max_norm:1.0 grads in
      let params, opt = Vega.adamw_step model ~lr:(sched opt.step) opt ~params ~grads in
      (loss, (params, opt)))
```

A structure may carry tensors that are not parameters, such as an RNG key or a
batch of indices. A step passes every leaf whose dtype is not a float through
unchanged, in the parameters and in the state.

## Skipping Overflowed Steps

In float16 training, a step whose gradients overflowed is skipped by selecting
between the updated and the previous parameters, in tensor arithmetic so it
still compiles:

<!-- $MDX skip -->
```ocaml
let finite = Vega.Loss_scale.grads_finite model grads in
let params', st' = Vega.adam_step model ~lr st ~params ~grads in
let params = Nx.Ptree.map2 model (fun _ p p' -> Nx.where finite p' p) params params' in
let st = Nx.Ptree.map2 (Vega.adam_ptree model) (fun _ s s' -> Nx.where finite s' s) st st'
```

`Vega.Loss_scale` adapts the scale itself; its documentation has the whole
loop.

## Next Steps

- [Learning Rate Schedules](03-schedules.md) — decay, warmup, restarts, and composition
- [Getting Started](01-getting-started.md) — parameters as a structure, your first optimizer
- [Optax Comparison](04-optax-comparison.md) — mapping from Python's Optax to Vega
