# `02-training-steps`

Fit a linear model `y = x . w + b` with a training step over its parameters,
a record of two tensors. Every optimizer runs through the same loop: compute
the gradients, clip them, step.

```bash
dune exec packages/vega/examples/02-training-steps/main.exe
```

## What You'll Learn

- Describing a record of parameters with a `walk` and `Nx.Ptree.instantiate`
- Transforming gradients before a step with `Vega.clip_by_global_norm`
- Swapping optimizers: `adamw_step`, `radam_step`, `lamb_step`, `lion_step`,
  `sgd_step`, each with its `*_init`
- Reading an optimizer state's leaves and their paths with `Nx.Ptree.visits`

## Key Functions

| Function                | Purpose                                                   |
| ----------------------- | --------------------------------------------------------- |
| `Nx.Ptree.instantiate`  | The structure of a record of tensors, from its `walk`     |
| `clip_by_global_norm`   | Rescale all gradients together to a bounded L2 norm       |
| `adamw_step`            | Adam with decoupled weight decay                          |
| `radam_step`            | Rectified Adam: momentum steps until the variance settles |
| `lamb_step`             | AdamW scaled per leaf by a trust ratio                    |
| `lion_step`             | Sign of an interpolated momentum, one rate for every step |
| `adam_ptree`            | The structure of an Adam state over the parameters        |

## How Composition Works

There is no optimizer object. A training step is a function of the parameters
and the optimizer state, and each stage is an ordinary call:

```
grads → clip_by_global_norm → adamw_step → (params', state')
```

Anything that maps gradients to gradients goes before the step; a learning
rate schedule applies to the state's step counter
([03-learning-rate-schedules](../03-learning-rate-schedules/)).

The state is itself a structure over the parameters. Its leaves are named by
the state's field and the parameter's path (`mu.w`, `nu.b`, `step`), the names
a checkpoint stores and a `Rune.jit`-compiled step takes as arguments.

## Try It

1. Remove the clipping and compare the losses.
2. Replace the hand-written gradients with `Rune.value_and_grad model loss`
   after adding `rune` to the example's libraries.
3. Try `Vega.adafactor_step` with `Vega.adafactor_init`.

## Next Steps

Continue to [03-learning-rate-schedules](../03-learning-rate-schedules/) to
learn about warmup, cosine decay, and schedule composition.
