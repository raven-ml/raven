# `01-basic-optimizers`

Your first optimizer. This example minimizes `f(x) = 0.5 * ||x||²` from a
starting point using SGD, Adam, and AdamW to compare convergence behavior.

```bash
dune exec packages/vega/examples/01-basic-optimizers/main.exe
```

## What You'll Learn

- Creating optimizers with `Vega.sgd`, `Vega.adam`, `Vega.adamw`
- Setting a constant learning rate with `Vega.Schedule.constant`
- Initializing per-parameter state with `Vega.init`
- Running optimization steps with `Vega.step`
- How different optimizers converge at different rates

## Key Functions

| Function            | Purpose                                         |
| ------------------- | ----------------------------------------------- |
| `Schedule.constant` | Create a fixed learning rate                    |
| `sgd`               | Stochastic gradient descent with optional momentum |
| `adam`               | Adam with bias-corrected moment estimates       |
| `adamw`             | Adam with decoupled weight decay                |
| `init`              | Create optimizer state matching a parameter     |
| `step`              | Apply one optimization step, return new param and state |

## How It Works

For `f(x) = 0.5 * ||x||²`, the gradient is simply `x`. Each optimizer starts
from `x = [5.0; -3.0]` and runs 50 steps toward the minimum at `[0; 0]`:

- **SGD** with `lr=0.1` converges fastest on this simple problem
- **Adam** with `lr=0.01` uses adaptive per-coordinate learning rates
- **AdamW** adds weight decay, which also helps push parameters toward zero

## Try It

1. Increase the learning rate for Adam and observe the effect on convergence.
2. Add momentum to SGD with `~momentum:0.9` and compare.
3. Try `Vega.lion` or `Vega.radam` as alternative optimizers.

## Next Steps

Continue to [02-composing-transforms](../02-composing-transforms/) to learn
how to build custom optimizers from primitives.
