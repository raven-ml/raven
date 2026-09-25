# `01-basic-optimizers`

Your first optimizer. This example minimizes `f(x) = 0.5 * ||x||²` from a
starting point using SGD, Adam, and AdamW to compare convergence behavior.

```bash
dune exec packages/vega/examples/01-basic-optimizers/main.exe
```

## What You'll Learn

- Naming the parameters' structure: `Nx.Ptree.tensor` for a single tensor
- Creating an optimizer's state with `Vega.sgd_init`, `adam_init`, `adamw_init`
- Running optimization steps with `Vega.sgd_step`, `adam_step`, `adamw_step`
- Passing a constant learning rate with `Vega.lr`
- How different optimizers converge at different rates

## Key Functions

| Function     | Purpose                                                        |
| ------------ | -------------------------------------------------------------- |
| `lr`         | A constant learning rate, as the scalar tensor the steps take  |
| `sgd_init`   | SGD state: a zero velocity and a step counter                  |
| `sgd_step`   | Stochastic gradient descent with optional momentum             |
| `adam_step`  | Adam with bias-corrected moment estimates                      |
| `adamw_step` | Adam with decoupled weight decay                               |

## How It Works

For `f(x) = 0.5 * ||x||²`, the gradient is simply `x`. Each optimizer starts
from `x = [5.0; -3.0]` and runs 50 steps toward the minimum at `[0; 0]`. A
step takes the state, the parameters and the gradients, and returns the new
parameters with the next state; the loop threads both.

- **SGD** with `lr=0.1` converges fastest on this simple problem
- **Adam** with `lr=0.01` uses adaptive per-coordinate learning rates
- **AdamW** adds weight decay, which also helps push parameters toward zero

## Try It

1. Increase the learning rate for Adam and observe the effect on convergence.
2. Add momentum to SGD with `~momentum:0.9` and compare.
3. Try `Vega.lion_step` or `Vega.radam_step`, with `lion_init` or `radam_init`.

## Next Steps

Continue to [02-training-steps](../02-training-steps/) to build a training
step over a structure of several parameters.
