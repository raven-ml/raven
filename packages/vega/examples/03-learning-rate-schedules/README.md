# `03-learning-rate-schedules`

Explore learning rate schedules. Evaluates several schedules at sampled steps,
then uses warmup + cosine decay in an optimization loop.

```bash
dune exec packages/vega/examples/03-learning-rate-schedules/main.exe
```

## What You'll Learn

- That a schedule is simply `int -> float` (step number to learning rate)
- How `constant`, `cosine_decay`, `warmup_cosine_decay`, `one_cycle`, and
  `piecewise_constant` shape the learning rate curve
- Composing schedules end-to-end with `Schedule.join`
- Plugging schedules into optimizers as the last positional argument

## Key Functions

| Function                | Purpose                                          |
| ----------------------- | ------------------------------------------------ |
| `Schedule.constant`     | Fixed learning rate                              |
| `Schedule.cosine_decay` | Cosine annealing to zero (or `alpha * init`)     |
| `Schedule.warmup_cosine_decay` | Linear warmup then cosine decay            |
| `Schedule.one_cycle`    | 1cycle: linear warmup then cosine decay          |
| `Schedule.piecewise_constant` | Step function with boundaries and values   |
| `Schedule.join`         | Sequence schedules end-to-end                    |

## Schedule Shapes

| Schedule | Shape |
| -------- | ----- |
| `constant` | Flat line |
| `cosine_decay` | Smooth decrease following a cosine curve |
| `warmup_cosine_decay` | Ramp up, then smooth decrease |
| `one_cycle` | Ramp up to peak, then cosine back down to near zero |
| `piecewise_constant` | Staircase drops at specified boundaries |

## Try It

1. Change `warmup_steps` in `warmup_cosine_decay` and observe how it affects
   the transition point.
2. Use `Schedule.cosine_decay_restarts` to see periodic warm restarts (SGDR).
3. Write a custom schedule as a plain function: `let my_schedule step = ...`

## Further Reading

- [Composing Transforms](../02-composing-transforms/) — how schedules plug
  into the `chain` API via `scale_by_learning_rate`
