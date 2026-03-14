# `02-composing-transforms`

Build custom optimizers by composing gradient transformation primitives.
Shows that optimizer aliases like `adamw` are just shorthand for `chain`.

```bash
dune exec packages/vega/examples/02-composing-transforms/main.exe
```

## What You'll Learn

- Recreating `adamw` from primitives using `Vega.chain`
- Adding gradient clipping to any optimizer
- That `chain` is associative (nesting doesn't change behavior)
- Using `Vega.update` + `Vega.apply_updates` for explicit two-step control

## Key Functions

| Function               | Purpose                                          |
| ---------------------- | ------------------------------------------------ |
| `chain`                | Compose gradient transformations sequentially    |
| `scale_by_adam`        | Adam's bias-corrected moment scaling             |
| `add_decayed_weights`  | Decoupled weight decay (add `rate * param`)      |
| `scale_by_learning_rate` | Multiply by `-lr` for gradient descent         |
| `clip_by_norm`         | Rescale updates if L2 norm exceeds a threshold   |
| `clip_by_value`        | Clamp updates element-wise to `[-delta, +delta]` |
| `update`               | Compute raw updates without applying them        |
| `apply_updates`        | Add updates to parameters                        |

## How Composition Works

Gradient transformations are chained left to right. The gradient flows through
each primitive in order:

```
grad → [clip_by_norm] → [scale_by_adam] → [add_decayed_weights] → [scale_by_learning_rate] → updates
```

Since `chain` is associative, you can build reusable sub-chains:

```ocaml
let adaptive = Vega.chain [Vega.scale_by_adam (); Vega.add_decayed_weights ...] in
let tx = Vega.chain [Vega.clip_by_norm 1.0; adaptive; Vega.scale_by_learning_rate lr]
```

## Try It

1. Add `Vega.centralize` at the beginning of the chain and observe the effect.
2. Move `clip_by_norm` after `scale_by_adam` instead of before — does it matter?
3. Try wrapping the chain with `Vega.apply_if_finite` for NaN protection.

## Next Steps

Continue to [03-learning-rate-schedules](../03-learning-rate-schedules/) to
learn about warmup, cosine decay, and schedule composition.
