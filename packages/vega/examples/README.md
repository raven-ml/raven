# Vega Examples

Learn Vega through progressively complex examples. Start with `01-basic-optimizers`
and work through the numbered examples in order.

## Examples

| Example | Concept | Key Functions |
|---------|---------|---------------|
| [`01-basic-optimizers`](./01-basic-optimizers/) | Minimize a quadratic with SGD, Adam, AdamW | `init`, `step`, `Schedule.constant` |
| [`02-composing-transforms`](./02-composing-transforms/) | Build custom optimizers from primitives | `chain`, `scale_by_adam`, `clip_by_norm`, `update` |
| [`03-learning-rate-schedules`](./03-learning-rate-schedules/) | Explore warmup, cosine decay, one-cycle | `Schedule.warmup_cosine_decay`, `Schedule.one_cycle`, `Schedule.join` |

## Running Examples

All examples can be run with:

```bash
dune exec packages/vega/examples/<name>/main.exe
```

For example:

```bash
dune exec packages/vega/examples/01-basic-optimizers/main.exe
```

## Quick Reference

### Basic Optimizer

```ocaml
open Vega

let lr = Schedule.constant 0.01 in
let tx = adam lr in
let st = ref (init tx param) in
for _ = 1 to steps do
  let p, s = step !st ~grad ~param:!param in
  param := p; st := s
done
```

### Custom Optimizer via chain

```ocaml
let tx =
  Vega.chain [
    Vega.clip_by_norm 1.0;
    Vega.scale_by_adam ();
    Vega.add_decayed_weights ~rate:(Vega.Schedule.constant 0.01) ();
    Vega.scale_by_learning_rate lr;
  ]
```

### Learning Rate Schedule

```ocaml
let lr =
  Vega.Schedule.warmup_cosine_decay
    ~init_value:0.0 ~peak_value:0.001
    ~warmup_steps:1000 ~decay_steps:9000 ()
in
let tx = Vega.adam lr
```
