# Vega Examples

Learn Vega through progressively complex examples. Start with `01-basic-optimizers`
and work through the numbered examples in order.

## Examples

| Example | Concept | Key Functions |
|---------|---------|---------------|
| [`01-basic-optimizers`](./01-basic-optimizers/) | Minimize a quadratic with SGD, Adam, AdamW | `sgd_step`, `adam_step`, `adamw_step`, `lr` |
| [`02-training-steps`](./02-training-steps/) | A training step over a record of parameters | `clip_by_global_norm`, `adam_ptree`, `Nx.Ptree.instantiate` |
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
let p = Nx.Ptree.tensor in
let st = ref (Vega.adam_init p !params) in
for _ = 1 to steps do
  let params', st' =
    Vega.adam_step p ~lr:(Vega.lr 0.01) !st ~params:!params ~grads
  in
  params := params';
  st := st'
done
```

### A Training Step

```ocaml
let step (params, st) =
  let grads = Vega.clip_by_global_norm model ~max_norm:1.0 (gradients params) in
  Vega.adamw_step model ~lr:(Vega.lr 1e-3) ~weight_decay:0.01 st ~params ~grads
```

### Learning Rate Schedule

```ocaml
let sched =
  Vega.Schedule.warmup_cosine_decay
    ~init_value:0.0 ~peak_value:0.001
    ~warmup_steps:1000 ~decay_steps:9000 ()
in
Vega.adam_step model ~lr:(sched st.step) st ~params ~grads
```
