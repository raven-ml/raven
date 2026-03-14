# Vega

Vega provides composable gradient-based optimizers for OCaml. Each optimizer is built from small, typed gradient transformations that compose via `chain`. The library depends only on Nx — no autodiff framework is required.

## Features

- **Optimizer aliases** — `adam`, `adamw`, `sgd`, `rmsprop`, `adagrad`, `lamb`, `lion`, `radam`, `lars`, `adan`, `adafactor`
- **Composable primitives** — `scale_by_adam`, `trace`, `add_decayed_weights`, `clip_by_norm`, and more, combined via `chain`
- **Learning rate schedules** — `constant`, `cosine_decay`, `warmup_cosine_decay`, `one_cycle`, `piecewise_constant`, `join`
- **Gradient processing** — clipping, centralization, noise injection
- **Robustness** — `apply_if_finite` skips NaN/Inf updates automatically
- **Serialization** — `state_to_tensors` / `state_of_tensors` for checkpointing

## Quick Start

<!-- $MDX skip -->
```ocaml
open Vega

let () =
  let lr = Schedule.constant 0.01 in
  let tx = adam lr in

  let param = ref (Nx.create Nx.float32 [| 2 |] [| 5.0; -3.0 |]) in
  let st = ref (init tx !param) in

  for i = 1 to 100 do
    (* For f(x) = 0.5 * ||x||², the gradient is x *)
    let p, s = step !st ~grad:!param ~param:!param in
    param := p;
    st := s;
    if i mod 25 = 0 then
      Printf.printf "step %3d  x = %s\n" i (Nx.data_to_string !param)
  done
```

## Next Steps

- [Getting Started](01-getting-started/) — installation, first optimizer, the step/update API
- [Composing Transforms](02-composing-transforms/) — building custom optimizers from primitives
- [Learning Rate Schedules](03-schedules/) — decay, warmup, restarts, and composition
- [Optax Comparison](04-optax-comparison/) — mapping from Python's Optax to Vega
