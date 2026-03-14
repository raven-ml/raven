# Vega

Composable gradient-based optimizers for OCaml, inspired by [Optax](https://github.com/google-deepmind/optax)

Vega provides typed, per-parameter optimizer primitives that compose via
chaining. Each primitive is a gradient transformation: it takes updates
(gradients) and returns modified updates. Primitives are chained to build
complete optimizers, giving you full control over the optimization pipeline
while common recipes are available as one-line aliases.

## Quick Start

Minimize `f(x) = 0.5 * ||x||^2` with Adam:

```ocaml
open Vega

let () =
  let lr = Schedule.constant 0.01 in
  let tx = adam lr in

  let param = ref (Nx.create Nx.float32 [| 2 |] [| 5.0; -3.0 |]) in
  let st = ref (init tx !param) in

  for i = 1 to 50 do
    (* For f(x) = 0.5 * ||x||^2, the gradient is x *)
    let p, s = step !st ~grad:!param ~param:!param in
    param := p;
    st := s;
    if i mod 10 = 0 then
      Printf.printf "step %2d  x = %s\n" i (Nx.data_to_string !param)
  done
```

## Features

- **Optimizer aliases**: `adam`, `adamw`, `sgd`, `rmsprop`, `adagrad`, `lamb`, `lion`, `radam`, `lars`, `adan`, `adafactor`
- **Composable primitives**: `scale_by_adam`, `scale_by_rms`, `trace`, `add_decayed_weights`, `scale_by_trust_ratio`, and more -- combine via `chain`
- **Learning rate schedules**: `constant`, `cosine_decay`, `warmup_cosine_decay`, `one_cycle`, `cosine_decay_restarts`, `piecewise_constant`, `join`
- **Gradient clipping**: `clip_by_value`, `clip_by_norm`
- **Gradient processing**: `centralize`, `add_noise`
- **Robustness**: `apply_if_finite` skips updates containing NaN/Inf
- **Serialization**: `state_to_tensors` / `state_of_tensors` for checkpointing
- **No autodiff dependency**: works with Nx directly

## Examples

- **01-basic-optimizers** -- Minimize a quadratic using SGD, Adam, and AdamW
- **02-composing-transforms** -- Build custom optimizers from primitives
- **03-learning-rate-schedules** -- Explore warmup, cosine decay, one-cycle, and more

## Contributing

See the [Raven monorepo README](../README.md) for guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
