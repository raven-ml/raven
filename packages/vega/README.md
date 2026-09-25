# Vega

Gradient-based optimizers for OCaml, inspired by [Optax](https://github.com/google-deepmind/optax)

Vega's optimizers step whole parameter structures: any record, list or tree of
tensors with an `Nx.Ptree.t`. An optimizer's state has the shape of the
parameters and is a structure itself, so a training step is a pure function
of the parameters and the state, which checkpoints by path and compiles under
`Rune.jit` as one program.

## Quick Start

Minimize `f(x) = 0.5 * ||x||^2` with Adam:

```ocaml
let () =
  let p = Nx.Ptree.tensor in
  let params = ref (Nx.create Nx.float32 [| 2 |] [| 5.0; -3.0 |]) in
  let st = ref (Vega.adam_init p !params) in
  for i = 1 to 50 do
    (* For f(x) = 0.5 * ||x||^2, the gradient is x *)
    let params', st' =
      Vega.adam_step p ~lr:(Vega.lr 0.01) !st ~params:!params ~grads:!params
    in
    params := params';
    st := st';
    if i mod 10 = 0 then
      Printf.printf "step %2d  x = %s\n" i (Nx.to_string !params)
  done
```

## Features

- **Optimizers**: `sgd_step`, `lars_step`, `adam_step`, `adamw_step`,
  `radam_step`, `lamb_step`, `rmsprop_step`, `adagrad_step`, `adan_step`,
  `lion_step`, `adafactor_step`, and L-BFGS (`lbfgs_step`, `minimize`)
- **States are structures**: `adam_ptree`, `sgd_ptree`, ... name every leaf of
  a state by path, for checkpoints and compiled steps
- **Learning rate schedules**: `constant`, `cosine_decay`, `warmup_cosine_decay`, `one_cycle`, `cosine_decay_restarts`, `piecewise_constant`, `join`
- **Gradient transformations**: `clip_by_global_norm`, `clip_by_value`, `global_norm`
- **Loss scaling** for float16 training: `Loss_scale`
- **No autodiff dependency**: works with Nx directly

## Examples

- **01-basic-optimizers** -- Minimize a quadratic using SGD, Adam, and AdamW
- **02-training-steps** -- A training step over a record of parameters
- **03-learning-rate-schedules** -- Explore warmup, cosine decay, one-cycle, and more

## Contributing

See the [Raven monorepo README](../README.md) for guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
