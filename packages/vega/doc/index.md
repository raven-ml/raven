# Vega

Vega provides gradient-based optimizers for OCaml. An optimizer steps a whole parameter structure, any value with an `Nx.Ptree.t`, and keeps its state in a record of values shaped like the parameters. The library depends only on Nx — no autodiff framework is required.

## Features

- **Optimizers** — `sgd_step`, `lars_step`, `adam_step`, `adamw_step`, `radam_step`, `lamb_step`, `rmsprop_step`, `adagrad_step`, `adan_step`, `lion_step`, `adafactor_step`, and L-BFGS (`lbfgs_step`, `minimize`), each with its `*_init`
- **States are structures** — `sgd_ptree`, `adam_ptree`, `lbfgs_ptree` and the others name every leaf of a state by path, for checkpoints and compiled steps
- **Jit-compilable steps** — every time-varying scalar is a tensor leaf, so a whole training step compiles as one `Rune.jit` program
- **Learning rate schedules** — `constant`, `cosine_decay`, `warmup_cosine_decay`, `one_cycle`, `piecewise_constant`, `join` — tensor arithmetic over a step counter, so one family serves eager and compiled loops alike
- **Gradient transformations** — `clip_by_global_norm`, `clip_by_value`, `global_norm`
- **Loss scaling** — `Loss_scale` for float16 training

## Quick Start

<!-- $MDX skip -->
```ocaml
let () =
  let p = Nx.Ptree.tensor in
  let params = ref (Nx.create Nx.float32 [| 2 |] [| 5.0; -3.0 |]) in
  let st = ref (Vega.adam_init p !params) in
  for i = 1 to 100 do
    (* For f(x) = 0.5 * ||x||², the gradient is x *)
    let params', st' =
      Vega.adam_step p ~lr:(Vega.lr 0.01) !st ~params:!params ~grads:!params
    in
    params := params';
    st := st';
    if i mod 25 = 0 then
      Printf.printf "step %3d  x = %s\n" i (Nx.to_string !params)
  done
```

`Nx.Ptree.tensor` is the structure of a single tensor. A model's parameters are usually a record, whose structure `Nx.Ptree.instantiate` builds from its `walk` ([Getting Started](01-getting-started.md)).

## Jit-Compiled Training Steps

The optimizers take the parameters' structure, an `Nx.Ptree.t` that
`Nx.Ptree.instantiate` builds from the model's module. A state's structure is
built from it: `Vega.adam_ptree model` is the structure of an Adam state over
`model`, with leaf paths `mu.…`, `nu.…` and `step`. Everything that changes
across steps (the moments, the step counter, the learning rate) is a tensor
leaf or derived from one. Everything fixed (`b1`, `b2`, `eps`, `weight_decay`)
is a float the compiler captures as a constant. So the forward pass, the
backward pass and the update compile into one program.

The example below is a fragment: `Model`, `objective`, `inputs` and `targets`
are the reader's, and it needs rune's `Rune.jit` on signatures.

<!-- $MDX skip -->
```ocaml
let model = Nx.Ptree.instantiate (module Model)
let state = Nx.Ptree.pair model (Vega.adam_ptree model)
let sched = Vega.Schedule.cosine_decay ~init_value:1e-3 ~decay_steps:1000 ()

let step =
  Rune.jit
    Nx.Ptree.(tensor @-> tensor @-> consumes state @@ returns (pair tensor state))
    (fun inputs targets (params, opt) ->
      let loss, grads =
        Rune.value_and_grad model (objective inputs targets) params
      in
      let grads = Vega.clip_by_global_norm model ~max_norm:1.0 grads in
      let params, opt =
        Vega.adamw_step model ~lr:(sched opt.step) opt ~params ~grads
      in
      (loss, (params, opt)))
```

Looping `step` over batches compiles once and replays. Each call consumes the
state it is given and returns the next one beside the loss, and the schedule
reads the state's own counter inside the program. The loss is a fresh result,
readable after later calls.

A step checks that the parameters, the gradients and the state share one
skeleton, and raises `Invalid_argument` naming the first path at which they
differ, for example
`Vega.adam_step: the root: length 1 in the gradients, length 2 in the parameters`.
A structure may carry leaves that are not floats (an RNG key, a counter); a
step passes them through unchanged.

Schedules are tensor arithmetic over the counter, so the same schedule drives an
eager loop; `Schedule.eval` reads one at a host step number for logging.

## Next Steps

- [Getting Started](01-getting-started.md) — installation, parameters as a structure, your first optimizer
- [Optimizers](02-optimizers.md) — training steps, choosing an optimizer, states as structures
- [Learning Rate Schedules](03-schedules.md) — decay, warmup, restarts, and composition
- [Optax Comparison](04-optax-comparison.md) — mapping from Python's Optax to Vega
