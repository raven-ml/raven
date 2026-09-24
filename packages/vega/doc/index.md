# Vega

Vega provides composable gradient-based optimizers for OCaml. Each optimizer is built from small, typed gradient transformations that compose via `chain`. The library depends only on Nx — no autodiff framework is required.

## Features

- **Optimizer aliases** — `adam`, `adamw`, `sgd`, `rmsprop`, `adagrad`, `lamb`, `lion`, `radam`, `lars`, `adan`, `adafactor`
- **Composable primitives** — `scale_by_adam`, `trace`, `add_decayed_weights`, `clip_by_norm`, and more, combined via `chain`
- **Structural steps** — `sgd_step`, `adam_step`, `adamw_step` and L-BFGS over any parameter structure, an `Nx.Ptree.t`; each state has a structure too (`sgd_ptree`, `adam_ptree`, `lbfgs_ptree`)
- **Jit-compilable steps** — every time-varying scalar is a tensor leaf, so a whole training step compiles as one `Rune.jit` program
- **Learning rate schedules** — `constant`, `cosine_decay`, `warmup_cosine_decay`, `one_cycle`, `piecewise_constant`, `join` — tensor arithmetic over a step counter, so one family serves eager and compiled loops alike
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
      Printf.printf "step %3d  x = %s\n" i (Nx.to_string !param)
  done
```

## Jit-Compiled Training Steps

The structural optimizers take the parameters' structure, an `Nx.Ptree.t` that
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

- [Getting Started](01-getting-started.md) — installation, first optimizer, the step/update API
- [Composing Transforms](02-composing-transforms.md) — building custom optimizers from primitives
- [Learning Rate Schedules](03-schedules.md) — decay, warmup, restarts, and composition
- [Optax Comparison](04-optax-comparison.md) — mapping from Python's Optax to Vega
