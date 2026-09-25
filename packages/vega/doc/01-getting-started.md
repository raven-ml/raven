# Getting Started

This guide shows you how to describe parameters, initialize an optimizer's
state, and run optimization steps.

## Installation

<!-- $MDX skip -->
```bash
opam install vega
```

Or build from source:

<!-- $MDX skip -->
```bash
git clone https://github.com/raven-ml/raven
cd raven && dune build vega
```

Add to your `dune` file:

<!-- $MDX skip -->
```dune
(executable
 (name main)
 (libraries vega nx))
```

## Your First Optimizer

Vega optimizers turn gradients into parameter updates. Here we minimize
`f(x) = 0.5 * ||x||²` (whose gradient is simply `x`) using SGD:

<!-- $MDX skip -->
```ocaml
let () =
  (* The parameters are one tensor; this is its structure *)
  let p = Nx.Ptree.tensor in

  (* Start from x = [5.0; -3.0] *)
  let params = ref (Nx.create Nx.float32 [| 2 |] [| 5.0; -3.0 |]) in

  (* The optimizer's state for these parameters *)
  let st = ref (Vega.sgd_init p !params) in

  for i = 1 to 30 do
    (* A step takes the state, the parameters and the gradients, and returns
       the new parameters and the next state *)
    let params', st' =
      Vega.sgd_step p ~lr:(Vega.lr 0.1) !st ~params:!params ~grads:!params
    in
    params := params';
    st := st';
    if i mod 10 = 0 then
      Printf.printf "step %2d  x = %s\n" i (Nx.to_string !params)
  done
```

Key points:
- Every function over parameters takes their structure first, here
  `Nx.Ptree.tensor`
- `sgd_init p params` creates the state: a zero velocity shaped like the
  parameters, and a step counter
- `Vega.lr 0.1` is the learning rate as the scalar tensor steps take
- The step returns both the updated parameters and the new state; the loop
  threads them

## Parameters as a Structure

A model's parameters are usually a record of tensors. Its structure comes from
a `walk` that names each field, and `Nx.Ptree.instantiate` turns the module
into the `Nx.Ptree.t` every optimizer takes:

<!-- $MDX skip -->
```ocaml
module Linear = struct
  type 'a t = { w : 'a; b : 'a }

  let walk c { w; b } =
    let open Nx.Ptree.Walk in
    let w = field c "w" leaf w in
    let b = field c "b" leaf b in
    { w; b }
end

let model : Nx.float32_t Linear.t Nx.Ptree.t =
  Nx.Ptree.instantiate (module Linear)
```

The gradients have the parameters' structure too, as `Rune.value_and_grad
model loss params` returns them. A step checks that the parameters, the
gradients and the state share one skeleton, and raises `Invalid_argument`
naming the first path at which they differ.

## Using Adam

Replace SGD with Adam for adaptive learning rates. Adam adjusts the effective
step size per element using running moment estimates:

<!-- $MDX skip -->
```ocaml
let st = Vega.adam_init model params in
let params, st = Vega.adam_step model ~lr:(Vega.lr 1e-3) st ~params ~grads
```

`adam_step` takes optional `~b1` (default 0.9), `~b2` (default 0.999), and
`~eps` (default 1e-8). The rest of the training loop is identical: swap the
`*_init` and the `*_step`.

## Optimizers

Each optimizer is an `*_init` that builds its state and an `*_step` that
advances it. Several share a state type.

| Step | Description | State | Key Parameters |
|------|-------------|-------|----------------|
| `sgd_step` | Stochastic gradient descent | `sgd_state` | `~momentum` |
| `lars_step` | SGD scaled per leaf by a trust ratio, for large batches | `sgd_state` | `~momentum`, `~weight_decay`, `~nesterov` |
| `adam_step` | Adam with bias correction | `adam_state` | `~b1`, `~b2`, `~eps` |
| `adamw_step` | Adam with decoupled weight decay | `adam_state` | `~b1`, `~b2`, `~eps`, `~weight_decay` |
| `radam_step` | Rectified Adam | `adam_state` | `~b1`, `~b2`, `~eps` |
| `lamb_step` | AdamW scaled per leaf by a trust ratio, for large batches | `adam_state` | `~b1`, `~b2`, `~eps`, `~weight_decay` |
| `rmsprop_step` | RMSprop | `rmsprop_state` | `~decay`, `~eps`, `~momentum` |
| `adagrad_step` | Adagrad | `adagrad_state` | `~eps` |
| `adan_step` | Adan, an adaptive Nesterov momentum | `adan_state` | `~b1`, `~b2`, `~b3`, `~eps`, `~weight_decay` |
| `lion_step` | Evolved sign momentum | `lion_state` | `~b1`, `~b2` |
| `adafactor_step` | Factored second moments, for memory | `adafactor_state` | `~decay_rate`, `~eps`, `~clipping_threshold` |
| `lbfgs_step` | L-BFGS, for deterministic objectives | `lbfgs_state` | `~lr` (optional), `~max_linesearch_steps` |

Every step but `lbfgs_step` takes `~lr`, a scalar tensor, first after the
structure. L-BFGS evaluates the objective itself; `Vega.minimize` runs it to
convergence.

## Next Steps

- [Optimizers](02-optimizers.md) — training steps, choosing an optimizer, states as structures
- [Learning Rate Schedules](03-schedules.md) — decay, warmup, restarts, and composition
- [Optax Comparison](04-optax-comparison.md) — mapping from Python's Optax to Vega
