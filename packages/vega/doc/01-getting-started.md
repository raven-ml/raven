# Getting Started

This guide shows you how to create optimizers, initialize state, and run
optimization steps.

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

Vega optimizers transform gradients into parameter updates. Here we minimize
`f(x) = 0.5 * ||x||²` (whose gradient is simply `x`) using SGD:

<!-- $MDX skip -->
```ocaml
open Vega

let () =
  (* Create an SGD optimizer with learning rate 0.1 *)
  let lr = Schedule.constant 0.1 in
  let tx = sgd lr in

  (* Start from x = [5.0; -3.0] *)
  let param = ref (Nx.create Nx.float32 [| 2 |] [| 5.0; -3.0 |]) in

  (* Initialize optimizer state from the parameter shape *)
  let st = ref (init tx !param) in

  for i = 1 to 30 do
    (* step takes state, gradient, and current param;
       returns (new_param, new_state) *)
    let p, s = step !st ~grad:!param ~param:!param in
    param := p;
    st := s;
    if i mod 10 = 0 then
      Printf.printf "step %2d  x = %s\n" i (Nx.data_to_string !param)
  done
```

Key points:
- `Schedule.constant 0.1` creates a fixed learning rate
- `init tx param` creates optimizer state matching the parameter's shape and dtype
- `step` returns both the updated parameter and the new optimizer state
- The optimizer state must be threaded through each step

## Using Adam

Replace `sgd` with `adam` for adaptive learning rates. Adam adjusts the
effective step size per-parameter using running moment estimates:

<!-- $MDX skip -->
```ocaml
let lr = Vega.Schedule.constant 0.001 in
let tx = Vega.adam lr
```

Adam takes optional parameters `~b1` (default 0.9), `~b2` (default 0.999),
and `~eps` (default 1e-8). The rest of the training loop is identical — just
swap the optimizer.

## The Update API

`step` is a convenience that combines two lower-level operations:

<!-- $MDX skip -->
```ocaml
(* step = update + apply_updates *)
let new_param, new_state = Vega.step state ~grad ~param

(* is equivalent to: *)
let updates, new_state = Vega.update state ~grad ~param in
let new_param = Vega.apply_updates ~param ~updates
```

The two-step API is useful when you need to inspect or modify the raw updates
before applying them (e.g., logging gradient norms, applying custom masks).

## Optimizer Aliases

Vega provides ready-to-use aliases that compose primitives internally:

| Alias | Description | Key Parameters |
|-------|-------------|----------------|
| `sgd` | Stochastic gradient descent | `~momentum`, `~nesterov` |
| `adam` | Adam with bias correction | `~b1`, `~b2`, `~eps` |
| `adamw` | Adam with decoupled weight decay | `~b1`, `~b2`, `~eps`, `~weight_decay` |
| `rmsprop` | RMSprop | `~decay`, `~eps`, `~momentum` |
| `adagrad` | Adagrad | `~eps` |
| `lamb` | LAMB for large-batch training | `~b1`, `~b2`, `~eps`, `~weight_decay` |
| `lion` | Evolved sign momentum | `~b1`, `~b2` |
| `radam` | Rectified Adam | `~b1`, `~b2`, `~eps` |
| `lars` | LARS for large-batch SGD | `~momentum`, `~weight_decay`, `~nesterov` |
| `adan` | Adan with gradient difference | `~b1`, `~b2`, `~b3`, `~eps`, `~weight_decay` |
| `adafactor` | Memory-efficient factored moments | `~b2_decay` |

All aliases take `lr` (a `Schedule.t`) as their last positional argument.
`adafactor` is the exception — it includes its own learning rate schedule
internally.

## Next Steps

- [Composing Transforms](../02-composing-transforms/) — build custom optimizers from primitives
- [Learning Rate Schedules](../03-schedules/) — decay, warmup, restarts, and composition
- [Optax Comparison](../04-optax-comparison/) — mapping from Python's Optax to Vega
