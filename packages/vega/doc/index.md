# Vega

Vega provides composable gradient-based optimizers for OCaml. Each optimizer is built from small, typed gradient transformations that compose via `chain`. The library depends only on Nx — no autodiff framework is required.

## Features

- **Optimizer aliases** — `adam`, `adamw`, `sgd`, `rmsprop`, `adagrad`, `lamb`, `lion`, `radam`, `lars`, `adan`, `adafactor`
- **Composable primitives** — `scale_by_adam`, `trace`, `add_decayed_weights`, `clip_by_norm`, and more, combined via `chain`
- **Structural steps** — `sgd_step`, `adam_step`, `adamw_step` over any `Nx.Ptree.S` parameter structure; the state is a parameter tree too (`Sgd_state`, `Adam_state`)
- **Jit-compilable steps** — every time-varying scalar is a tensor leaf, so a whole training step compiles as one `Rune.jit2` program
- **Learning rate schedules** — `constant`, `cosine_decay`, `warmup_cosine_decay`, `one_cycle`, `piecewise_constant`, `join`; tensor mirrors (`*_t`) of the same formulas for use inside compiled steps
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

The structural optimizers' state is a parameter tree like the parameters themselves: `Vega.Adam_state.Make (Model)` turns the Adam state over your model into an `Nx.Ptree.S` you can embed in a compiled step's input and output records. Everything that changes across steps — the moments, the bias corrections, the step counter, the learning rate — is a tensor leaf; everything fixed (`b1`, `b2`, `eps`, `weight_decay`) is an ordinary float the compiler captures as a constant. So the forward pass, the backward pass and the update compile into a single program with no host round-trips:

<!-- $MDX skip -->
```ocaml
module Opt = Vega.Adam_state.Make (Model)

type step_in = {
  params : Model.t;
  opt : Opt.t;                      (* [\[@ptree.using Opt\]] with ppx_ptree *)
  inputs : Nx.float32_t;
  targets : (int32, Nx.int32_elt) Nx.t;
}

(* step_out carries params, opt and the loss; both records derive
   map/map2/iter by hand or with ppx_ptree. *)

let train_step { params; opt; inputs; targets } =
  let loss, grads =
    Rune.value_and_grad model (objective inputs targets) params
  in
  let grads = Vega.clip_by_global_norm model ~max_norm:1.0 grads in
  let lr =
    Vega.Schedule.cosine_decay_t ~init_value:1e-3 ~decay_steps:1000 opt.step
  in
  let params, opt = Vega.adamw_step model ~lr opt ~params ~grads in
  { Step_out.params; opt; loss }

(* ~donate:true releases the previous generation's device buffers once the
   call completes — this loop never reads the pre-step state. *)
let step =
  Rune.jit2 ~donate:true (module Step_in) (module Step_out) train_step
```

Looping `step` over batches compiles once and replays: the state flows out and back in as leaves, and the schedule derives from the state's own counter inside the program. `~donate:true` keeps the loop at about two generations of device buffers instead of one per call awaiting collection — it consumes the handles it frees, so leave it off while the loop still reads the state it feeds in (on the CPU device it changes nothing). The same record works for data-parallel `Rune.pmap2` — replicate the parameters and the state, shard the batch.

The scalar schedule family (`int -> float`) remains for eager loops, which evaluate it at their own step counter; the `*_t` mirrors compute the same formulas from a step tensor for compiled loops.

## Next Steps

- [Getting Started](01-getting-started/) — installation, first optimizer, the step/update API
- [Composing Transforms](02-composing-transforms/) — building custom optimizers from primitives
- [Learning Rate Schedules](03-schedules/) — decay, warmup, restarts, and composition
- [Optax Comparison](04-optax-comparison/) — mapping from Python's Optax to Vega
