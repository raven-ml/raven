# Getting Started

This guide walks through the whole kaun-next pattern once: define a model as a record, write the training step yourself, run the loop, evaluate. Everything else in the library is a refinement of this page.

## Installation

<!-- $MDX skip -->
```bash
opam install kaun-next vega
```

Or build from source:

<!-- $MDX skip -->
```bash
git clone https://github.com/raven-ml/raven
cd raven && dune build kaun-next
```

Add to your `dune` file:

<!-- $MDX skip -->
```dune
(executable
 (name main)
 (libraries kaun-next rune-next vega nx))
```

Note the layering: kaun-next's library depends only on nx and rune-next. Optimizers come from [vega](/docs/vega/), which you depend on directly and compose in your own code — there is no trainer in between.

## A Model Is a Record

A kaun-next layer is a plain record of tensors with an `apply` function; `Linear.t` holds a weight matrix and an optional bias. A model is a record of layers, made traversable by three one-liners that delegate to each field — the `Nx.Ptree.S` interface shared by the whole Raven ecosystem:

```ocaml
open Kaun_next

module Mlp = struct
  type t = { l1 : Linear.t; l2 : Linear.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { l1; l2 } =
    { l1 = Linear.map f l1; l2 = Linear.map f l2 }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { l1 = Linear.map2 f p.l1 q.l1; l2 = Linear.map2 f p.l2 q.l2 }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { l1; l2 } =
    Linear.iter f l1;
    Linear.iter f l2

  let apply p x = Linear.apply p.l2 (Nx.tanh (Linear.apply p.l1 x))
end
```

`apply` is just a function — no base class, no forward method, no parameter registry. The three traversals are what let `Rune_next` differentiate values of `Mlp.t`, `Vega` step them, and `Checkpoint` save them.

## Initialize Parameters

Constructors draw from the implicit RNG scope; wrap the program in `Nx.Rng.run` for reproducibility. `Linear.init` gives Glorot-uniform weights and zero bias:

```ocaml
let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  (* XOR dataset. *)
  let x =
    Nx.create Nx.float32 [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |]
  in
  let y = Nx.create Nx.float32 [| 4; 1 |] [| 0.; 1.; 1.; 0. |] in

  let params =
    {
      Mlp.l1 = Linear.init ~inputs:2 ~outputs:8;
      l2 = Linear.init ~inputs:8 ~outputs:1;
    }
  in

  (* The objective: logits from the model, binary cross-entropy on top.
     Losses take raw logits and evaluate in log space. *)
  let loss p = Loss.sigmoid_bce (Mlp.apply p x) y in

  (* The training step: value_and_grad + one Adam update. Gradients and
     optimizer moments are values of Mlp.t, like the parameters. *)
  let step (params, ostate) =
    let l, grads = Rune_next.value_and_grad (module Mlp) loss params in
    let params, ostate =
      Vega.adam_step (module Mlp) ~lr:0.05 ostate ~params ~grads
    in
    ((params, ostate), Nx.item [] l)
  in

  (* The loop: plain OCaml. *)
  let state = ref (params, Vega.adam_init (module Mlp) params) in
  for i = 1 to 500 do
    let s, l = step !state in
    state := s;
    if i mod 100 = 0 then Printf.printf "step %4d  loss %.6f\n" i l
  done;

  (* Evaluate. *)
  let pred = Fn.sigmoid (Mlp.apply (fst !state) x) in
  for i = 0 to 3 do
    Printf.printf "[%.0f, %.0f] -> %.3f\n"
      (Nx.item [ i; 0 ] x)
      (Nx.item [ i; 1 ] x)
      (Nx.item [ i; 0 ] pred)
  done
```

This is the complete program — it is [`examples/01-xor`](https://github.com/raven-ml/raven/tree/main/packages/kaun-next/examples/01-xor) nearly verbatim. Three things to notice:

- **Every piece of training state is a value of your type.** Parameters, gradients, and the Adam moments (`ostate.mu`, `ostate.nu`) are all `Mlp.t` values you can print, inspect, checkpoint, or swap.
- **The step is yours.** Want gradient clipping? Insert `Vega.clip_by_global_norm (module Mlp) ~max_norm:1.0 grads` before the update. A learning-rate schedule? Evaluate one at your step counter. Nothing is hidden behind a trainer.
- **`(module Mlp)` is the only plumbing.** The same first-class module drives differentiation, optimization, and (with `names` added) checkpointing.

## Scaling Up: Minibatches

For real datasets, `Data.batches2` cuts paired tensors into a `Seq.t` of minibatches, and the epoch loop is ordinary `Seq` iteration. With `~shuffle:true`, each traversal of the sequence draws a fresh permutation from the RNG scope, so iterating once per epoch reshuffles every epoch:

<!-- $MDX skip -->
```ocaml
let train_x, train_y, test_x, test_y = Kaun_next_datasets.mnist () in
let batches = Data.batches2 ~shuffle:true ~batch_size:128 (train_x, train_y) in
let state = ref (params, Vega.adamw_init (module Mlp) params) in
for _epoch = 1 to 3 do
  batches
  |> Seq.iter (fun (x, y) ->
      let s, _ = step !state (x, y) in
      state := s)
done
```

where `step` now takes the batch as an argument so the loss closes over it:

<!-- $MDX skip -->
```ocaml
let step (params, ostate) (x, y) =
  let loss p = Loss.softmax_cross_entropy_sparse (Mlp.apply p x) y in
  let l, grads = Rune_next.value_and_grad (module Mlp) loss params in
  let params, ostate =
    Vega.adamw_step (module Mlp) ~lr:1e-3 ostate ~params ~grads
  in
  ((params, ostate), Nx.item [] l)
```

[`examples/02-mnist`](https://github.com/raven-ml/raven/tree/main/packages/kaun-next/examples/02-mnist) runs this end to end, reaching ~97% test accuracy in one epoch.

## Next Steps

- [Layers and Models](02-layers-and-models/) — the full layer catalog, nesting records, stateful layers
- [Training](03-training/) — losses, data, metrics, clipping, schedules
- [Checkpoints and Pretrained Models](04-checkpoints-and-pretrained/) — saving, resuming, loading GPT-2 from the Hub
