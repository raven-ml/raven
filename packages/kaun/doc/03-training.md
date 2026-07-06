# Training

Kaun has no trainer. A training step is a function you write: it composes `Rune.value_and_grad` with one optimizer update, and the training loop is ordinary `Seq` iteration over minibatches. This guide covers each ingredient — the step, optimizers, losses, data, and metrics — and how they compose.

## The Layering

Kaun's library depends only on nx and rune. Optimizers come from [vega](/docs/vega/), an independent package that also sits directly on nx: you add `vega` to your own project's dependencies and compose it in your code. The three libraries meet in the one record type you define — all of them operate on any `Nx.Ptree.S` structure, so there is no adapter layer and nothing to configure.

## Anatomy of a Training Step

Every training step in kaun has the same three lines: a loss closing over the batch, a `value_and_grad`, an optimizer update.

```ocaml
open Kaun

module Mlp = struct
  type t = { l1 : Linear.t; l2 : Linear.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { l1; l2 } =
    { l1 = Linear.map f l1; l2 = Linear.map f l2 }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { l1 = Linear.map2 f p.l1 q.l1; l2 = Linear.map2 f p.l2 q.l2 }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { l1; l2 } =
    Linear.iter f l1;
    Linear.iter f l2

  let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
end

let step (params, ostate) (x, y) =
  let loss p = Loss.softmax_cross_entropy_sparse (Mlp.apply p x) y in
  let l, grads = Rune.value_and_grad (module Mlp) loss params in
  let params, ostate =
    Vega.adamw_step (module Mlp) ~lr:1e-3 ostate ~params ~grads
  in
  ((params, ostate), Nx.item [] l)
```

The state is a pair `(params, ostate)` of ordinary values — parameters of type `Mlp.t`, optimizer state a record of `Mlp.t`-shaped moments. Nothing is hidden: print a gradient leaf, swap the optimizer, or checkpoint everything mid-run.

Because the step is yours, extensions are insertions, not configuration. Gradient clipping goes between the backward pass and the update:

<!-- $MDX skip -->
```ocaml
let grads = Vega.clip_by_global_norm (module Mlp) ~max_norm:1.0 grads in
```

A learning-rate schedule is a function `int -> float` you evaluate at your own step counter:

<!-- $MDX skip -->
```ocaml
let sched =
  Vega.Schedule.cosine_decay ~init_value:1e-3 ~decay_steps:1000 ()
in
let step k (params, ostate) (x, y) =
  ...
  Vega.adamw_step (module Mlp) ~lr:(sched k) ostate ~params ~grads
```

## Optimizers

Vega's structural optimizers step whole parameter structures. Each keeps its state as a small record of parameter-shaped values that your loop threads explicitly:

| Optimizer | Init | Step | State |
|-----------|------|------|-------|
| SGD (+ momentum) | `Vega.sgd_init` | `Vega.sgd_step ~lr ?momentum` | `{ velocity }` |
| Adam | `Vega.adam_init` | `Vega.adam_step ~lr ?b1 ?b2 ?eps` | `{ mu; nu; step }` |
| AdamW | `Vega.adamw_init` | `Vega.adamw_step ~lr ... ?weight_decay` | `{ mu; nu; step }` |

Steps are pure: they consume a state and return `(params', state')`. AdamW's `weight_decay` defaults to `0.01` and is decoupled — applied to the parameters directly rather than through the adaptive scaling.

```ocaml
let () =
  Nx.Rng.run ~seed:0 @@ fun () ->
  let params =
    {
      Mlp.l1 = Linear.init ~inputs:4 ~outputs:16;
      l2 = Linear.init ~inputs:16 ~outputs:3;
    }
  in
  let ostate = Vega.adamw_init (module Mlp) params in
  (* The moments are Mlp.t values, inspectable like the parameters. *)
  Printf.printf "mu.l1.w: %s  step: %d\n"
    (Nx.shape_to_string (Nx.shape ostate.mu.l1.w))
    ostate.step
```

## Losses

A loss maps predictions and targets to a scalar tensor — the shape `Rune.grad` requires of an objective. Classification losses take raw logits and evaluate in log space, so they are finite and accurate at any logit magnitude:

| Loss | Use |
|------|-----|
| `Loss.mse`, `Loss.mae`, `Loss.huber` | regression |
| `Loss.sigmoid_bce` | binary or multi-label classification (logits + 0/1 or soft targets) |
| `Loss.softmax_cross_entropy` | multiclass, dense targets (one-hot or soft) |
| `Loss.softmax_cross_entropy_sparse` | multiclass, int32 class indices |

All reduce with `` `Mean`` by default (keeping the objective's scale independent of batch size); pass `~reduction:`` `Sum`` to add instead.

```ocaml
let () =
  let logits =
    Nx.create Nx.float32 [| 2; 3 |] [| 2.0; -1.0; 0.5; 0.1; 3.0; -0.2 |]
  in
  let labels = Nx.create Nx.int32 [| 2 |] [| 0l; 1l |] in
  let l = Loss.softmax_cross_entropy_sparse logits labels in
  Printf.printf "loss = %.4f\n" (Nx.item [] l)
```

## Data

A dataset is a tensor — or a pair of tensors — whose axis 0 indexes examples. `Data.batches` and `Data.batches2` cut it into a standard `Seq.t` of minibatches; there is no iterator type of its own, so `Seq.map`, `Seq.take`, `Seq.fold_left` compose any further transformation:

```ocaml
let () =
  let x = Nx.create Nx.float32 [| 10; 2 |] (Array.init 20 float_of_int) in
  let y = Nx.create Nx.int32 [| 10 |] (Array.init 10 Int32.of_int) in
  Data.batches2 ~batch_size:4 (x, y)
  |> Seq.iter (fun (xb, yb) ->
      Printf.printf "batch: x %s  y %s\n"
        (Nx.shape_to_string (Nx.shape xb))
        (Nx.shape_to_string (Nx.shape yb)))
  (* [4; 2]/[4], [4; 2]/[4], [2; 2]/[2] — the last batch is smaller;
     pass ~drop_last:true to drop it. *)
```

Without shuffling, batches are views of the data, not copies, in order. With `~shuffle:true`, each *traversal* of the sequence draws a fresh permutation from the ambient RNG scope — so iterating the same sequence once per epoch reshuffles every epoch, `batches2` permutes inputs and targets together, and running the loop under `Nx.Rng.run` makes the whole schedule of permutations reproducible:

<!-- $MDX skip -->
```ocaml
Nx.Rng.run ~seed:42 @@ fun () ->
let data = Data.batches2 ~shuffle:true ~batch_size:128 (train_x, train_y) in
let state = ref (params, Vega.adamw_init (module Mlp) params) in
for _epoch = 1 to 10 do
  data |> Seq.iter (fun batch -> state := fst (step !state batch))
done
```

The `kaun.datasets` library provides MNIST, Fashion-MNIST, and CIFAR-10 loaders returning plain tensors ready for `Data.batches2`.

## Metrics

Metrics map a batch of predictions and integer labels to a plain `float` — evaluation summaries, never differentiated (train against `Loss`). Multiclass metrics share the sparse-label convention of `Loss.softmax_cross_entropy_sparse`: logits (or probabilities) in, int32 class indices as labels, argmax over the last axis as the predicted class:

```ocaml
let () =
  let logits =
    Nx.create Nx.float32 [| 4; 3 |]
      [| 2.0; 0.1; 0.3; 0.2; 1.5; 0.1; 0.1; 0.2; 3.0; 1.0; 0.5; 0.2 |]
  in
  let labels = Nx.create Nx.int32 [| 4 |] [| 0l; 1l; 2l; 1l |] in
  Printf.printf "accuracy: %.2f\n" (Metric.accuracy logits labels);
  Printf.printf "macro F1: %.2f\n" (Metric.f1 logits labels)
```

Available: `accuracy`, `top_k_accuracy`, `confusion_matrix`, `precision`, `recall`, `f1` (each `` `Macro`` or `` `Micro``), and `auc_roc`.

Metrics are pure and hold no state; aggregation across batches is your fold. Batch means of `accuracy`, `top_k_accuracy`, and micro-averaged scores weighted by batch size equal the dataset value:

```ocaml
let () =
  Nx.Rng.run ~seed:0 @@ fun () ->
  let params =
    {
      Mlp.l1 = Linear.init ~inputs:4 ~outputs:16;
      l2 = Linear.init ~inputs:16 ~outputs:3;
    }
  in
  let test_x = Nx.randn Nx.float32 [| 100; 4 |] in
  let test_y = Nx.create Nx.int32 [| 100 |] (Array.init 100 (fun i -> Int32.of_int (i mod 3))) in
  let correct, total =
    Data.batches2 ~batch_size:32 (test_x, test_y)
    |> Seq.fold_left
         (fun (correct, total) (x, y) ->
           let n = (Nx.shape x).(0) in
           let acc = Metric.accuracy (Mlp.apply params x) y in
           (correct +. (acc *. float_of_int n), total + n))
         (0., 0)
  in
  Printf.printf "test accuracy: %.2f%%\n"
    (100. *. correct /. float_of_int total)
```

**Warning.** `auc_roc` and macro-averaged scores do not decompose over batches: averaging per-batch values is not the dataset-level metric. Compute them once over the full evaluation set.

## Train vs. Eval Mode

There is no `model.train()`/`model.eval()`; mode is an explicit `~training` argument on the functions that care:

- `Dropout.apply ~rate ~training` — training draws a fresh mask and rescales; eval is the identity.
- `Batch_norm.apply p stats ~training` — training normalizes with batch statistics and returns updated running statistics; eval uses the running statistics. Thread the statistics through the loop with `Rune.value_and_grad_aux` — see [Layers and Models](02-layers-and-models/) for the full pattern.

A model forward that takes `~training` serves both phases; evaluation is the same function with `~training:false`.

## Going Further

Because the step is an ordinary function of ordinary values, rune's other transformations apply directly: per-sample gradients are `Rune.vmap2` of `Rune.grad` over the batch (see the [rune transformations guide](/docs/rune/transformations/)), and `Rune.value_and_grad_aux` threads any auxiliary output — predictions for logging, updated statistics — out of the objective.

For complete programs, see [`examples/02-mnist`](https://github.com/raven-ml/raven/tree/main/packages/kaun/examples/02-mnist) (MLP, AdamW, accuracy evaluation) and [`examples/03-mnist-cnn`](https://github.com/raven-ml/raven/tree/main/packages/kaun/examples/03-mnist-cnn) (CNN, dropout, checkpointing).

## Next Steps

- [Checkpoints and Pretrained Models](04-checkpoints-and-pretrained/) — persisting parameters, optimizer state, and counters
- [PyTorch Comparison](05-pytorch-comparison/) — the same concepts in PyTorch vocabulary
