# Getting Started

This guide covers installation, key concepts, and two complete examples:
learning XOR and classifying MNIST digits.

## Installation

<!-- $MDX skip -->
```bash
opam install kaun
```

Or build from source:

<!-- $MDX skip -->
```bash
git clone https://github.com/raven-ml/raven
cd raven && dune build kaun
```

## Key Concepts

**Layer.** A layer is a record `{ init; apply }`. `init` creates fresh
parameters and state. `apply` runs the forward pass. Layers compose with
`Layer.sequential` (homogeneous float pipelines) and `Layer.compose`
(heterogeneous, e.g. embedding to dense).

**Ptree.** A `Ptree.t` is a tree of tensors. Dict nodes hold named
subtrees, list nodes hold ordered subtrees, and leaves hold tensors.
Parameters and state are both `Ptree.t` values — plain data you can
inspect, map, serialize, and load.

**Layer.vars.** A `vars` bundles `params` (trainable), `state`
(non-trainable, e.g. batch norm running statistics), and a `dtype`
witness.

**Train.** `Train.make` pairs a model with an optimizer. `Train.init`
creates the initial training state. `Train.fit` trains over a `Data.t`
pipeline. `Train.predict` runs inference.

**Data.** `Data.t` is a lazy, composable iterator. Build from tensors or
arrays, shuffle, batch, map, and feed to `Train.fit`.

**Optim.** An optimizer combines a learning-rate schedule with an update
rule. Schedules are functions `int -> float`.

## Example: XOR

The XOR problem is the simplest non-linear classification task. This
example trains a small network to learn it.

<!-- $MDX skip -->
```ocaml
open Kaun

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->

  (* XOR dataset: 4 examples, 2 features each *)
  let x = Nx.create Nx.Float32 [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |] in
  let y = Nx.create Nx.Float32 [| 4; 1 |] [| 0.; 1.; 1.; 0. |] in

  (* Model: 2 -> 4 -> 1 with tanh activation *)
  let model =
    Layer.sequential
      [
        Layer.linear ~in_features:2 ~out_features:4 ();
        Layer.tanh ();
        Layer.linear ~in_features:4 ~out_features:1 ();
      ]
  in

  (* Create a trainer: model + optimizer *)
  let trainer =
    Train.make ~model
      ~optimizer:(Optim.adam ~lr:(Optim.Schedule.constant 0.01) ())
  in

  (* Initialize training state (model vars + optimizer state) *)
  let st = Train.init trainer ~dtype:Nx.Float32 in

  (* Train for 1000 steps on the same data *)
  let st =
    Train.fit trainer st
      ~report:(fun ~step ~loss _st ->
        if step mod 200 = 0 then
          Printf.printf "step %4d  loss %.6f\n" step loss)
      (Data.repeat 1000 (x, fun pred -> Loss.binary_cross_entropy pred y))
  in

  (* Predict *)
  let pred = Train.predict trainer st x |> Nx.sigmoid in
  Printf.printf "\npredictions (expected 0 1 1 0):\n";
  for i = 0 to 3 do
    Printf.printf "  [%.0f, %.0f] -> %.3f\n"
      (Nx.item [ i; 0 ] x)
      (Nx.item [ i; 1 ] x)
      (Nx.item [ i; 0 ] pred)
  done
```

Key points:

- `Data.repeat 1000 (x, loss_fn)` creates a pipeline that yields the
  same `(input, loss_fn)` pair 1000 times.
- The loss function `fun pred -> Loss.binary_cross_entropy pred y`
  receives the model output and computes a scalar loss.
- `Train.predict` runs in evaluation mode (no dropout, no state
  updates).

## Example: MNIST

A convolutional network for handwritten digit classification using the
built-in MNIST dataset loader.

<!-- $MDX skip -->
```ocaml
open Kaun

let batch_size = 64
let epochs = 3
let lr = 0.001

let model =
  Layer.sequential
    [
      Layer.conv2d ~in_channels:1 ~out_channels:16 ();
      Layer.relu ();
      Layer.max_pool2d ~kernel_size:(2, 2) ();
      Layer.conv2d ~in_channels:16 ~out_channels:32 ();
      Layer.relu ();
      Layer.max_pool2d ~kernel_size:(2, 2) ();
      Layer.flatten ();
      Layer.linear ~in_features:(32 * 7 * 7) ~out_features:128 ();
      Layer.relu ();
      Layer.linear ~in_features:128 ~out_features:10 ();
    ]

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->

  Printf.printf "Loading MNIST...\n%!";
  let (x_train, y_train), (x_test, y_test) = Kaun_datasets.mnist () in
  let n_train = (Nx.shape x_train).(0) in
  Printf.printf "  train: %d  test: %d\n%!" n_train (Nx.shape x_test).(0);

  (* Fixed test batches *)
  let test_batches = Data.prepare ~batch_size (x_test, y_test) in

  (* Trainer *)
  let trainer =
    Train.make ~model
      ~optimizer:(Optim.adam ~lr:(Optim.Schedule.constant lr) ())
  in
  let st = ref (Train.init trainer ~dtype:Nx.Float32) in

  for epoch = 1 to epochs do
    (* Shuffle training data each epoch *)
    let train_data =
      Data.prepare ~shuffle:true ~batch_size (x_train, y_train)
      |> Data.map (fun (x, y) ->
             (x, fun logits -> Loss.cross_entropy_sparse logits y))
    in
    let num_batches = n_train / batch_size in
    let tracker = Metric.tracker () in

    st :=
      Train.fit trainer !st
        ~report:(fun ~step ~loss _st ->
          Metric.observe tracker "loss" loss;
          Printf.printf "\r  batch %d/%d  loss: %.4f%!" step num_batches loss)
        train_data;
    Printf.printf "\n%!";

    (* Evaluate on test set *)
    Data.reset test_batches;
    let test_acc =
      Metric.eval
        (fun (x, y) ->
          let logits = Train.predict trainer !st x in
          Metric.accuracy logits y)
        test_batches
    in
    Printf.printf "epoch %d  train_loss: %.4f  test_acc: %.2f%%\n%!" epoch
      (Metric.mean tracker "loss")
      (test_acc *. 100.)
  done
```

Key points:

- `Kaun_datasets.mnist ()` returns `((x_train, y_train), (x_test, y_test))`
  as float32 tensor pairs. Images have shape `[N; 1; 28; 28]` (NCHW),
  labels `[N]`.
- `Data.prepare ~shuffle:key ~batch_size (x, y)` creates a shuffled,
  batched pipeline of tensor pairs.
- `Data.map` attaches the loss function to each batch, producing the
  `(input, loss_fn)` pairs that `Train.fit` expects.
- `Metric.eval` folds a function over a data pipeline and returns the
  mean.
- `Metric.tracker` accumulates running means for reporting.

## Next Steps

- [Layers and Models](../02-layers-and-models/) — full layer catalog, composition patterns, custom layers
- [Training](../03-training/) — optimizers, schedules, losses, data pipelines, custom loops
- [Checkpoints and Pretrained Models](../04-checkpoints-and-pretrained/) — saving, loading, HuggingFace Hub
