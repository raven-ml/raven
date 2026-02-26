# Kaun

Kaun is a neural network library for OCaml built on
[Rune](https://github.com/raven-ml/raven/tree/main/rune). It provides
composable layers, parameter trees, optimizers, data pipelines, and a
high-level training loop. Pretrained weights load from the HuggingFace
Hub via SafeTensors.

## Features

- **Composable layers**: `sequential`, `compose`, and custom `{ init; apply }` records
- **Parameter trees**: `Ptree.t` for inspection, serialization, and transformation
- **High-level training**: `Train.fit` with data pipelines, or `Train.step` for manual control
- **Optimizers**: SGD, Adam, AdamW, RMSprop, Adagrad with LR schedules
- **Losses**: cross-entropy, binary cross-entropy, MSE, MAE
- **Metrics**: accuracy, precision, recall, F1, running tracker, dataset evaluation
- **Layers**: linear, conv1d/2d, layer_norm, rms_norm, batch_norm, embedding, dropout, pooling, multi-head attention with GQA and RoPE
- **Checkpointing**: SafeTensors save/load, HuggingFace Hub integration
- **Datasets**: MNIST and Fashion-MNIST loaders

## Quick Start

Train a model on the XOR problem:

<!-- $MDX skip -->
```ocaml
open Kaun

let () =
  let rngs = Rune.Rng.key 42 in
  let dtype = Rune.float32 in
  let x = Rune.create dtype [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |] in
  let y = Rune.create dtype [| 4; 1 |] [| 0.; 1.; 1.; 0. |] in

  let model = Layer.sequential [
    Layer.linear ~in_features:2 ~out_features:4 ();
    Layer.tanh ();
    Layer.linear ~in_features:4 ~out_features:1 ();
  ] in

  let trainer = Train.make ~model
    ~optimizer:(Optim.adam ~lr:(Optim.Schedule.constant 0.01) ())
  in
  let st = Train.init trainer ~rngs ~dtype in
  let st = Train.fit trainer st ~rngs
    (Data.repeat 1000 (x, fun pred -> Loss.binary_cross_entropy pred y))
  in
  let pred = Train.predict trainer st x |> Rune.sigmoid in
  for i = 0 to 3 do
    Printf.printf "[%.0f, %.0f] -> %.3f\n"
      (Rune.item [ i; 0 ] x) (Rune.item [ i; 1 ] x) (Rune.item [ i; 0 ] pred)
  done
```

## Next Steps

- [Getting Started](01-getting-started/) — installation, XOR and MNIST examples, key concepts
- [Layers and Models](02-layers-and-models/) — layer catalog, composition, custom layers
- [Training](03-training/) — optimizers, losses, data pipelines, metrics, custom loops
- [Checkpoints and Pretrained Models](04-checkpoints-and-pretrained/) — SafeTensors, HuggingFace Hub
