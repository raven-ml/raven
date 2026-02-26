# Kaun

Neural networks and training utilities for OCaml, built on [Rune](../rune/)

Kaun provides composable layers, optimizers with learning-rate schedules,
automatic differentiation over parameter trees, data pipelines, and a
high-level training loop. It also supports loading pretrained models from
HuggingFace.

## Quick Start

Train a small network on XOR:

```ocaml
open Kaun

let () =
  let rngs = Rune.Rng.key 42 in
  let dtype = Rune.float32 in

  let x = Rune.create dtype [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |] in
  let y = Rune.create dtype [| 4; 1 |] [| 0.; 1.; 1.; 0. |] in

  let model =
    Layer.sequential
      [
        Layer.linear ~in_features:2 ~out_features:4 ();
        Layer.tanh ();
        Layer.linear ~in_features:4 ~out_features:1 ();
      ]
  in
  let trainer =
    Train.make ~model
      ~optimizer:(Optim.adam ~lr:(Optim.Schedule.constant 0.01) ())
  in
  let st = Train.init trainer ~rngs ~dtype in
  let st =
    Train.fit trainer st ~rngs
      ~report:(fun ~step ~loss _st ->
        if step mod 200 = 0 then Printf.printf "step %4d  loss %.6f\n" step loss)
      (Data.repeat 1000 (x, fun pred -> Loss.binary_cross_entropy pred y))
  in
  let pred = Train.predict trainer st x |> Rune.sigmoid in
  for i = 0 to 3 do
    Printf.printf "  [%.0f, %.0f] -> %.3f\n"
      (Rune.item [ i; 0 ] x) (Rune.item [ i; 1 ] x) (Rune.item [ i; 0 ] pred)
  done
```

## Features

- **Layers**: linear, conv1d, conv2d, layer norm, RMS norm, batch norm, embedding, dropout, multi-head attention with RoPE, and all standard activations (relu, gelu, tanh, sigmoid, etc.)
- **Composition**: `Layer.sequential` and `Layer.compose` for building models
- **Optimizers**: SGD, Adam, AdamW, RMSprop, Adagrad with gradient clipping
- **Schedules**: constant, cosine decay, warmup cosine, exponential decay, warmup linear
- **Training**: `Train.fit` iterates over `Data.t` pipelines with early stopping and per-step reporting; `Train.step` for manual control
- **Data pipelines**: lazy, composable iterators with shuffle, batching, and `Data.prepare` for the common (x, y) tensor pair workflow
- **Metrics**: running trackers, dataset evaluation, accuracy, precision, recall, F1
- **Losses**: cross-entropy, sparse cross-entropy, binary cross-entropy, MSE, MAE
- **Parameter trees**: `Ptree.t` for heterogeneous tensor storage, mapping, and serialization
- **Checkpointing**: save/load to SafeTensors format
- **HuggingFace**: download pretrained weights and configs (`kaun.hf`)
- **Datasets**: MNIST and FashionMNIST loaders (`kaun.datasets`)

## Libraries

| Library | opam package | Description |
|---------|-------------|-------------|
| `kaun` | `kaun` | Core: layers, optimizers, training, data, metrics |
| `kaun_hf` | `kaun.hf` | HuggingFace Hub integration |
| `kaun_datasets` | `kaun.datasets` | Dataset loaders (MNIST, FashionMNIST) |

## Examples

- **01-xor** -- Binary classification on XOR with a 2-layer network
- **02-mnist** -- CNN with conv2d, pooling, and multi-epoch training on MNIST
- **03-bert** -- Fine-tune pretrained BERT for sentiment classification
- **04-gpt2** -- Autoregressive text generation with pretrained GPT-2

## Contributing

See the [Raven monorepo README](../README.md) for guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
