# kaun

Kaun is a neural network library for OCaml built on [rune](/docs/rune/) autodiff. It provides the building blocks for training networks — layers, activations, initializers, losses, data batching, metrics, checkpoints — as plain records and pure functions. There is no layer object and no trainer: a model is a typed record you write, and a training step is a few lines you own end to end.

The glue is `Nx.Ptree.S`, the traversal interface from [nx](/docs/nx/): [rune](/docs/rune/) (transformations) and [vega](/docs/vega/) (optimizers) each sit on nx independently, kaun's library depends only on nx and rune, and the three compose in your code through the one record type you define.

## Features

- **Layers** — `Linear`, `Conv` (2-D, NCHW), `Embedding`, `Attention` (multi-head, causal masking), `Layer_norm`, `Batch_norm` (running statistics as explicit state); each is a parameter record with `init`/`make`, `apply`, and traversals
- **Stateless functions** — `Fn` activations (`relu`, `gelu`, `silu`, `softmax`, ...), `Pool` (max/avg 2-D pooling), `Dropout` with an explicit `~training` flag
- **Initializers** — `Init`: Glorot/Xavier, He/Kaiming, LeCun, generic `variance_scaling`
- **Losses** — `Loss`: MSE, MAE, Huber, `sigmoid_bce`, `softmax_cross_entropy` (dense or sparse labels), all evaluated in log space
- **Data** — `Data.batches`/`batches2` cut in-memory tensors into a standard `Seq.t` of minibatches, with reproducible per-epoch shuffling
- **Metrics** — `Metric`: accuracy, top-k accuracy, confusion matrix, precision/recall/F1, AUC-ROC
- **Checkpoints** — `Checkpoint` saves named parameter structures as [safetensors](https://huggingface.co/docs/safetensors/)
- **Pretrained models** — `kaun.hf` downloads HuggingFace Hub checkpoints and adapts them (`rename`, `transpose`, `split`) onto your own records
- **Datasets** — `kaun.datasets`: MNIST, Fashion-MNIST, CIFAR-10 loaders returning plain tensors

## Quick Start

Train a two-layer network on XOR — the model is a record, the training step is `value_and_grad` plus one optimizer update, and the loop is a plain `for` loop:

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

  let apply p x = Linear.apply p.l2 (Nx.tanh (Linear.apply p.l1 x))
end

let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
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
  let loss p = Loss.sigmoid_bce (Mlp.apply p x) y in
  let step (params, ostate) =
    let l, grads = Rune.value_and_grad (module Mlp) loss params in
    let params, ostate =
      Vega.adam_step (module Mlp) ~lr:0.05 ostate ~params ~grads
    in
    ((params, ostate), Nx.item [] l)
  in
  let state = ref (params, Vega.adam_init (module Mlp) params) in
  for _ = 1 to 500 do
    let s, _ = step !state in
    state := s
  done;
  Printf.printf "%s\n"
    (Nx.to_string (Fn.sigmoid (Mlp.apply (fst !state) x)))
  (* predictions approach [0; 1; 1; 0] *)
```

`Vega` is composed purely in user code — kaun does not depend on it; add `vega` to your own project's dependencies.

## Libraries

| Library | opam package | Description |
|---------|--------------|-------------|
| `Kaun` | `kaun` | Layers, losses, data, metrics, checkpoints |
| `Kaun_hf` | `kaun.hf` | HuggingFace Hub download and checkpoint adaptation |
| `Kaun_datasets` | `kaun.datasets` | MNIST, Fashion-MNIST, CIFAR-10 loaders |

## Next Steps

- [Getting Started](01-getting-started/) — installation and the model/step/loop pattern
- [Layers and Models](02-layers-and-models/) — the layer catalog and models as records
- [Training](03-training/) — the composable training step, data, and metrics
- [Checkpoints and Pretrained Models](04-checkpoints-and-pretrained/) — safetensors, the HuggingFace Hub, GPT-2
- [PyTorch Comparison](05-pytorch-comparison/) — mapping `nn.Module` vocabulary to kaun
