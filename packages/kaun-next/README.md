# Kaun-next

Neural networks for OCaml, built on [rune-next](../rune-next/) autodiff.

Kaun-next provides the building blocks for training neural networks —
layers, activations, initializers, losses, data batching, metrics,
checkpoints — as plain records and pure functions. There is no layer
object and no trainer: a model is a typed record you write, and a
training step is a few lines you own end to end.

The glue is `Nx.Ptree.S`, the traversal interface from [nx](../nx/):
[rune-next](../rune-next/) (transformations) and [vega](../vega/)
(optimizers) each sit on nx independently, kaun-next's library depends
only on nx and rune-next, and the three compose in your code through
the one record type you define.

## The Core Idea

A model is a record of layer records, made traversable by three
one-liners. The same traversals serve differentiation (`Rune_next`),
optimization (`Vega`), and checkpointing (`Checkpoint`):

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

  let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
end
```

A training step composes `value_and_grad` with one optimizer update —
no framework in between:

```ocaml
let step (params, ostate) (x, y) =
  let loss p = Loss.softmax_cross_entropy_sparse (Mlp.apply p x) y in
  let l, grads = Rune_next.value_and_grad (module Mlp) loss params in
  let params, ostate =
    Vega.adamw_step (module Mlp) ~lr:1e-3 ostate ~params ~grads
  in
  ((params, ostate), Nx.item [] l)
```

(`Vega` is composed purely in user code — kaun-next does not depend on
it; add `vega` to your own project's dependencies.)

And the training loop is ordinary `Seq` iteration over minibatches:

```ocaml
let state = ref (params, Vega.adamw_init (module Mlp) params) in
Data.batches2 ~shuffle:true ~batch_size:128 (train_x, train_y)
|> Seq.iter (fun batch ->
    let s, l = step !state batch in
    state := s;
    Printf.printf "loss %.4f\n" l)
```

Every piece of training state — parameters, gradients, optimizer
moments — is a value of your record type that you can print, inspect,
checkpoint, or swap.

## Features

- **Layers** — `Linear`, `Conv` (2-D, NCHW), `Embedding`, `Attention`
  (multi-head, causal masking, plus the pure
  `scaled_dot_product_attention` core), `Layer_norm`, `Batch_norm`
  (running statistics as explicit state); each is a parameter record
  with `init`/`make`, `apply`, and traversals
- **Stateless functions** — `Fn` activations (`relu`, `gelu`, `silu`,
  `softmax`, ...), `Pool` (max/avg 2-D pooling), `Dropout` with an
  explicit `~training` flag
- **Initializers** — `Init`: Glorot/Xavier, He/Kaiming, LeCun, generic
  `variance_scaling`; any function of the right type is an initializer
- **Losses** — `Loss`: MSE, MAE, Huber, `sigmoid_bce`,
  `softmax_cross_entropy` (dense or sparse labels), all evaluated in
  log space for stability
- **Data** — `Data.batches`/`batches2` cut in-memory tensors into a
  standard `Seq.t` of minibatches; shuffling reshuffles per epoch and is
  reproducible under `Nx.Rng.run`
- **Metrics** — `Metric`: accuracy, top-k accuracy, confusion matrix,
  precision/recall/F1 (macro/micro), AUC-ROC
- **Checkpoints** — `Checkpoint` saves named parameter structures as
  [safetensors](https://huggingface.co/docs/safetensors/); one file can
  hold model, optimizer state, and counters side by side
- **Optimizers** — from the independent [vega](../vega/) package:
  `adam`, `adamw`, `sgd` steps over the same record structures
  (`Vega.adam_step (module Model) ...`); depend on it from your own
  project
- **Pretrained models** — `kaun-next.hf` downloads HuggingFace Hub
  checkpoints and adapts them (`rename`, `transpose`, `split`) onto your
  own records
- **Datasets** — `kaun-next.datasets`: MNIST, Fashion-MNIST, CIFAR-10
  loaders returning plain tensors

## Quick Start

Train a two-layer network on XOR (the full program is
[`examples/01-xor`](examples/01-xor)):

```ocaml
let () =
  Nx.Rng.run ~seed:42 @@ fun () ->
  let x = Nx.create Nx.float32 [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |] in
  let y = Nx.create Nx.float32 [| 4; 1 |] [| 0.; 1.; 1.; 0. |] in
  let params =
    {
      Mlp.l1 = Linear.init ~inputs:2 ~outputs:8;
      l2 = Linear.init ~inputs:8 ~outputs:1;
    }
  in
  let loss p = Loss.sigmoid_bce (Mlp.apply p x) y in
  let step (params, ostate) =
    let l, grads = Rune_next.value_and_grad (module Mlp) loss params in
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
  Nx.print_data (Fn.sigmoid (Mlp.apply (fst !state) x))
```

```sh
dune exec packages/kaun-next/examples/01-xor/main.exe
```

## Pretrained Models: the GPT-2 Story

Hub checkpoints name and lay out tensors by the exporting framework's
conventions. `kaun-next.hf` loads them as `Checkpoint.t` values and
adapts them checkpoint-to-checkpoint — rename entries, transpose
weights, split fused projections — until they match your model's own
`names`; typed parameters then come out through `Checkpoint.to_params`:

```ocaml
let params =
  Kaun_next_hf.load_checkpoint "gpt2"
  |> Gpt2.of_hf ~n_layer:cfg.n_layer (* split fused c_attn into q/k/v, rename *)
  |> Checkpoint.to_params (module Gpt2.Params) ~like:(Gpt2.make cfg)
```

[`examples/04-gpt2`](examples/04-gpt2) runs this end to end: it defines
GPT-2 as a record of kaun-next layers (~150 lines), loads the real
weights, and generates text.

## Libraries

| Library | opam package | Description |
|---------|--------------|-------------|
| `kaun_next` | `kaun-next` | Layers, losses, data, metrics, checkpoints |
| `kaun_next_hf` | `kaun-next.hf` | HuggingFace Hub download and checkpoint adaptation |
| `kaun_next_datasets` | `kaun-next.datasets` | MNIST, Fashion-MNIST, CIFAR-10 loaders |

## Examples

- [`01-xor`](examples/01-xor) — binary classification with a two-layer
  MLP: the whole model/step/loop pattern in one page
- [`02-mnist`](examples/02-mnist) — MLP on MNIST with `Data.batches2`
  minibatches, AdamW, and `Metric.accuracy` evaluation
- [`03-mnist-cnn`](examples/03-mnist-cnn) — CNN with `Conv`, `Pool` and
  `Dropout`, plus saving and reloading the trained parameters with
  `Checkpoint`
- [`04-gpt2`](examples/04-gpt2) — text generation with pretrained GPT-2
  loaded from the HuggingFace Hub

## Scope and Limitations

- Everything runs eagerly on CPU through [Nx](../nx/); there is no GPU
  backend or JIT yet.
- Layer coverage is deliberately small: no recurrent layers, and
  `Attention` has no rotary embeddings or KV cache (write them from the
  `scaled_dot_product_attention` core when needed). `Conv` is
  im2col-based and not tuned for large inputs.
- `Batch_norm` is single-precision only; the other layers are generic
  over float dtypes.
- `Metric.auc_roc` and macro-averaged scores do not decompose over
  batches — compute them on the full evaluation set (see the `Metric`
  docs).
- `kaun-next.hf` shells out to `curl` for downloads and needs it on
  `PATH`.
- Vectorizing a dropout forward with `Rune_next.vmap` draws the same
  mask for every lane (see the rune-next README); batch through the
  layer's own batch axes instead.

## Contributing

See the [Raven monorepo README](../README.md) for guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
