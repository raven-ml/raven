# Layers and Models

Kaun has no layer abstraction. A layer is a plain record of tensors with an `apply` function; a model is a record of layers with hand-written one-line traversals. This guide covers the built-in layers, the model-as-record pattern, and stateful layers.

## The Layer Pattern

Every parameterized layer module follows the same shape, with `Linear` as the reference:

- a parameter record — `Linear.t` is `{ w; b }` with `w : [| inputs; outputs |]` and an optional bias;
- constructors — `Linear.init ~inputs ~outputs` for the float32 defaults, `Linear.make` for initializer, bias, and dtype control;
- an `apply` function — `Linear.apply p x` is `x @ p.w + p.b`, treating leading axes as batch axes;
- traversals — `map`, `map2`, `iter` satisfying `Nx.Ptree.S`, plus `names` for checkpointing.

```ocaml
open Kaun

let () =
  Nx.Rng.run ~seed:0 @@ fun () ->
  let layer = Linear.init ~inputs:4 ~outputs:2 in
  let x = Nx.randn Nx.float32 [| 8; 4 |] in
  let y = Linear.apply layer x in
  Printf.printf "y has shape %s\n" (Nx.shape_to_string (Nx.shape y))
  (* [8; 2] *)
```

There is nothing else to a layer: `layer.w` is an ordinary tensor you can read, and `{ layer with w = ... }` an ordinary record update.

## The Catalog

| Module | Parameters | Apply |
|--------|-----------|-------|
| `Linear` | `w`, optional `b` | dense map over the last axis |
| `Conv` | `w`, optional `b` | 2-D convolution, NCHW, `` `Valid``/`` `Same`` padding |
| `Embedding` | `table` | token-id to row lookup (int32 ids in, floats out) |
| `Attention` | `q`, `k`, `v`, `out` projections | multi-head self-attention, optional causal mask |
| `Layer_norm` | `gamma`, `beta` | normalization over the last axis |
| `Batch_norm` | `gamma`, `beta` + separate `Stats.t` | normalization over the batch, running statistics |

Stateless companions, all pure functions:

| Module | Contents |
|--------|----------|
| `Fn` | `relu`, `leaky_relu`, `sigmoid`, `tanh`, `gelu`, `gelu_approx`, `silu`, `softplus`, `softmax`, `log_softmax` |
| `Pool` | `max_pool2d`, `avg_pool2d` over the last two axes |
| `Dropout` | `apply ~rate ~training` |
| `Init` | `glorot_*`, `he_*`, `lecun_*`, `variance_scaling`, constants |

Every `apply` is differentiable through rune, in both reverse and forward mode.

## Models Are Records of Layers

Records nest into records, and the traversals delegate field by field. Adding `names` — one name per leaf, prefixed by field — makes the same module usable with `Checkpoint`:

```ocaml
module Mlp = struct
  type t = { l1 : Linear.t; l2 : Linear.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { l1; l2 } =
    { l1 = Linear.map f l1; l2 = Linear.map f l2 }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    { l1 = Linear.map2 f p.l1 q.l1; l2 = Linear.map2 f p.l2 q.l2 }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { l1; l2 } =
    Linear.iter f l1;
    Linear.iter f l2

  let names { l1; l2 } =
    List.map (( ^ ) "l1.") (Linear.names l1)
    @ List.map (( ^ ) "l2.") (Linear.names l2)

  let apply p x = Linear.apply p.l2 (Fn.relu (Linear.apply p.l1 x))
end
```

The `map`/`map2`/`iter` trio is what `Rune.grad`, `Vega.adam_step`, and friends consume; `names` is the extra line `Checkpoint` needs (the `Checkpoint.Named` module type). This scales without new concepts: a transformer block is a record of five layers, a transformer is a record with a `block list` field traversed with `List.map`/`List.map2`/`List.iter`. [`examples/04-gpt2`](https://github.com/raven-ml/raven/tree/main/packages/kaun/examples/04-gpt2) defines all of GPT-2 this way in ~150 lines.

A CNN mixes parameterized and stateless pieces freely, since a forward pass is just function composition:

```ocaml
module Cnn = struct
  type t = { c1 : Conv.t; c2 : Conv.t; fc : Linear.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { c1; c2; fc } =
    { c1 = Conv.map f c1; c2 = Conv.map f c2; fc = Linear.map f fc }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    {
      c1 = Conv.map2 f p.c1 q.c1;
      c2 = Conv.map2 f p.c2 q.c2;
      fc = Linear.map2 f p.fc q.fc;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { c1; c2; fc } =
    Conv.iter f c1;
    Conv.iter f c2;
    Linear.iter f fc

  let apply p ~training x =
    let n = (Nx.shape x).(0) in
    Conv.apply p.c1 x |> Fn.relu
    |> Pool.max_pool2d ~kernel_size:(2, 2)
    |> Conv.apply p.c2 |> Fn.relu
    |> Pool.max_pool2d ~kernel_size:(2, 2)
    |> Nx.reshape [| n; 16 * 5 * 5 |]
    |> Dropout.apply ~rate:0.25 ~training
    |> Linear.apply p.fc
end

let () =
  Nx.Rng.run ~seed:0 @@ fun () ->
  let params =
    {
      Cnn.c1 = Conv.init ~in_channels:1 ~out_channels:8 ~kernel_size:(3, 3);
      c2 = Conv.init ~in_channels:8 ~out_channels:16 ~kernel_size:(3, 3);
      fc = Linear.init ~inputs:(16 * 5 * 5) ~outputs:10;
    }
  in
  let x = Nx.randn Nx.float32 [| 2; 1; 28; 28 |] in
  let logits = Cnn.apply params ~training:false x in
  Printf.printf "logits: %s\n" (Nx.shape_to_string (Nx.shape logits))
  (* [2; 10] *)
```

Note the `~training` flag threaded to `Dropout.apply`: mode is an argument, not a mutable model state. This is [`examples/03-mnist-cnn`](https://github.com/raven-ml/raven/tree/main/packages/kaun/examples/03-mnist-cnn)'s model.

## Attention

`Attention.t` is four `Linear` projections (query, key, value, output). The head count is not a parameter — the projections are `embed_dim × embed_dim` whatever the head count — so `num_heads` is an argument of `apply`, like `eps` of `Layer_norm.apply`:

```ocaml
let () =
  Nx.Rng.run ~seed:0 @@ fun () ->
  let attn = Attention.init ~embed_dim:16 in
  let x = Nx.randn Nx.float32 [| 2; 5; 16 |] in
  (* batch 2, sequence 5 *)
  let y = Attention.apply ~num_heads:4 ~causal:true attn x in
  Printf.printf "y: %s\n" (Nx.shape_to_string (Nx.shape y))
  (* [2; 5; 16] — same shape; causal masks future positions *)
```

`scaled_dot_product_attention` is the pure core — `softmax (q @ kᵀ / sqrt d) @ v`, no parameters, no head bookkeeping. Leading axes broadcast, so stacked heads are just a batch axis. Use it directly for cross-attention, externally projected queries and keys, or custom masking via `?mask`:

```ocaml
let () =
  Nx.Rng.run ~seed:0 @@ fun () ->
  let q = Nx.randn Nx.float32 [| 3; 8 |] in
  let k = Nx.randn Nx.float32 [| 5; 8 |] in
  let v = Nx.randn Nx.float32 [| 5; 4 |] in
  let y = Attention.scaled_dot_product_attention q k v in
  Printf.printf "y: %s\n" (Nx.shape_to_string (Nx.shape y))
  (* [3; 4] — one weighted average of value rows per query row *)
```

There are no rotary embeddings or KV cache; write them from this core when needed.

## Stateful Layers: Batch_norm

`Batch_norm` is the one layer with non-parameter state. It is two structures: trainable parameters (`Batch_norm.t`, the affine `gamma` and `beta` — differentiated and optimized like any other parameters) and running statistics (`Batch_norm.Stats.t`, per-feature mean and variance — never differentiated, updated by every training forward).

`apply` in training mode normalizes with the current batch's statistics and returns updated running statistics; in eval mode it normalizes with the running statistics and returns them unchanged. A training step threads the updated statistics out of the objective through `Rune.value_and_grad_aux`'s auxiliary channel — they ride through differentiation undifferentiated:

```ocaml
module Net = struct
  type t = { l1 : Linear.t; bn : Batch_norm.t; l2 : Linear.t }

  let map (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t) { l1; bn; l2 } =
    { l1 = Linear.map f l1; bn = Batch_norm.map f bn; l2 = Linear.map f l2 }

  let map2 (f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t) p q =
    {
      l1 = Linear.map2 f p.l1 q.l1;
      bn = Batch_norm.map2 f p.bn q.bn;
      l2 = Linear.map2 f p.l2 q.l2;
    }

  let iter (f : 'a 'b. ('a, 'b) Nx.t -> unit) { l1; bn; l2 } =
    Linear.iter f l1;
    Batch_norm.iter f bn;
    Linear.iter f l2

  let forward p stats ~training x =
    let h = Linear.apply p.l1 x in
    let h, stats = Batch_norm.apply p.bn stats ~training h in
    (Linear.apply p.l2 (Fn.relu h), stats)
end

let () =
  Nx.Rng.run ~seed:0 @@ fun () ->
  let x = Nx.randn Nx.float32 [| 16; 4 |] in
  let y = Nx.randn Nx.float32 [| 16; 1 |] in
  let bn, stats = Batch_norm.init ~features:8 in
  let params =
    {
      Net.l1 = Linear.init ~inputs:4 ~outputs:8;
      bn;
      l2 = Linear.init ~inputs:8 ~outputs:1;
    }
  in

  (* One training step: stats' rides the auxiliary channel. *)
  let step (params, stats, ostate) =
    let objective p =
      let pred, stats' = Net.forward p stats ~training:true x in
      (Loss.mse pred y, stats')
    in
    let loss, grads, stats' =
      Rune.value_and_grad_aux (module Net) objective params
    in
    let params, ostate =
      Vega.adam_step (module Net) ~lr:1e-2 ostate ~params ~grads
    in
    ((params, stats', ostate), Nx.item [] loss)
  in

  let state = ref (params, stats, Vega.adam_init (module Net) params) in
  for _ = 1 to 20 do
    let s, _ = step !state in
    state := s
  done;

  (* Evaluation reuses the same forward with ~training:false and discards
     the returned statistics. *)
  let params, stats, _ = !state in
  let pred, _ = Net.forward params stats ~training:false x in
  Printf.printf "eval predictions: %s\n"
    (Nx.shape_to_string (Nx.shape pred))
```

The statistics update inside `apply` is detached, so no gradient flows through `stats'` — that is what makes the auxiliary channel safe. Statistics have their own traversals and `names` (`Batch_norm.Stats` is itself a `Ptree.S` structure), so they checkpoint like parameters under their own prefix; see [Checkpoints](04-checkpoints-and-pretrained/).

`Batch_norm` is single-precision only; the other layers are generic over float dtypes via their `make` constructors.

## Initializers

An initializer is a plain function from fan geometry, dtype, and shape to a fresh tensor — `Init.t`. Layers accept one as `?w_init`/`?bias_init` and supply the fans from their own geometry:

```ocaml
let () =
  Nx.Rng.run ~seed:0 @@ fun () ->
  (* He-normal weights for a ReLU network, no bias. *)
  let layer =
    Linear.make ~w_init:Init.he_normal ~bias:false ~inputs:64 ~outputs:64
      Nx.float32
  in
  ignore layer
```

The named families (Glorot/Xavier, He/Kaiming, LeCun) are instances of `Init.variance_scaling`; any function of the right type is an initializer, so custom schemes need no registration. Random initializers draw from the implicit RNG scope — wrap model construction in `Nx.Rng.run` for reproducibility.

## Next Steps

- [Training](03-training/) — the composable training step, data, and metrics
- [Checkpoints and Pretrained Models](04-checkpoints-and-pretrained/) — `names`, safetensors, the Hub
