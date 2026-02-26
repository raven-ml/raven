# Layers and Models

A `Layer.t` pairs parameter initialization with a forward computation.
This guide covers the built-in layers, composition, custom layers, and
the `vars` type.

## The Layer Type

A layer is a record with two fields:

<!-- $MDX skip -->
```ocaml
type ('input, 'output) Layer.t = {
  init :
    'layout.
    rngs:Rune.Rng.key -> dtype:(float, 'layout) Rune.dtype -> 'layout vars;
  apply :
    'layout 'in_elt.
    params:Ptree.t ->
    state:Ptree.t ->
    dtype:(float, 'layout) Rune.dtype ->
    training:bool ->
    ?rngs:Rune.Rng.key ->
    ?ctx:Context.t ->
    ('input, 'in_elt) Rune.t ->
    ('output, 'layout) Rune.t * Ptree.t;
}
```

The type parameters `'input` and `'output` describe the element types.
Most layers use `(float, float) Layer.t` — they accept and produce float
tensors. `embedding` is `(int32, float) Layer.t` — it accepts int32
indices and produces float vectors.

Use `Layer.init` and `Layer.apply` instead of accessing fields directly:

<!-- $MDX skip -->
```ocaml
let vars = Layer.init model ~rngs:(Rune.Rng.key 42) ~dtype:Rune.float32
let output, vars' = Layer.apply model vars ~training:false x
```

## The vars Type

`Layer.vars` bundles trainable parameters, non-trainable state, and a
dtype witness:

<!-- $MDX skip -->
```ocaml
Layer.params vars   (* Ptree.t — trainable parameters *)
Layer.state vars    (* Ptree.t — non-trainable state (e.g. batch norm stats) *)
Layer.dtype vars    (* dtype witness *)
```

Use `Layer.with_params` and `Layer.with_state` to replace components:

<!-- $MDX skip -->
```ocaml
let vars' = Layer.with_params vars new_params
```

## Composition

### sequential

`Layer.sequential` chains `(float, float) Layer.t` layers in order.
Parameters are stored as a `Ptree.List`:

<!-- $MDX skip -->
```ocaml
let model = Layer.sequential [
  Layer.linear ~in_features:784 ~out_features:128 ();
  Layer.relu ();
  Layer.linear ~in_features:128 ~out_features:10 ();
]
```

### compose

`Layer.compose` chains two layers with different input/output types.
Parameters are stored as a `Ptree.Dict` with keys `"left"` and
`"right"`:

<!-- $MDX skip -->
```ocaml
(* embedding (int32 -> float) composed with a linear layer (float -> float) *)
let embed_then_project =
  Layer.compose
    (Layer.embedding ~vocab_size:10000 ~embed_dim:256 ())
    (Layer.linear ~in_features:256 ~out_features:128 ())
(* embed_then_project : (int32, float) Layer.t *)
```

## Dense

<!-- $MDX skip -->
```ocaml
Layer.linear ~in_features:784 ~out_features:128 ()
```

Fully connected layer computing `xW + b`. Optional `~weight_init` and
`~bias_init` arguments override the defaults (Glorot uniform for
weights, zeros for bias).

## Convolution

<!-- $MDX skip -->
```ocaml
(* 1D: input [batch; in_channels; length] *)
Layer.conv1d ~in_channels:3 ~out_channels:16 ()
Layer.conv1d ~in_channels:3 ~out_channels:16 ~kernel_size:5 ~stride:2 ~padding:`Valid ()

(* 2D: input [batch; in_channels; height; width] *)
Layer.conv2d ~in_channels:1 ~out_channels:32 ()
Layer.conv2d ~in_channels:1 ~out_channels:32 ~kernel_size:(5, 5) ()
```

`conv1d` supports configurable `~kernel_size` (default 3), `~stride`
(default 1), `~dilation` (default 1), and `~padding` (default `` `Same ``).

`conv2d` supports configurable `~kernel_size` (default `(3, 3)`). Stride
is `(1, 1)` and padding is `` `Same ``.

## Normalization

<!-- $MDX skip -->
```ocaml
Layer.layer_norm ~dim:128 ()              (* learnable gamma and beta *)
Layer.layer_norm ~dim:128 ~eps:1e-6 ()

Layer.rms_norm ~dim:128 ()                (* learnable scale, no bias *)

Layer.batch_norm ~num_features:32 ()      (* learnable scale and bias,
                                             running mean/var in state *)
```

`batch_norm` updates running statistics during training and uses them
during evaluation. Normalization axes are inferred from rank: rank 2
uses `[0]`, rank 3 uses `[0; 2]`, rank 4 uses `[0; 2; 3]`.

## Embedding

<!-- $MDX skip -->
```ocaml
Layer.embedding ~vocab_size:10000 ~embed_dim:256 ()
```

Input: int32 token indices of any shape. Output: float tensors with
`embed_dim` appended to the input shape.

When `~scale:true` (the default), output vectors are multiplied by
`sqrt(embed_dim)`.

## Regularization

<!-- $MDX skip -->
```ocaml
Layer.dropout ~rate:0.1 ()
```

During training (`~training:true`), randomly zeros elements with
probability `rate`. Requires `~rngs` during training. Identity during
evaluation.

## Activations

All activation layers have no parameters:

<!-- $MDX skip -->
```ocaml
Layer.relu ()       (* max(0, x) *)
Layer.gelu ()       (* Gaussian error linear unit *)
Layer.silu ()       (* x * sigmoid(x) *)
Layer.tanh ()       (* hyperbolic tangent *)
Layer.sigmoid ()    (* logistic function *)
```

## Pooling

<!-- $MDX skip -->
```ocaml
Layer.max_pool2d ~kernel_size:(2, 2) ()
Layer.avg_pool2d ~kernel_size:(2, 2) ()
Layer.max_pool2d ~kernel_size:(2, 2) ~stride:(1, 1) ()
```

`~stride` defaults to `~kernel_size`. No parameters.

## Reshape

<!-- $MDX skip -->
```ocaml
Layer.flatten ()
```

Flattens all dimensions after the batch dimension:
`[batch; d1; ...; dn]` becomes `[batch; d1 * ... * dn]`.

## Multi-Head Attention

<!-- $MDX skip -->
```ocaml
Attention.multi_head_attention ~embed_dim:256 ~num_heads:8 ()
```

Input shape: `[batch; seq_len; embed_dim]`. Output shape:
`[batch; seq_len; embed_dim]`.

Options:

- `~num_kv_heads` — for grouped query attention (GQA). Default: same as
  `num_heads`.
- `~is_causal:true` — applies a causal mask to prevent attending to
  future positions.
- `~rope:true` — applies rotary position embeddings to Q and K.
  `~rope_theta` sets the base frequency (default 10000.0).
- `~dropout` — attention dropout rate. Requires `~rngs` during training.

Pass an attention mask via `Context`:

<!-- $MDX skip -->
```ocaml
let ctx =
  Context.empty
  |> Context.set ~name:Attention.attention_mask_key (Ptree.P mask)
in
Layer.apply model vars ~training:false ~ctx input
```

The mask is a bool or int32 tensor of shape `[batch; seq_k]`. Nonzero
positions are kept, zero positions are masked.

RoPE is also available as a standalone function:

<!-- $MDX skip -->
```ocaml
let x' = Attention.rope x              (* default theta=10000, seq_dim=-2 *)
let x' = Attention.rope ~theta:500000. ~seq_dim:1 x
```

## Custom Layers

A custom layer is a `{ init; apply }` record. Here is a residual block:

<!-- $MDX skip -->
```ocaml
let residual_block ~dim () : (float, float) Layer.t =
  let inner = Layer.sequential [
    Layer.linear ~in_features:dim ~out_features:dim ();
    Layer.relu ();
    Layer.linear ~in_features:dim ~out_features:dim ();
  ] in
  {
    init = inner.init;
    apply = (fun ~params ~state ~dtype ~training ?rngs ?ctx x ->
      let y, state' = inner.apply ~params ~state ~dtype ~training ?rngs ?ctx x in
      (Rune.add x y, state'));
  }
```

Use `Layer.make_vars` to build vars in custom `init` functions:

<!-- $MDX skip -->
```ocaml
Layer.make_vars ~params ~state:Ptree.empty ~dtype
```

## Context

`Context.t` carries per-call auxiliary data that specific layers read
during the forward pass. Most layers ignore it.

<!-- $MDX skip -->
```ocaml
let ctx =
  Context.empty
  |> Context.set ~name:"attention_mask" (Ptree.P mask)
  |> Context.set ~name:"token_type_ids" (Ptree.P ids)
in
Layer.apply model vars ~training:false ~ctx input_ids
```

Context is forwarded through `compose` and `sequential` to all sublayers.
`Train.fit`, `Train.step`, and `Train.predict` accept an optional `~ctx`
argument.

## Weight Initialization

Override default initialization with `Init.t` values:

<!-- $MDX skip -->
```ocaml
Layer.linear ~in_features:128 ~out_features:64
  ~weight_init:(Init.he_normal ())
  ~bias_init:Init.zeros
  ()
```

Available initializers:

- `Init.zeros`, `Init.ones`, `Init.constant v`
- `Init.uniform ~scale ()`, `Init.normal ~stddev ()`
- `Init.glorot_uniform ()`, `Init.glorot_normal ()`
- `Init.he_uniform ()`, `Init.he_normal ()`
- `Init.lecun_uniform ()`, `Init.lecun_normal ()`
- `Init.variance_scaling ~scale ~mode ~distribution ()`

## Next Steps

- [Training](../03-training/) — optimizers, losses, data pipelines, training loops
- [Checkpoints and Pretrained Models](../04-checkpoints-and-pretrained/) — saving, loading, HuggingFace Hub
