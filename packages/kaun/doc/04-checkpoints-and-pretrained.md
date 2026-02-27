# Checkpoints and Pretrained Models

This guide covers saving and loading model parameters with SafeTensors,
and downloading pretrained weights from the HuggingFace Hub.

## SafeTensors Checkpointing

Kaun serializes parameter trees to the
[SafeTensors](https://huggingface.co/docs/safetensors/) format. Tensor
paths from the tree structure become file keys (e.g. `layers.0.weight`).

### Saving

<!-- $MDX skip -->
```ocaml
let vars = Train.vars st in
Checkpoint.save "model.safetensors" (Layer.params vars)
```

### Loading

`Checkpoint.load` requires a `~like` template that defines the expected
tree structure and dtypes. Tensors are cast to the template's dtype if
needed. Extra keys in the file are ignored.

<!-- $MDX skip -->
```ocaml
(* Initialize model to get the tree structure *)
let vars = Layer.init model ~dtype:Nx.Float32 in
let params = Checkpoint.load "model.safetensors" ~like:(Layer.params vars) in
let vars = Layer.with_params vars params
```

### Saving and Loading State

To save both parameters and non-trainable state (e.g. batch norm
running statistics):

<!-- $MDX skip -->
```ocaml
(* Save *)
let vars = Train.vars st in
Checkpoint.save "params.safetensors" (Layer.params vars);
Checkpoint.save "state.safetensors" (Layer.state vars)

(* Load *)
let vars = Layer.init model ~dtype:Nx.Float32 in
let params = Checkpoint.load "params.safetensors" ~like:(Layer.params vars) in
let state = Checkpoint.load "state.safetensors" ~like:(Layer.state vars) in
let vars = Layer.with_params vars params |> fun v -> Layer.with_state v state
```

### Resuming Training

Use `Train.make_state` to create training state from loaded weights:

<!-- $MDX skip -->
```ocaml
let trainer = Train.make ~model ~optimizer in
let st = Train.make_state trainer vars in
(* Continue training from here *)
let st = Train.fit trainer st data
```

## HuggingFace Hub

The `kaun-hf` package provides access to the HuggingFace Hub for
downloading pretrained model weights and configurations.

### Downloading Files

<!-- $MDX skip -->
```ocaml
let path = Kaun_hf.download_file ~model_id:"bert-base-uncased"
  ~filename:"config.json" ()
(* path : string — local filesystem path *)
```

Files are cached under `$RAVEN_CACHE_ROOT/huggingface` (or
`$XDG_CACHE_HOME/raven/huggingface`). Subsequent calls return the cached
path.

Options:

- `~token` — HuggingFace API token for private repositories. Defaults
  to the `HF_TOKEN` environment variable.
- `~cache_dir` — override the default cache directory.
- `~offline:true` — only return cached files, do not download.
- `~revision:(Rev "v1.0")` — download a specific tag, branch, or commit.
  Default is `Main`.

### Loading Configuration

<!-- $MDX skip -->
```ocaml
let config = Kaun_hf.load_config ~model_id:"bert-base-uncased" ()
(* config : Jsont.json *)
```

Returns the parsed `config.json` from the repository.

### Loading Weights

<!-- $MDX skip -->
```ocaml
let weights = Kaun_hf.load_weights ~model_id:"bert-base-uncased" ()
(* weights : (string * Kaun.Ptree.tensor) list *)
```

Returns a flat list of `(name, tensor)` pairs from the model's
SafeTensors checkpoint. Sharded checkpoints are handled transparently:
when `model.safetensors.index.json` is present, all shards are
downloaded and merged.

Tensor names are the raw keys from the SafeTensors file (e.g.
`bert.encoder.layer.0.attention.self.query.weight`). Your model code
maps these to its own parameter structure.

### Loading a Pretrained Model

The typical pattern for loading pretrained weights:

1. Build the model architecture from the config.
2. Initialize to get the parameter tree structure.
3. Load weights and map them to the tree.

<!-- $MDX skip -->
```ocaml
(* 1. Build model from config *)
let config = Kaun_hf.load_config ~model_id:"bert-base-uncased" () in
let model = build_bert_model config in

(* 2. Initialize to get tree structure *)
let vars = Layer.init model ~dtype:Nx.Float32 in

(* 3. Load and map weights *)
let weights = Kaun_hf.load_weights ~model_id:"bert-base-uncased" () in
let params = map_weights_to_ptree weights (Layer.params vars) in
let vars = Layer.with_params vars params in

(* 4. Use for inference *)
let trainer = Train.make ~model
  ~optimizer:(Optim.adam ~lr:(Optim.Schedule.constant 1e-5) ())
in
let st = Train.make_state trainer vars in
let logits = Train.predict trainer st input_ids
```

### Cache Management

<!-- $MDX skip -->
```ocaml
(* Clear all cached files *)
Kaun_hf.clear_cache ()

(* Clear a specific model's cache *)
Kaun_hf.clear_cache ~model_id:"bert-base-uncased" ()
```

## Next Steps

- [Getting Started](../01-getting-started/) — XOR and MNIST examples
- [Layers and Models](../02-layers-and-models/) — layer catalog, composition, custom layers
- [Training](../03-training/) — optimizers, losses, data pipelines, training loops
