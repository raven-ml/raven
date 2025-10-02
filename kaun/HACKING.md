# Kaun Developer Guide

## Architecture

Kaun is a Flax-inspired neural network library built on Rune's automatic differentiation. It provides functional, composable layers with explicit parameter management.

### Core Components

- **[lib/kaun/layer.ml](lib/kaun/layer.ml)**: Layer abstraction with init/apply pattern
- **[lib/kaun/ptree.ml](lib/kaun/ptree.ml)**: Parameter tree structure for nested parameters
- **[lib/kaun/optimizer/](lib/kaun/optimizer/)**: Optimizers (SGD, Adam, etc.)
- **[lib/kaun/loss.ml](lib/kaun/loss.ml)**: Loss functions (cross-entropy, MSE, etc.)
- **[lib/kaun/ops.ml](lib/kaun/ops.ml)**: High-level operations (conv, attention, etc.)
- **[lib/kaun/initializers.ml](lib/kaun/initializers.ml)**: Weight initialization strategies

### Key Design Principles

1. **Functional layer API**: Layers are pure functions with explicit init/apply
2. **Parameter trees**: Hierarchical parameter structure mirrors model architecture
3. **Training-aware**: Layers know if they're in training or inference mode
4. **Rune-native**: Built directly on Rune tensors and AD

## Layer Model

### The Module Pattern

Every layer is a record with `init` and `apply`:

```ocaml
type module_ = {
  init : rngs:Rune.Rng.key -> dtype:(float, 'layout) Rune.dtype -> 'layout Ptree.t;
  apply : 'layout Ptree.t -> training:bool -> ?rngs:Rune.Rng.key -> 'layout tensor -> 'layout tensor;
}
```

**Two-phase design:**
1. **Init**: Create parameter tree from RNG seed and dtype
2. **Apply**: Execute forward pass with parameters and input

### Parameter Trees

Parameters organized as trees:

```ocaml
type 'layout t =
  | Tensor of (float, 'layout) Rune.t
  | List of 'layout t list
  | Record of (string * 'layout t) list
```

**Why trees?**
- Natural nesting: `model.layers.0.weight`
- Composability: Combine layer parameters
- Serialization: Easy to save/load
- Gradient alignment: Structure matches gradient tree

### Training Mode

Layers behave differently during training vs inference:

```ocaml
let apply params ~training ?rngs x =
  if training then
    (* Apply dropout, update batch norm stats, etc. *)
    ...
  else
    (* Use saved batch norm stats, no dropout *)
    ...
```

**Key differences:**
- Dropout: Active in training, disabled in inference
- Batch norm: Update stats in training, use frozen stats in inference
- RNG: Required in training for stochastic layers

## Development Workflow

### Building and Testing

```bash
# Build kaun
dune build kaun/

# Run tests
dune build kaun/test/test_kaun.exe && _build/default/kaun/test/test_kaun.exe

# Run examples
dune exec kaun/example/mnist_mlp.exe
```

### Testing Layers

Test both initialization and forward pass:

```ocaml
let test_dense () =
  let layer = Layer.dense ~units:10 () in

  (* Test init *)
  let rng = Rune.Rng.key 42 in
  let params = layer.init ~rngs:rng ~dtype:Rune.Float32 in

  (* Verify parameter shapes *)
  match params with
  | Record fields ->
      let w = Ptree.Record.find "weight" fields in
      (* Check shape, initialization range, etc. *)
      ...

  (* Test apply *)
  let x = Rune.randn Float32 [|32; 20|] in
  let y = layer.apply params ~training:true x in
  (* Verify output shape *)
  Alcotest.(check (array int)) "output shape" [|32; 10|] (Rune.shape y)
```

### Gradient Checking

Verify autodiff correctness:

```ocaml
let test_layer_gradients () =
  let layer = Layer.conv2d ~in_channels:3 ~out_channels:16 () in
  let params = layer.init ~rngs:(Rune.Rng.key 0) ~dtype:Float32 in

  let forward p x =
    layer.apply p ~training:true x |> Rune.sum
  in

  let x = Rune.randn Float32 [|2; 3; 32; 32|] in
  Rune.gradcheck forward params x  (* Finite difference validation *)
```

## Adding Layers

### Simple Stateless Layer

Activation functions have no parameters:

```ocaml
let gelu () = {
  init = (fun ~rngs:_ ~dtype:_ -> List []);
  apply = (fun _params ~training:_ ?rngs:_ x -> Activations.gelu x);
}
```

### Parameterized Layer

Dense layer with weight and bias:

```ocaml
let dense ~units () = {
  init = (fun ~rngs ~dtype ->
    let fan_in = (* infer from first input *) in
    let w = Initializers.glorot_uniform rngs dtype [|fan_in; units|] in
    let b = Rune.zeros dtype [|units|] in
    Ptree.record_of [("weight", Tensor w); ("bias", Tensor b)]);

  apply = (fun params ~training:_ ?rngs:_ x ->
    match params with
    | Record fields ->
        let w = Ptree.Record.find "weight" fields |> unwrap_tensor in
        let b = Ptree.Record.find "bias" fields |> unwrap_tensor in
        Rune.add (Rune.dot x w) b
    | _ -> failwith "dense: invalid params");
}
```

### Stateful Layer (Batch Norm)

Batch norm tracks running mean/variance:

```ocaml
let batch_norm ~num_features ?(momentum=0.1) () = {
  init = (fun ~rngs:_ ~dtype ->
    let gamma = Rune.ones dtype [|num_features|] in
    let beta = Rune.zeros dtype [|num_features|] in
    let running_mean = Rune.zeros dtype [|num_features|] in
    let running_var = Rune.ones dtype [|num_features|] in
    Ptree.record_of [
      ("gamma", Tensor gamma);
      ("beta", Tensor beta);
      ("running_mean", Tensor running_mean);
      ("running_var", Tensor running_var);
    ]);

  apply = (fun params ~training ?rngs:_ x ->
    if training then
      (* Compute batch stats, update running stats *)
      let mean = Rune.mean x ~axes:[|0|] in
      let var = Rune.var x ~axes:[|0|] in
      (* Update running_mean, running_var with momentum *)
      ...
    else
      (* Use frozen running stats *)
      let mean = get_param params "running_mean" in
      let var = get_param params "running_var" in
      ...
    );
}
```

## Optimizers

### Optimizer Pattern

Optimizers maintain state and update parameters:

```ocaml
type 'layout optimizer_state
type 'layout optimizer = {
  init : 'layout Ptree.t -> 'layout optimizer_state;
  update : 'layout optimizer_state -> 'layout Ptree.t (* grads *) ->
           'layout Ptree.t (* params *) ->
           'layout Ptree.t * 'layout optimizer_state;
}
```

**Usage:**

```ocaml
let opt = Optimizer.adam ~lr:0.001 () in
let opt_state = opt.init params in

(* Training step *)
let loss, grads = Rune.value_and_grad loss_fn params x y in
let params, opt_state = opt.update opt_state grads params in
```

### Adding Optimizers

Implement init and update:

```ocaml
let sgd ~lr () = {
  init = (fun _params -> ());  (* SGD has no state *)

  update = (fun _state grads params ->
    let updated = Ptree.map2 (fun g p ->
      Rune.sub p (Rune.mul (Rune.scalar (Rune.dtype p) lr) g)
    ) grads params in
    (updated, ())
  );
}
```

Adam with momentum:

```ocaml
let adam ~lr ?(beta1=0.9) ?(beta2=0.999) () = {
  init = (fun params ->
    let m = Ptree.map (fun t -> Rune.zeros_like t) params in
    let v = Ptree.map (fun t -> Rune.zeros_like t) params in
    {m; v; step = 0});

  update = (fun state grads params ->
    let step = state.step + 1 in
    (* Compute biased first moment estimate *)
    let m = Ptree.map2 (fun m_t g_t ->
      Rune.add (Rune.mul (scalar beta1) m_t)
               (Rune.mul (scalar (1. -. beta1)) g_t)
    ) state.m grads in
    (* ... Adam update logic ... *)
    (updated_params, {m; v; step})
  );
}
```

## Common Patterns

### Sequential Composition

Chain layers:

```ocaml
let sequential layers = {
  init = (fun ~rngs ~dtype ->
    let params = List.map (fun layer ->
      layer.init ~rngs ~dtype
    ) layers in
    List params);

  apply = (fun params ~training ?rngs x ->
    List.fold_left2 (fun acc layer p ->
      layer.apply p ~training ?rngs acc
    ) x layers (match params with List ps -> ps | _ -> assert false));
}
```

### Residual Connections

```ocaml
let residual layer = {
  init = layer.init;
  apply = (fun params ~training ?rngs x ->
    let y = layer.apply params ~training ?rngs x in
    Rune.add x y);  (* x + F(x) *)
}
```

### Multi-Input Layers

```ocaml
let attention () = {
  init = ...;
  apply = (fun params ~training ?rngs (q, k, v) ->
    (* Attention over query, key, value *)
    ...);
}
```

## Common Pitfalls

### Shape Inference in Init

Cannot infer input shape during init:

```ocaml
(* Wrong: input shape unknown *)
let init ~rngs ~dtype =
  let fan_in = ??? in  (* No input yet! *)
  ...

(* Solution 1: Explicit in_features *)
let dense ~in_features ~out_features () = ...

(* Solution 2: Lazy init on first apply *)
let dense ~out_features () = {
  init = (fun ~rngs ~dtype -> Lazy);
  apply = (fun params ~training ?rngs x ->
    match params with
    | Lazy ->
        let in_features = Array.get (Rune.shape x) 1 in
        (* Initialize now *)
        ...
    | Record _ -> (* Use existing params *) ...
  );
}
```

### Training Mode Mismatch

Always pass `~training` correctly:

```ocaml
(* Wrong: always training mode *)
let y = model.apply params ~training:true x

(* Correct: match phase *)
let y_train = model.apply params ~training:true x_train in
let y_eval = model.apply params ~training:false x_test
```

### Parameter Tree Structure

Match structure when extracting parameters:

```ocaml
(* Wrong: assume flat structure *)
match params with
| Tensor t -> ...  (* Fails for nested params *)

(* Correct: match expected structure *)
match params with
| Record fields ->
    let w = Ptree.Record.find "weight" fields in
    ...
```

### RNG Threading

Stochastic layers need RNG:

```ocaml
(* Wrong: no RNG for dropout *)
let y = dropout_layer.apply params ~training:true x  (* Error! *)

(* Correct: pass RNG *)
let y = dropout_layer.apply params ~training:true ~rngs x
```

## Performance

- **Batch operations**: Use vectorized Rune ops, avoid loops
- **JIT compilation**: Use `Rune.jit` for hot paths
- **Parameter updates**: Update in-place when possible (future optimization)
- **Gradient checkpointing**: Trade compute for memory (future feature)

## Code Style

- **Layer constructors**: Use labeled arguments: `~units`, `~kernel_size`
- **Parameter names**: Standard naming: `weight`, `bias`, `gamma`, `beta`
- **Errors**: `"layer_name: error description"`
- **Documentation**: Document expected input/output shapes

## Related Documentation

- [CLAUDE.md](../CLAUDE.md): Project-wide conventions
- [README.md](README.md): User-facing documentation
- [rune/HACKING.md](../rune/HACKING.md): Autodiff and gradients
- Flax documentation for API inspiration
