# Kaun vs. PyTorch / Flax -- A Practical Comparison

This guide explains how Kaun relates to PyTorch and Flax, focusing on:

* How core concepts map (modules/layers, parameters, training loops)
* Where the APIs feel similar vs. deliberately different
* How to translate common patterns between frameworks

Kaun's design is closer to Flax than to PyTorch: layers are pure data,
parameters are explicit trees, and forward passes are functions rather
than method calls. If you know Flax, Kaun will feel familiar. If you
know only PyTorch, the main shift is from mutable objects to immutable
records.

---

## 1. Big-Picture Differences

| Aspect            | PyTorch                                        | Flax (Linen)                       | Kaun (OCaml)                                          |
| ----------------- | ---------------------------------------------- | ---------------------------------- | ----------------------------------------------------- |
| Language          | Python, dynamic                                | Python (JAX), dynamic              | OCaml, statically typed                               |
| Model definition  | `nn.Module` class with `forward`               | `nn.Module` class with `__call__`  | `Layer.t` record with `init` and `apply`              |
| Parameter storage | Mutable attributes on module                   | Frozen dict returned by `init`     | `Ptree.t` tree returned by `Layer.init`               |
| Forward pass      | `model(x)` (stateful method)                   | `model.apply(params, x)`           | `Layer.apply model vars ~training x`                  |
| Mutation          | Modules are mutable objects                    | Params are immutable dicts         | `Layer.vars` and `Ptree.t` are immutable              |
| Autograd          | Dynamic tape (`autograd`)                      | Functional transforms (`jax.grad`) | Rune effect-based autodiff                            |
| Optimizer         | `torch.optim.Adam(model.parameters(), lr=...)` | `optax.adam(lr)`                   | `Optim.adam ~lr:(Schedule.constant lr) ()`            |
| Training loop     | Manual (or Lightning/etc.)                     | Manual (or Orbax/etc.)             | `Train.fit` or manual `Train.step`                    |
| Data loading      | `DataLoader`                                   | `tf.data` or manual                | `Data.t` lazy pipeline                                |
| Checkpointing     | `torch.save` / `torch.load` (pickle)           | Orbax / msgpack                    | SafeTensors via `Checkpoint.save` / `Checkpoint.load` |
| RNG               | Global `torch.manual_seed`                     | Explicit PRNGKey threading         | Implicit scope via `Nx.Rng.run ~seed`                 |
| Device management | `model.to("cuda")`, `tensor.cuda()`            | `jax.device_put`                   | CPU by default; JIT manages devices internally        |

---

## 2. Defining Models

### PyTorch

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = MLP(784, 128, 10)
```

### Flax

```python
import flax.linen as nn
import jax

class MLP(nn.Module):
    hidden: int
    out_features: int

    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(self.hidden)(x))
        return nn.Dense(self.out_features)(x)

model = MLP(hidden=128, out_features=10)
params = model.init(jax.random.PRNGKey(0), jnp.ones([1, 784]))
```

### Kaun

<!-- $MDX skip -->
```ocaml
open Kaun

let model =
  Layer.sequential
    [
      Layer.linear ~in_features:784 ~out_features:128 ();
      Layer.relu ();
      Layer.linear ~in_features:128 ~out_features:10 ();
    ]

let vars =
  Nx.Rng.run ~seed:0 @@ fun () ->
  Layer.init model ~dtype:Nx.Float32
```

Key differences:

* PyTorch defines models as classes. Flax defines models as dataclasses
  with `__call__`. Kaun uses `Layer.t` records -- plain data, not classes.
* `Layer.sequential` replaces class-based composition for homogeneous
  float pipelines. `Layer.compose` handles heterogeneous types (e.g.
  embedding into dense).
* Activation functions are layers (`Layer.relu ()`) rather than free
  functions called inside `forward`. This keeps the composition uniform.

---

## 3. Parameters

### PyTorch

```python
# Parameters live inside the module
for name, param in model.named_parameters():
    print(name, param.shape)

# state_dict is an OrderedDict
sd = model.state_dict()
model.load_state_dict(sd)
```

### Flax

```python
# Params are a frozen dict returned by init
params = model.init(key, x)["params"]
jax.tree_util.tree_map(lambda p: p.shape, params)
```

### Kaun

<!-- $MDX skip -->
```ocaml
(* vars bundles params, state, and dtype *)
let params = Layer.params vars   (* Ptree.t *)
let state  = Layer.state vars    (* Ptree.t *)
let dt     = Layer.dtype vars    (* (float, 'layout) Nx.dtype *)

(* Inspect parameter shapes *)
let paths = Ptree.flatten_with_paths params
(* [("0.weight", P tensor); ("0.bias", P tensor); ...] *)

(* Count total parameters *)
let n = Ptree.count_parameters params

(* Replace parameters *)
let vars' = Layer.with_params vars new_params
```

Key differences:

* PyTorch stores parameters as mutable module attributes. Flax returns
  frozen dicts. Kaun returns `Ptree.t` -- a tree with `Tensor` leaves,
  `Dict` nodes, and `List` nodes.
* `Ptree.t` is plain immutable data. You can map, fold, flatten, and
  serialize it without going through the model.
* `Layer.vars` also carries non-trainable state (e.g. batch norm
  running statistics), separate from trainable parameters.

---

## 4. Forward Pass

### PyTorch

```python
model.train()
output = model(x)          # stateful: dropout active, batchnorm updates

model.eval()
with torch.no_grad():
    output = model(x)      # no dropout, batchnorm uses running stats
```

### Flax

```python
output = model.apply(params, x)
output = model.apply(params, x, train=True, rngs={"dropout": key})
```

### Kaun

<!-- $MDX skip -->
```ocaml
(* Training: dropout active, batchnorm updates running stats *)
let output, vars' = Layer.apply model vars ~training:true x

(* Evaluation: no dropout, batchnorm uses running stats *)
let output, vars' = Layer.apply model vars ~training:false x

(* Or through the trainer *)
let logits = Train.predict trainer st x
```

Key differences:

* PyTorch uses `model.train()` / `model.eval()` to switch mode globally.
  Kaun passes `~training` as an argument on each call.
* `Layer.apply` returns `(output, updated_vars)`. The updated vars carry
  new state (e.g. batch norm statistics). Parameters are unchanged.
* `Train.predict` is a shortcut for evaluation mode with no state updates.

---

## 5. Optimizers and LR Schedules

### PyTorch

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

optimizer.zero_grad()
loss.backward()
optimizer.step()
scheduler.step()
```

### Flax / Optax

```python
import optax

tx = optax.adam(learning_rate=optax.cosine_decay_schedule(1e-3, 10000))
opt_state = tx.init(params)
updates, opt_state = tx.update(grads, opt_state, params)
params = optax.apply_updates(params, updates)
```

### Kaun

<!-- $MDX skip -->
```ocaml
(* Schedule is a function: step -> lr *)
let schedule = Optim.Schedule.cosine_decay ~init_value:1e-3 ~decay_steps:10000 ()

(* Optimizer takes the schedule directly *)
let optimizer = Optim.adam ~lr:schedule ()

(* Manual update *)
let st = Optim.init optimizer params in
let updates, st' = Optim.step optimizer st params grads in
let params' = Optim.apply_updates params updates

(* Or use the convenience function *)
let params', st' = Optim.update optimizer st params grads
```

Available optimizers:

<!-- $MDX skip -->
```ocaml
Optim.sgd ~lr:schedule ~momentum:0.9 ~nesterov:true ()
Optim.adam ~lr:schedule ~b1:0.9 ~b2:0.999 ~eps:1e-8 ()
Optim.adamw ~lr:schedule ~weight_decay:0.01 ()
Optim.rmsprop ~lr:schedule ~decay:0.9 ~momentum:0.0 ()
Optim.adagrad ~lr:schedule ~eps:1e-8 ()
```

Available schedules:

<!-- $MDX skip -->
```ocaml
Optim.Schedule.constant 1e-3
Optim.Schedule.cosine_decay ~init_value:1e-3 ~decay_steps:10000 ()
Optim.Schedule.warmup_cosine ~init_value:0. ~peak_value:1e-3 ~warmup_steps:1000
Optim.Schedule.warmup_linear ~init_value:0. ~peak_value:1e-3 ~warmup_steps:1000
Optim.Schedule.exponential_decay ~init_value:1e-3 ~decay_rate:0.96 ~decay_steps:1000
```

Key differences:

* PyTorch couples the optimizer to the model via `model.parameters()`.
  Kaun and Optax are decoupled -- they operate on parameter trees.
* PyTorch separates scheduler from optimizer. Kaun (like Optax) bakes
  the schedule into the optimizer via the `~lr` argument.
* A Kaun schedule is just `int -> float`. Compose them by writing a
  plain OCaml function.

---

## 6. Loss Functions

### PyTorch

```python
loss = nn.functional.cross_entropy(logits, labels)
loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
loss = nn.functional.mse_loss(pred, target)
```

### Kaun

<!-- $MDX skip -->
```ocaml
(* Multi-class with one-hot labels *)
Loss.cross_entropy logits one_hot_labels

(* Multi-class with integer labels *)
Loss.cross_entropy_sparse logits class_indices

(* Binary classification (raw logits, not sigmoid) *)
Loss.binary_cross_entropy logits labels

(* Regression *)
Loss.mse predictions targets
Loss.mae predictions targets
```

Key differences:

* PyTorch's `cross_entropy` expects integer labels (like
  `cross_entropy_sparse`). Kaun offers both one-hot and integer
  variants.
* All Kaun losses return scalar means and are differentiable through
  Rune's autodiff.
* Kaun losses are plain functions, not module methods. There is no
  `nn.CrossEntropyLoss()` class.

---

## 7. Training Loops

### PyTorch (manual loop)

```python
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = nn.functional.cross_entropy(logits, y_batch)
        loss.backward()
        optimizer.step()
        print(f"loss: {loss.item():.4f}")
```

### Kaun with Train.fit

<!-- $MDX skip -->
```ocaml
let trainer =
  Train.make ~model
    ~optimizer:(Optim.adam ~lr:(Optim.Schedule.constant 1e-3) ())

let st = Nx.Rng.run ~seed:42 @@ fun () ->
  Train.init trainer ~dtype:Nx.Float32

(* Train over a data pipeline *)
let st =
  Train.fit trainer st
    ~report:(fun ~step ~loss _st ->
      Printf.printf "step %d  loss %.4f\n" step loss)
    data
```

`Train.fit` takes a `Data.t` where each element is `(input, loss_fn)`.
The loss function receives the model output and returns a scalar loss.
Gradient computation, optimizer step, and state threading are handled
internally.

### Kaun with Train.step (manual loop)

For fine-grained control, use `Train.step` directly:

<!-- $MDX skip -->
```ocaml
let st = ref (Nx.Rng.run ~seed:42 @@ fun () ->
  Train.init trainer ~dtype:Nx.Float32)

let () =
  Data.iter
    (fun (x, y) ->
      let loss, st' =
        Train.step trainer !st ~training:true
          ~loss:(fun logits -> Loss.cross_entropy_sparse logits y)
          x
      in
      st := st';
      Printf.printf "loss: %.4f\n" (Nx.item [] loss))
    data
```

### Early stopping

Raise `Train.Early_stop` inside the `~report` callback:

<!-- $MDX skip -->
```ocaml
let st =
  Train.fit trainer st
    ~report:(fun ~step:_ ~loss _st ->
      if loss < 0.001 then raise Train.Early_stop)
    data
```

Key differences:

* PyTorch training loops are fully manual: zero gradients, forward,
  backward, step. Kaun's `Train.fit` handles the entire loop.
* `Train.step` is the escape hatch for custom loops, but you never call
  `backward` or `zero_grad` -- differentiation is implicit.
* State threading replaces mutation. `Train.fit` returns the final
  state; `Train.step` returns `(loss, new_state)`.

---

## 8. Data Loading

### PyTorch

```python
from torch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(x_train, y_train)
loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)

for x_batch, y_batch in loader:
    ...
```

### Kaun

<!-- $MDX skip -->
```ocaml
(* From tensor pairs -- the common case *)
let data =
  Data.prepare ~shuffle:true ~batch_size:64 (x_train, y_train)
  |> Data.map (fun (x, y) ->
         (x, fun logits -> Loss.cross_entropy_sparse logits y))

(* From arrays *)
let data = Data.of_array examples |> Data.shuffle |> Data.batch 32

(* From a generator function *)
let data = Data.of_fn 10000 generate_example

(* Repeat a fixed example (useful for toy problems) *)
let data = Data.repeat 1000 (x, loss_fn)

(* Consumers *)
Data.iter process data
Data.fold accumulate init data
let arr = Data.to_array data
```

Key differences:

* PyTorch uses `Dataset` + `DataLoader` classes with worker processes
  for parallel loading. Kaun uses `Data.t`, a lazy composable iterator.
* `Data.prepare` is the standard shortcut: it slices tensors, optionally
  shuffles, and batches in one call. `~drop_last` defaults to `true`.
* Pipelines are single-pass. Call `Data.reset` before iterating again
  (e.g. between epochs).
* `Data.map` attaches the loss function to each batch, producing the
  `(input, loss_fn)` pairs that `Train.fit` expects.

---

## 9. Checkpointing

### PyTorch

```python
# Save
torch.save(model.state_dict(), "model.pt")

# Load
model.load_state_dict(torch.load("model.pt"))
```

### Kaun

<!-- $MDX skip -->
```ocaml
(* Save parameters *)
let vars = Train.vars st in
Checkpoint.save "model.safetensors" (Layer.params vars)

(* Load parameters *)
let vars = Layer.init model ~dtype:Nx.Float32 in
let params = Checkpoint.load "model.safetensors" ~like:(Layer.params vars) in
let vars = Layer.with_params vars params

(* Save both params and state (e.g. batch norm stats) *)
Checkpoint.save "params.safetensors" (Layer.params vars);
Checkpoint.save "state.safetensors" (Layer.state vars)

(* Resume training from loaded weights *)
let st = Train.make_state trainer vars
```

Key differences:

* PyTorch uses Python pickle by default (arbitrary code execution risk).
  Kaun uses SafeTensors -- a flat, memory-mappable format with no code
  execution.
* `Checkpoint.load` requires a `~like` template defining the expected
  tree structure and dtypes. Extra keys in the file are ignored, and
  tensors are cast to the template's dtype if needed.
* Pretrained weights from HuggingFace Hub are available via
  `Kaun_hf.load_weights`.

---

## 10. Quick Cheat Sheet

| Task                         | PyTorch                                            | Kaun                                                           |
| ---------------------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| Define a model               | `class M(nn.Module): ...`                          | `Layer.sequential [Layer.linear ...; Layer.relu (); ...]`      |
| Initialize parameters        | `model = M()` (implicit)                           | `Layer.init model ~dtype:Nx.Float32`                           |
| Forward pass (training)      | `model.train(); y = model(x)`                      | `Layer.apply model vars ~training:true x`                      |
| Forward pass (eval)          | `model.eval(); y = model(x)`                       | `Train.predict trainer st x`                                   |
| Count parameters             | `sum(p.numel() for p in model.parameters())`       | `Ptree.count_parameters (Layer.params vars)`                   |
| Create optimizer             | `Adam(model.parameters(), lr=1e-3)`                | `Optim.adam ~lr:(Optim.Schedule.constant 1e-3) ()`             |
| Cosine decay schedule        | `CosineAnnealingLR(opt, T_max=N)`                  | `Optim.Schedule.cosine_decay ~init_value:lr ~decay_steps:N ()` |
| Compute loss                 | `F.cross_entropy(logits, labels)`                  | `Loss.cross_entropy_sparse logits labels`                      |
| Training step                | `zero_grad(); loss.backward(); opt.step()`         | `Train.step trainer st ~training:true ~loss x`                 |
| Full training loop           | Manual `for` loop                                  | `Train.fit trainer st data`                                    |
| Early stopping               | Manual condition check                             | `raise Train.Early_stop` inside `~report`                      |
| Gradient clipping            | `clip_grad_norm_(model.parameters(), max_norm)`    | `Optim.clip_by_global_norm max_norm grads`                     |
| Data loading                 | `DataLoader(dataset, batch_size=64, shuffle=True)` | `Data.prepare ~shuffle:true ~batch_size:64 (x, y)`             |
| Save checkpoint              | `torch.save(model.state_dict(), path)`             | `Checkpoint.save path (Layer.params vars)`                     |
| Load checkpoint              | `model.load_state_dict(torch.load(path))`          | `Checkpoint.load path ~like:(Layer.params vars)`               |
| Compose heterogeneous layers | Define inside `forward`                            | `Layer.compose embedding_layer dense_layer`                    |
| Dropout                      | `nn.Dropout(p=0.1)`                                | `Layer.dropout ~rate:0.1 ()`                                   |
| Batch normalization          | `nn.BatchNorm2d(32)`                               | `Layer.batch_norm ~num_features:32 ()`                         |
| Layer normalization          | `nn.LayerNorm(128)`                                | `Layer.layer_norm ~dim:128 ()`                                 |
| Set RNG seed                 | `torch.manual_seed(42)`                            | `Nx.Rng.run ~seed:42 @@ fun () -> ...`                         |
