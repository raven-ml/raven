# Training

This guide covers optimizers, learning-rate schedules, loss functions,
data pipelines, the high-level training loop, metrics, and custom
training.

## Optimizers

An `Optim.algorithm` pairs a learning-rate schedule with an update rule.
All optimizers take `~lr` as a `Schedule.t`:

<!-- $MDX skip -->
```ocaml
(* SGD with momentum *)
Optim.sgd ~lr:(Optim.Schedule.constant 0.1) ~momentum:0.9 ()

(* Adam *)
Optim.adam ~lr:(Optim.Schedule.constant 1e-3) ()

(* AdamW with weight decay *)
Optim.adamw ~lr:(Optim.Schedule.constant 1e-3) ~weight_decay:0.01 ()

(* RMSprop *)
Optim.rmsprop ~lr:(Optim.Schedule.constant 1e-3) ()

(* Adagrad *)
Optim.adagrad ~lr:(Optim.Schedule.constant 0.01) ()
```

`sgd` supports optional `~momentum` (default 0.0) and `~nesterov`
(default false). `adam` and `adamw` support `~b1` (default 0.9), `~b2`
(default 0.999), and `~eps` (default 1e-8). `rmsprop` supports `~decay`
(default 0.9), `~eps`, and `~momentum`.

## Learning-Rate Schedules

A schedule is a function `int -> float` mapping 1-based step numbers to
learning rates:

<!-- $MDX skip -->
```ocaml
(* Fixed learning rate *)
Optim.Schedule.constant 1e-3

(* Cosine decay from 1e-3 to 0 over 10000 steps *)
Optim.Schedule.cosine_decay ~init_value:1e-3 ~decay_steps:10000 ()

(* Cosine decay with minimum alpha *)
Optim.Schedule.cosine_decay ~init_value:1e-3 ~decay_steps:10000 ~alpha:1e-5 ()

(* Linear warmup from 0 to 1e-3 over 1000 steps *)
Optim.Schedule.warmup_linear ~init_value:0. ~peak_value:1e-3 ~warmup_steps:1000

(* Cosine warmup *)
Optim.Schedule.warmup_cosine ~init_value:0. ~peak_value:1e-3 ~warmup_steps:1000

(* Exponential decay *)
Optim.Schedule.exponential_decay ~init_value:1e-3 ~decay_rate:0.96 ~decay_steps:1000
```

Compose schedules by writing a custom function:

<!-- $MDX skip -->
```ocaml
let warmup_then_cosine step =
  if step <= 1000 then
    Optim.Schedule.warmup_linear ~init_value:0. ~peak_value:1e-3 ~warmup_steps:1000 step
  else
    Optim.Schedule.cosine_decay ~init_value:1e-3 ~decay_steps:9000 () (step - 1000)
```

## Loss Functions

All loss functions return scalar tensors that are differentiable through
Rune's autodiff:

<!-- $MDX skip -->
```ocaml
(* Multi-class: logits [batch; num_classes], one-hot labels [batch; num_classes] *)
Loss.cross_entropy logits one_hot_labels

(* Multi-class with integer labels: logits [batch; num_classes], labels [batch] *)
Loss.cross_entropy_sparse logits class_indices

(* Binary: raw logits (not sigmoid), labels in {0, 1} *)
Loss.binary_cross_entropy logits labels

(* Regression *)
Loss.mse predictions targets
Loss.mae predictions targets
```

## Data Pipelines

`Data.t` is a lazy, composable iterator. Build pipelines by chaining
constructors, transformers, and consumers.

### Constructors

<!-- $MDX skip -->
```ocaml
(* From arrays *)
Data.of_array [| example1; example2; example3 |]

(* From tensors: slices along first dimension *)
Data.of_tensor x         (* yields x[0], x[1], ... *)
Data.of_tensors (x, y)   (* yields (x[0], y[0]), (x[1], y[1]), ... *)

(* From a function *)
Data.of_fn 1000 (fun i -> generate_example i)

(* Repeat a value *)
Data.repeat 1000 (x, loss_fn)
```

### Transformers

<!-- $MDX skip -->
```ocaml
(* Map each element *)
Data.map (fun (x, y) -> (preprocess x, y)) data

(* Batch into arrays of size n *)
Data.batch 32 data               (* yields arrays of 32 elements *)
Data.batch ~drop_last:true 32 data

(* Batch and map in one step *)
Data.map_batch 32 collate_fn data

(* Shuffle *)
Data.shuffle rng_key data
```

### Consumers

<!-- $MDX skip -->
```ocaml
Data.iter (fun x -> process x) data
Data.iteri (fun i x -> Printf.printf "%d: %f\n" i x) data
Data.fold (fun acc x -> acc +. x) 0. data
Data.to_array data
Data.to_seq data
```

### The prepare Shortcut

`Data.prepare` combines tensor slicing, optional shuffle, and batching
into one call. It is the standard way to feed tensor data to training:

<!-- $MDX skip -->
```ocaml
let train_data =
  Data.prepare ~shuffle:rng_key ~batch_size:64 (x_train, y_train)
  |> Data.map (fun (x, y) ->
         (x, fun logits -> Loss.cross_entropy_sparse logits y))
```

`Data.prepare` yields `(x_batch, y_batch)` tensor pairs. The `Data.map`
step attaches the loss function, producing the `(input, loss_fn)` pairs
that `Train.fit` expects.

`~drop_last` defaults to `true` in `prepare`.

### Resetting

Pipelines are single-pass. Call `Data.reset` to iterate again:

<!-- $MDX skip -->
```ocaml
Data.reset test_batches;
let acc = Metric.eval eval_fn test_batches
```

## High-Level Training

### Train.make and Train.init

Create a trainer by pairing a model with an optimizer, then initialize:

<!-- $MDX skip -->
```ocaml
let trainer = Train.make ~model
  ~optimizer:(Optim.adam ~lr:(Optim.Schedule.constant 1e-3) ())

let st = Train.init trainer ~rngs:(Rune.Rng.key 42) ~dtype:Rune.float32
```

### Train.fit

`Train.fit` trains over a data pipeline and returns the final state:

<!-- $MDX skip -->
```ocaml
let st = Train.fit trainer st ~rngs
  ~report:(fun ~step ~loss _st ->
    Printf.printf "step %d  loss %.4f\n" step loss)
  data
```

Each element of `data` is `(input, loss_fn)` where `loss_fn` takes the
model output and returns a scalar loss.

The optional `~report` callback is called after every step. The `~step`
number is 1-based.

### Early Stopping

Raise `Train.Early_stop` inside `~report` to end training early.
`Train.fit` catches it and returns the current state:

<!-- $MDX skip -->
```ocaml
let st = Train.fit trainer st ~rngs
  ~report:(fun ~step:_ ~loss st ->
    if loss < 0.001 then raise Train.Early_stop)
  data
```

### Train.predict

Run inference in evaluation mode (no dropout, no state updates):

<!-- $MDX skip -->
```ocaml
let logits = Train.predict trainer st x
```

### Train.step

For manual control over single training steps:

<!-- $MDX skip -->
```ocaml
let loss, st' = Train.step trainer st ~training:true ~rngs
  ~loss:(fun logits -> Loss.cross_entropy_sparse logits y)
  x
```

### Starting from Pretrained Weights

Use `Train.make_state` to create training state from externally loaded
weights instead of random initialization:

<!-- $MDX skip -->
```ocaml
let vars = (* load from checkpoint *) in
let st = Train.make_state trainer vars
```

## Metrics

### Metric Functions

Metric functions are plain `predictions -> targets -> float` functions:

<!-- $MDX skip -->
```ocaml
(* Multi-class: logits [batch; num_classes], labels [batch] *)
Metric.accuracy logits targets

(* Binary classification *)
Metric.binary_accuracy ~threshold:0.5 predictions targets

(* Precision, recall, F1 with averaging mode *)
Metric.precision Metric.Macro logits targets
Metric.recall Metric.Micro logits targets
Metric.f1 Metric.Weighted logits targets
```

Averaging modes: `Macro` (unweighted mean of per-class scores), `Micro`
(global aggregation), `Weighted` (mean weighted by class support).

### Dataset Evaluation

`Metric.eval` folds a function over a data pipeline and returns the
mean:

<!-- $MDX skip -->
```ocaml
Data.reset test_batches;
let test_acc =
  Metric.eval
    (fun (x, y) ->
      let logits = Train.predict trainer st x in
      Metric.accuracy logits y)
    test_batches
```

`Metric.eval_many` evaluates multiple named metrics at once:

<!-- $MDX skip -->
```ocaml
let results =
  Metric.eval_many
    (fun (x, y) ->
      let logits = Train.predict trainer st x in
      [ ("accuracy", Metric.accuracy logits y);
        ("f1", Metric.f1 Metric.Macro logits y) ])
    test_batches
(* results : (string * float) list *)
```

### Running Tracker

`Metric.tracker` accumulates running means during training:

<!-- $MDX skip -->
```ocaml
let tracker = Metric.tracker () in
(* In the training loop: *)
Metric.observe tracker "loss" loss_value;
Metric.observe tracker "accuracy" acc_value;

(* After an epoch: *)
Printf.printf "%s\n" (Metric.summary tracker);
(* "accuracy: 0.9150  loss: 0.4231" *)

Metric.reset tracker
```

## Gradient Utilities

### Gradient Clipping

Clip gradients by global L2 norm to prevent exploding gradients. Use
this with `Train.step` in custom training loops:

<!-- $MDX skip -->
```ocaml
let clipped_grads = Optim.clip_by_global_norm 1.0 grads
```

### Global Norm

Compute the L2 norm across all leaf tensors:

<!-- $MDX skip -->
```ocaml
let norm = Optim.global_norm grads
```

### Manual Gradient Computation

`Grad.value_and_grad` differentiates a function with respect to a
`Ptree.t`:

<!-- $MDX skip -->
```ocaml
let loss, grads = Grad.value_and_grad
  (fun params ->
    let output, _state = model.apply ~params ~state ~dtype ~training:true x in
    Loss.mse output y)
  params
```

`Grad.value_and_grad_aux` returns auxiliary data alongside the loss:

<!-- $MDX skip -->
```ocaml
let loss, grads, new_state = Grad.value_and_grad_aux
  (fun params ->
    let output, new_state = model.apply ~params ~state ~dtype ~training:true x in
    (Loss.mse output y, new_state))
  params
```

## Next Steps

- [Layers and Models](../02-layers-and-models/) — full layer catalog, composition, custom layers
- [Checkpoints and Pretrained Models](../04-checkpoints-and-pretrained/) — saving, loading, HuggingFace Hub
