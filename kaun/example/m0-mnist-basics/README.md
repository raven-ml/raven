# `m0-mnist-basics`

Train MNIST end-to-end with an explicit Kaun loop. This example performs every
step manually—dataset shuffling, batching, forward/backward passes, optimizer
updates, and metric accumulation—so you can see what `Training.fit` automates
later on. It uses a small convolutional network so you can observe tensor shapes
changing through conv/pool/flatten stages.

- **Source**: [`mnist.ml`](./mnist.ml)

MNIST images are fetched automatically into Kaun’s cache on the first run.

```bash
cd kaun/example/m0-mnist-basics
dune exec kaun/example/m0-mnist-basics/mnist.exe
```

### Model backbone

```ocaml
Layer.sequential
  [
    Layer.conv2d ~in_channels:1 ~out_channels:8 ();
    Layer.relu ();
    Layer.avg_pool2d ~kernel_size:(2, 2) ();
    Layer.conv2d ~in_channels:8 ~out_channels:16 ();
    Layer.relu ();
    Layer.avg_pool2d ~kernel_size:(2, 2) ();
    Layer.flatten ();
    Layer.linear ~in_features:784 ~out_features:128 ();
    Layer.relu ();
    Layer.linear ~in_features:128 ~out_features:10 ();
  ]
```

### Loop internals

- Uses `Kaun_datasets.mnist` to pull `(image, label)` tensors, then performs
  shuffling, batching, and stacking explicitly.
- Calls `value_and_grad` on the loss closure for each mini-batch and updates
  parameters with `Optimizer.step`/`Optimizer.apply_updates_inplace`.
- Tracks loss/accuracy through `Metrics.Collection` so you can log intermediate
  values and final epoch summaries.

### Sample console output

Manual driver (`mnist.exe`):

```
Creating datasets...
Train dataset created in 1.82s
Initializing model...

Epoch 1/10
  Batch 1: 0.142s (fwd+bwd: 0.091s, opt: 0.012s, metric: 0.018s) - Loss: 2.2991
  ...
  loss: 0.0973
  accuracy: 0.9714
  Epoch time: 5.62s
  Test: loss=0.0856 accuracy=0.9748 (eval time: 1.09s)
```

### Experiments to try

- Adjust `batch_size` and inspect the effect on throughput versus accuracy.
- Swap `Optimizer.adam` for `Optimizer.nadam` or `Optimizer.sgd` to contrast
  convergence behaviour.
- Adjust the convolution widths (e.g. increase `out_channels`) to trade accuracy
  for runtime.

### Next steps

- If you want to revisit the training primitives, see
  [**`02-xor-eval`**](../02-xor-eval#readme).
- Move on to [**`m1-mnist-fit`**](../m1-mnist-fit#readme) to see the same
  architecture driven by Kaun’s high-level training helpers.

[Back to the examples index](../#readme)
