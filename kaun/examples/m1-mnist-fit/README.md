# `m1-mnist-fit`

This example runs the same MNIST CNN as `m0-mnist-basics`, but hands most of the
work to Kaun’s high-level helpers. `Training.fit` manages batching, optimizer
steps, metric tracking, and progress reporting so you can focus on
configuration.

- **Source**: [`main.ml`](./main.ml)

```bash
cd kaun/examples/m1-mnist-fit
dune exec kaun/examples/m1-mnist-fit/main.exe
```

### Highlights

- `Dataset.prepare` shuffles, batches, and prefetches examples with a single
  call for both train and validation splits.
- `Training.fit` wires the model, loss function, optimizer, and metrics into a
  managed epoch loop that prints progress and captures history.
- `Training.History` exposes final and best metrics, making it easy to report
  outcomes or trigger checkpoints.

### Sample output

```
Loading datasets...
Datasets ready!

Epoch 1/10 │ train_loss=2.30  train_accuracy=0.11 │ val_loss=2.18  val_accuracy=0.24
...
Epoch 10/10 │ train_loss=0.09  train_accuracy=0.97 │ val_loss=0.08  val_accuracy=0.98

=== Training Complete ===
Final train loss: 0.0865
Final train accuracy: 0.9732
Best val loss: 0.0794
```

### Experiments

- Increase `prefetch` or tweak `batch_size` in `Dataset.prepare` to observe
  throughput changes.
- Swap in a different optimizer (e.g. `Optimizer.nadam`) by altering a single
  argument to `Training.fit`.
- Extend the metrics list to include precision/recall via
  `Metrics.Collection.create`.

### Next steps

- Compare with [**`m0-mnist-basics`**](../m0-mnist-basics#readme) to understand
  the boilerplate `Training.fit` eliminates.
- Move forward to [**`m2-bert`**](../m2-bert#readme) to explore pretrained
  transformer embeddings.

[Back to the examples index](../#readme)
