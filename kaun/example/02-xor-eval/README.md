# `02-xor-eval`

This example extends the XOR network by adding evaluation helpers. After
training the two-layer perceptron, it generates binary predictions, converts
them to integers, and reports accuracy—showing how to reuse trained parameters
for inference.

- **Source**: [`xor_eval.ml`](./xor_eval.ml)

```bash
cd kaun/example/02-xor-eval
dune exec kaun/example/02-xor-eval/xor_eval.exe
```

### What it covers

- Reuses the model from `01-xor` but separates training, prediction, and
  evaluation into clear functions.
- Demonstrates `Rune.to_bigarray` to bridge Kaun tensors with standard OCaml
  array processing.
- Computes integer predictions with a configurable probability threshold and
  derives accuracy without additional libraries.

### Sample output

```
Epoch 100: loss=0.006596
Epoch 200: loss=0.002419
Epoch 300: loss=0.001346
Epoch 400: loss=0.000880
Epoch 500: loss=0.000629

Predictions: [0; 1; 1; 0]
Accuracy: 1.00
```

### Next steps

- Jump to [**`m0-mnist-basics`**](../m0-mnist-basics#readme) to apply the same
  loop structure to MNIST with explicit batching and metrics.
- Revisit [**`01-xor`**](../01-xor#readme) if you want to see the pure training
  loop without evaluation helpers.

[Back to the examples index](../#readme)
