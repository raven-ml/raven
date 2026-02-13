# `01-xor`

This example trains a two-layer perceptron on the XOR truth table. It shows how
to build a minimal supervised training loop with Rune tensors,
`Kaun.value_and_grad`, and in-place optimizer updates.

- **Source**: [`main.ml`](./main.ml)

```bash
cd kaun/examples/01-xor
dune exec kaun/examples/01-xor/main.exe
```

### What’s happening?

- `Layer.sequential` wires up a tiny MLP with `tanh` and `sigmoid` activations.
- `Kaun.init` seeds parameters deterministically via `Rune.Rng.key 42`.
- `Optimizer.adam` runs with a fixed learning rate (`Optimizer.Schedule.constant 0.1`).
- `value_and_grad` differentiates the closure that runs the forward pass and measures binary cross-entropy.
- `Optimizer.step` and `Optimizer.apply_updates_inplace` update the parameters in place each epoch.

### Sample output

```
Epoch 500: Loss = 0.000629

Final predictions (should be close to [0; 1; 1; 0]):
Nx Info:
  Shape: [4; 1]
  Dtype: float32
  Strides: [1; 1]
  Offset: 0
  Size: 4
  Data: [[1.32195e-05],
         [0.999169],
         [0.999513],
         [0.00117599]]
```

### Try this

- Trim or raise the epoch count to see how quickly the network saturates.
- Replace `Layer.tanh` with `Layer.relu` to observe slower convergence on XOR.
- Swap in `Optimizer.sgd` to observe how momentum-free updates behave.

### Next steps

- Continue with [**`02-xor-eval`**](../02-xor-eval#readme) to add explicit
  prediction and accuracy helpers.
- When you’re ready for richer datasets, start the model track at
  [**`m0-mnist-basics`**](../m0-mnist-basics#readme).

[Back to the examples index](../#readme)
