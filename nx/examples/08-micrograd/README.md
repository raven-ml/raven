# Micrograd Demo in OCaml

This is an OCaml implementation of Andrej Karpathy's micrograd demo using the nx tensor library.

## Overview

This demo implements:
- A simple automatic differentiation engine (`engine.ml`)
- Neural network components: Neuron, Layer, and MLP (`nn.ml`)
- A training demo on the moon dataset (`demo.ml`)

## Running the Demo

1. Build the project:
```bash
dune build
```

2. Run the demo:
```bash
dune exec ./demo.exe
```

This will:
- Generate a moon dataset using nx.datasets
- Create a 2-layer neural network (2 → 16 → 16 → 1)
- Train the network using SVM loss with gradient descent
- Generate a visualization of the dataset using hugin
- Save the visualization to `micrograd_moon_dataset.png`

## Key Features

- **Automatic Differentiation**: The engine tracks operations and computes gradients automatically
- **Neural Network Components**: Modular design with Neuron, Layer, and MLP modules
- **SVM Loss**: Uses max-margin loss for binary classification
- **Visualization**: Generates decision boundary visualization data

## Architecture

The implementation follows the original micrograd design:
- `Value` type wraps tensors and tracks gradients
- Operations like `+`, `*`, `relu` build a computation graph
- `backward()` computes gradients via reverse-mode autodiff
- Neural network modules compose these operations

## Differences from Python Version

- Uses nx tensors instead of scalars (more efficient)
- OCaml's type system ensures correctness
- Functional programming style with immutable data structures
- Visualization currently shows data points only (decision boundary requires contour plot support)

## Dependencies

- `nx`: Tensor operations
- `nx.datasets`: Dataset generation (make_moons)
- `hugin`: (Optional) Plotting library