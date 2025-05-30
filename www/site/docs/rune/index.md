# rune Documentation

Rune is our JAX. It brings automatic differentiation to OCaml using algebraic effects.

## What rune Does

Rune lets you write numerical functions in OCaml and automatically compute their derivatives. Need gradients for optimization? Just wrap your function with `grad`. Want it to run fast? Wrap it with `jit` (coming post-v1).

The magic comes from OCaml 5's effect system. While other autodiff libraries modify your code with macros or operator overloading, rune uses effects to transform functions cleanly. Write normal OCaml code, get derivatives for free.

## Current Status

Rune implements reverse-mode automatic differentiation today. You can compute gradients of scalar functions, which is what you need for machine learning.

What works:
- Reverse-mode AD (backpropagation)
- Basic tensor operations
- Gradient computation for neural networks
- Integration with nx arrays

What's coming post-v1:
- JIT compilation to LLVM/Metal/CUDA
- Forward-mode AD
- Higher-order derivatives
- vmap and other JAX-style transformations

## Design

Rune introduces a separate `Tensor` type for differentiable computations. This isn't redundant with nx, it's deliberate. Tensors track computation graphs for autodiff, while nx arrays are just data. Convert between them when crossing the boundary.

## Learn More

- [Getting Started](/docs/rune/getting-started/) - Installation and first steps
- [API Reference](https://ocaml.org/p/rune/latest/doc/Rune/index.html) - Complete API docs (when released)