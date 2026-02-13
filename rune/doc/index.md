# rune Documentation

Rune is our JAX. It brings automatic differentiation to OCaml using algebraic effects.

## What rune Does

Rune lets you write numerical functions in OCaml and automatically compute their derivatives. Need gradients for optimization? Just wrap your function with `grad`. Want it to run fast? Wrap it with `jit` (coming post-v1).

<!-- XXX: This is an internal defail. We can mention it as this is interesting to know, but choose the right tone here -->
The magic comes from OCaml 5's effect system. While other autodiff libraries modify your code with macros or operator overloading, rune uses effects to transform functions cleanly. Write normal OCaml code, get derivatives for free.

## Current Status

Rune implements reverse-mode automatic differentiation today. You can compute gradients of scalar functions, which is what you need for machine learning.

What works:
- Reverse-mode AD (backpropagation)
- Basic tensor operations
- Gradient computation for neural networks
- Higher-order derivatives (with nested `grad` transformations)

What's coming post-v1:
- JIT compilation to LLVM/Metal/CUDA
- Forward-mode AD
- vmap and other JAX-style transformations

## Design

<!-- XXX: Review this document, lots of wrong things. We don't have a Tensor type. -->

Rune introduces a separate `Tensor` type for differentiable computations. This isn't redundant with nx, it's deliberate. Tensors track computation graphs for autodiff, while nx arrays are just data. Convert between them when crossing the boundary.

## Learn More

- [Getting Started](/docs/rune/getting-started/) - Installation and first steps
- [API Reference](https://ocaml.org/p/rune/latest/doc/Rune/index.html) - Complete API docs (coming soon)