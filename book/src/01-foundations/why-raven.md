# Why Raven for Modern ML

The modern ML stack revolves around three pillars:

1. **Expressive model authoring** for deep and multimodal architectures.
2. **Differentiable compute** with hardware acceleration.
3. **End-to-end productivity** from experimentation to deployment.

Python ecosystems solve these with NumPy, PyTorch/JAX, and a constellation of tools.  
Raven brings the same capabilities to OCaml while keeping the benefits we rely on in large-scale systems—strong typing, compile-time guarantees, predictable performance, and ergonomic build tooling.

## The Ecosystem at a Glance

| Raven component | Python analogue              | Role in the book                                    |
| --------------- | ---------------------------- | --------------------------------------------------- |
| `Nx`            | NumPy                        | Tensor algebra, data transforms, I/O                |
| `Rune`          | JAX                          | Reverse-mode AD, gradient-based optimizers          |
| `Kaun`          | PyTorch/Flax                 | Neural network layers, training utilities           |
| `Talon`         | pandas/Polars                | Feature engineering, dataset wrangling              |
| `Saga`          | Hugging Face tokenizers      | Text normalization, subword vocabularies            |
| `Sowilo`        | OpenCV                       | Image preprocessing, differentiable vision ops      |
| `Fehu`          | Gymnasium + Stable-Baselines | Reinforcement learning environments and algorithms  |
| `Quill`         | Jupyter                      | Markdown-first notebooks for publishing experiments |

Throughout this book we combine these libraries to deliver the same kinds of projects you would build in Python: transformer backbones, diffusion models, multimodal systems, and production inference services.

## Why OCaml?

OCaml’s type system makes illegal states unrepresentable. Complex model parameter trees, dataset schemas, and pipeline contracts become compile-time guarantees instead of runtime surprises. Algebraic data types help us encode heterogeneity—vital when juggling tensors, strings, and metadata.

Moreover, OCaml 5’s multicore runtime plus effect handlers open the door for advanced differentiable programming (Rune) and JIT compilation. Raven leverages this to target CPUs today and GPUs tomorrow without rewriting your model definitions.

## Where Raven Stands Today

- **Nx** is production-ready on CPU, with pluggable backends in development.
- **Rune** offers reverse-mode AD and `value_and_grad` utilities. JIT and `vmap` are on the roadmap; we note the gaps when they matter.
- **Kaun** is stabilising around MNIST-scale workflows—perfect for demonstrating modern training techniques before scaling.
- **Talon**, **Saga**, and **Sowilo** cover practical data wrangling for tabular, text, and image modalities.
- **Fehu** delivers policy gradient and DQN baselines for reinforcement learning chapters.

We acknowledge the ecosystem is evolving. Wherever features are experimental or upcoming, we provide alternative paths and tips for contributing.

## A Learning Path for Modern ML

1. **Master the building blocks.** Learn how Nx and Rune express tensors, gradients, and randomness.
2. **Compose training pipelines.** Use Kaun and Talon to create rich data pipelines and robust optimization loops.
3. **Advance to state-of-the-art architectures.** Build attention mechanisms, transformers, and generative models using Raven’s abstractions.
4. **Ship to production.** Package inference services, integrate with telemetry, and manage models as first-class OCaml packages.

This chapter is your map. In the next chapter we prepare the OCaml skills you will need to make modern ML feel idiomatic.

