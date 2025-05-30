# Roadmap

Raven is in pre-alpha. We're building towards our first milestone: an end-to-end demo of training MNIST with visualizations in a Quill notebook.

## Alpha Release

Our immediate goal is demonstrating the complete ML workflow in OCaml:

- Train a neural network on MNIST using Kaun
- Visualize training metrics with Hugin
- Run everything in a Quill notebook
- Achieve reasonable performance on CPU

Once we hit this milestone, we'll stabilize the APIs and cut v1.0.

## Post-Alpha: Training Large Models

After v1.0, we'll shift the goalpost to training large models. Our focus will become mostly performance, as the requirement for real-world ML workflows.

**JIT Compilation** (!!)
- Build JIT compiler for Rune
- Target LLVM, CUDA, and Metal for hardware acceleration

**Accelerated Backends**
- Metal backend for Nx (macOS GPU support)
- CUDA backend for Nx (NVIDIA GPU support)
- Seamless integration with the JIT

**Deep Learning at Scale**
- Complete Kaun with all features needed for modern LLMs
- Train and run inference for Llama 3 as our benchmark
- Enable distributed training across multiple GPUs

## Beyond

Future development depends on community adoption and potential sponsorship. If we achieve sustainable development, priorities include:

**Performance Parity with Python**
- Match NumPy/PyTorch performance through JIT optimization
- Prove OCaml can compete on raw compute, not just developer experience

**Expanded Ecosystem (based on community feedback)**
- Dataframe manipulation library 
- Domain-specific libraries for common workflows

**Distributed Computing**
- First-class distributed training support
- Tools for deploying models at scale

But first, we need to prove the concept works. That starts with MNIST in a notebook.
