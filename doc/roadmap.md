# Roadmap

## Current Status

Raven is in **alpha**. The core stack (Nx → Rune → Kaun) works end-to-end: we have successfully trained GPT-2 on CPU using the full Raven stack.

| Library | Status | What works |
|---------|--------|------------|
| **nx** | Alpha | Full NumPy-like API, linear algebra, FFT, I/O (npy, images) |
| **rune** | Alpha | Reverse and forward-mode AD, vmap, gradient checking |
| **kaun** | Alpha | Layers, optimizers, training loops, HuggingFace Hub, MNIST/GPT-2 examples |
| **brot** | Alpha | All 5 algorithms, full pipeline, HF tokenizer.json compat, training |
| **talon** | Alpha | DataFrames, row operations, aggregations, CSV I/O |
| **hugin** | Alpha | 2D/3D plots, scatter, bar, contour, images |
| **fehu** | Alpha | Environments (CartPole, GridWorld, MountainCar), vectorized envs, GAE |
| **sowilo** | Alpha | Geometric transforms, filters, edge detection, morphological ops |
| **quill** | Alpha | TUI, web frontend, batch eval, markdown notebook format |

APIs will change. Bug reports and feedback are welcome.

## Beta: JIT Compilation & Performance

The beta cycle will focus on **JIT compilation with performance close to PyTorch**.

- Complete LLVM-based JIT compiler for Rune
- Target CPU, CUDA, and Metal for hardware acceleration
- Optimize compilation pipeline and runtime performance
- Benchmark against PyTorch on standard workloads

## V1: Developer Experience & Production Scale

Once performance is competitive, V1 will focus on **developer experience and production readiness**:

- Comprehensive documentation and tutorials
- Finalize and stabilize all public APIs
- Delightful developer tooling (Quill notebooks, Kaun-board training dashboard)
- Migration guides for NumPy/PyTorch users
- Multi-GPU and distributed training
- Model serving infrastructure
- Deployment through MirageOS
