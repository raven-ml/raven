# Roadmap

## Current Status

Raven is in **alpha**. The core stack (Nx -> Rune -> Kaun) works end-to-end: we have successfully trained GPT-2 on CPU using the full Raven stack.

| Library    | Status | What works                                                                |
| ---------- | ------ | ------------------------------------------------------------------------- |
| **nx**     | Alpha  | Full NumPy-like API, linear algebra, FFT, I/O (npy, images)               |
| **rune**   | Alpha  | Reverse and forward-mode AD, vmap, gradient checking                      |
| **kaun**   | Alpha  | Layers, optimizers, training loops, HuggingFace Hub, MNIST/GPT-2 examples |
| **brot**   | Alpha  | All 5 algorithms, full pipeline, HF tokenizer.json compat, training       |
| **talon**  | Alpha  | DataFrames, row operations, aggregations, CSV I/O                         |
| **hugin**  | Alpha  | 2D/3D plots, scatter, bar, contour, images                                |
| **fehu**   | Alpha  | Environments (CartPole, GridWorld, MountainCar), vectorized envs, GAE     |
| **sowilo** | Alpha  | Geometric transforms, filters, edge detection, morphological ops          |
| **quill**  | Alpha  | TUI, web frontend, batch eval, markdown notebook format                   |

APIs will change. Bug reports and feedback are welcome.

## Beta: JIT Compilation & Performance

The beta cycle focuses on **JIT compilation with performance close to PyTorch**.

- Integrate tolk (an OCaml port of tinygrad) as a JIT transformation in Rune
- Target CPU, CUDA, Metal, OpenCL, and HIP
- Kernel fusion and optimization
- Benchmark against PyTorch on standard workloads

## V1: Production-Ready Training & Deployment

V1 makes Raven **production-ready**: train models, deploy them as unikernels or static binaries.

**Training**:
- Gradient accumulation, mixed precision, gradient checkpointing
- Flash attention for efficient transformer training
- ONNX import for PyTorch model portability
- Parallel data loading, layer completions

**Deployment**:
- AOT compilation to standalone binaries (CPU and GPU)
- Inference engine with KV cache, continuous batching, and PagedAttention
- Post-training quantization (INT8/INT4)
- MirageOS unikernel deployment -- tolk AOT generates all compute at compile time, no BLAS dependency, enabling deployment as unikernels
