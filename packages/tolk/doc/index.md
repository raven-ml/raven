# tolk

Tolk is a port of [tinygrad](https://github.com/tinygrad/tinygrad) in OCaml — a minimal compiler for GPU tensor computation. It takes tensor-level computation graphs, optimizes them, and emits efficient kernels for CPU (via Clang), Metal, CUDA, and OpenCL backends.

## Features

- **Three-level IR** — tensor graphs, kernel DAGs, and linear programs with shared conventions (sub-axes, tagging, map\_children)
- **Symbolic simplification** — three-phase algebraic pipeline for index expressions with div/mod folding
- **Hardware decompositions** — transcendentals, int64 emulation, float type promotion, and late op rewrites
- **Codegen pipeline** — range simplification, GPU dimension mapping, beam search optimization, and linearization
- **Schedule pipeline** — tensor-to-kernel graph transformation with range analysis and multi-device sharding
- **JIT integration** — used by Rune's `jit` transformation to compile and dispatch kernels at runtime

## Architecture

Tolk follows a layered compilation pipeline:

1. **Tensor IR** — high-level operation graph (reductions, reshapes, movement ops)
2. **Schedule** — transforms tensor graphs into kernel graphs via rangeify and indexing
3. **Codegen** — optimizes kernel structure (range simplification, GPU dims, beam search)
4. **Lowering** — lowers to linear program IR (devectorization, expansion, decompositions)
5. **Renderer** — emits backend-specific source code (C, Metal, CUDA, OpenCL)
6. **Runtime** — compiles and dispatches kernels on target devices

## Libraries

- `tolk` — codegen pipeline, renderer, device abstraction, and runtime
- `tolk.ir` — IR definitions (tensor, kernel, program), symbolic simplification, decompositions
- `tolk.cpu` — CPU backend (Clang compilation, ELF loading)
- `tolk.metal` — Metal backend (macOS GPU)
