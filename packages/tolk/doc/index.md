# tolk

Tolk is a port of [tinygrad](https://github.com/tinygrad/tinygrad) in OCaml — a minimal compiler for GPU tensor computation. It takes tensor-level computation graphs, optimizes them, and emits kernels for the CPU (C, compiled with Clang), Metal, CUDA and NVIDIA's kernel driver (CUDA C, compiled with NVRTC), and AMD (HIP, compiled with comgr).

Most programs never call tolk themselves: [`Rune.jit`](../../rune/doc/06-compilation.md) compiles through it. This page is for working on the compiler, or for driving it directly.

## Features

- **Three-level IR** — tensor graphs, kernel DAGs, and linear programs with shared conventions (sub-axes, tagging, map\_children)
- **Symbolic simplification** — three-phase algebraic pipeline for index expressions with div/mod folding
- **Hardware decompositions** — transcendentals, int64 emulation, float type promotion, and late op rewrites
- **Codegen pipeline** — range simplification, GPU dimension mapping, beam search optimization, and linearization
- **Schedule pipeline** — tensor-to-kernel graph transformation with range analysis, for one device
- **JIT integration** — used by Rune's `jit` transformation to compile and dispatch kernels at runtime

## Architecture

Tolk follows a layered compilation pipeline:

1. **Tensor IR** — high-level operation graph (reductions, reshapes, movement ops)
2. **Schedule** — transforms tensor graphs into kernel graphs via rangeify and indexing
3. **Codegen** — optimizes kernel structure (range simplification, GPU dims, beam search)
4. **Lowering** — lowers to linear program IR (devectorization, expansion, decompositions)
5. **Renderer** — emits backend-specific source code (C, Metal, CUDA, HIP)
6. **Engine** — runs the compiled kernels on [rig](../../rig/doc/index.md)'s devices (`tolk.engine`)

## Libraries

- `tolk` — the compiler: it takes a program and a target, and returns data (source, binaries, memory plans and schedules). It depends on no runtime.
- `tolk.engine` — runs that output on [rig](../../rig/doc/index.md)'s devices.
