# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Raven is a comprehensive OCaml ecosystem for machine learning and data science, comparable to Python's NumPy/JAX/Matplotlib stack. The project is in pre-alpha and consists of multiple interconnected libraries.

### Core Libraries Architecture

**Production Libraries:**
- **ndarray/**: Core N-dimensional array library (NumPy-equivalent) with CPU backend
- **hugin/**: Publication-quality plotting library (Matplotlib-equivalent) 
- **rune/**: Automatic differentiation and JIT compilation (JAX-inspired)
- **quill/**: Interactive notebook environment (Jupyter-equivalent)

**Extension Libraries:**
- **ndarray-io/**: Input/output for various data formats
- **ndarray-cv/**: Computer vision operations  
- **ndarray-datasets/**: ML dataset loaders

**Next-Generation Development (dev/next/):**
- **nx/**: Redesigned tensor library with clean frontend/backend separation
- **rune/**: Advanced auto-differentiation with OCaml 5 algebraic effects

## Build System & Commands

```bash
# Build entire ecosystem
dune build

# Build specific libraries
dune build ndarray hugin rune quill
dune build dev/next/nx dev/next/rune  # Next-gen libraries

# Run tests
dune test
dune test ndarray/test  # Specific library tests

# Interactive REPL
dune utop ndarray
dune utop dev/next/rune/lib  # Next-gen development

# Build documentation  
dune build @doc

# Quill notebook server
dune exec quill -- serve quill/example/
dune exec quill -- exec quill/example/demo.md  # Execute markdown
```

## Architecture Patterns

### Multi-Backend Design
The next-generation architecture (dev/next/) introduces a clean separation:
```
Frontend API → Backend Interface → Concrete Backends (CPU/GPU)
```

**Key Files:**
- `dev/next/nx/lib/core/backend_intf.ml`: Minimal backend interface (~50 operations)
- `dev/next/nx/lib/core/frontend.ml`: High-level NumPy-like API
- `dev/next/nx/lib/native/`: CPU backend implementation

### Effect-Based Transformations (Next-Gen)
Rune uses OCaml 5 algebraic effects for non-intrusive transformations:
```ocaml
let f x = Tensor.add x x in
let grad_f = grad f in          (* Automatic differentiation *)
let jit_f = jit f in           (* JIT compilation *)
let composed = grad (jit f)     (* Composable transformations *)
```

## Development Workflows

### Working with Main Libraries
1. Most stable APIs are in the root-level libraries (ndarray/, hugin/, etc.)
2. Follow existing patterns in `lib/` directories
3. Tests use Alcotest framework
4. Examples are in each library's `example/` directory

### Working with Next-Gen Libraries (dev/next/)
1. These are experimental rewrites using advanced OCaml features
2. Focus on `dev/next/nx/` for tensor operations
3. `dev/next/rune/` uses algebraic effects (requires OCaml 5)
4. Architecture inspired by JAX and Tinygrad

### Library Dependencies
- All libraries depend on `ndarray` as the core
- Quill integrates all visualization and computation libraries
- Next-gen libraries are independent experiments

## Key Conventions

### Module Structure
- Libraries use `lib/` for source code, `test/` for tests
- Core interfaces are in `*_intf.ml` files
- Backend implementations in dedicated subdirectories

### Type Safety
- Heavy use of GADTs for type-safe tensor operations
- Phantom types for device and dtype tracking
- Effect types for transformation safety (next-gen)

### Performance Considerations
- Zero-copy operations where possible
- Bigarray-based storage for numerical data
- JIT compilation for GPU backends (experimental)

## Common Issues

### OCaml Version Compatibility
- Main libraries: OCaml >= 4.14 (some >= 5.0)
- Next-gen libraries: Require OCaml 5 for algebraic effects

### Backend Selection
- Production: CPU backend via Bigarray
- Experimental: Metal (macOS), CUDA planned
- Use appropriate library based on platform and performance needs

### Development Branch Strategy
- `main`: Stable production libraries
- `next`: Experimental next-generation development
- Current focus is on next-gen architecture in `dev/next/`