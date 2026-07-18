# Nx OxCaml Backend

An experimental high-performance backend for Nx that leverages OxCaml's unboxed types.

## Overview

This backend implements the Nx backend interface using OxCaml's unboxed types for improved performance:

- **Unboxed arithmetic**: Uses `float#`, `int32#`, `int64#` for zero-allocation numeric operations
- **Parallel execution**: Built-in support for parallel operations (currently sequential, Domain support planned)
- **Memory efficiency**: Reduces GC pressure by avoiding boxing/unboxing overhead

## Building

Nx uses OCaml 5.5's first-class polymorphic parameters in its typed parameter
tree API. The packaged OxCaml variants are currently based on older OCaml
releases, so this backend cannot be built against the current Nx source until
an OCaml 5.5-compatible OxCaml toolchain is available. Lowering Nx's compiler
floor would lose the type-safe mixed-dtype parameter tree API and is not a
viable workaround.

With a compatible OxCaml compiler, build from this directory:

```bash
cd packages/nx-oxcaml
dune build --root .
```

## Benchmark Results

See [bench/](bench/README.md) for the comparative benchmark workflow against
the Nx C backend.
