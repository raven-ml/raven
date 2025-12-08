# Nx OxCaml Backend

An experimental high-performance backend for Nx that leverages OxCaml's unboxed types.

## Overview

This backend implements the Nx backend interface using OxCaml's unboxed types for improved performance:

- **Unboxed arithmetic**: Uses `float#`, `int32#`, `int64#` for zero-allocation numeric operations
- **Parallel execution**: Built-in support for parallel operations (currently sequential, Domain support planned)
- **Memory efficiency**: Reduces GC pressure by avoiding boxing/unboxing overhead

## Building

```bash
cd dev/nx-oxcaml
dune pkg lock --root . 
dune build --root . 
```
