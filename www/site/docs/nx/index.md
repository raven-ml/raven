# nx Documentation

Nx is our NumPy. It provides n-dimensional arrays with the operations you need for numerical computing.

## What nx Does

Nx gives you multidimensional arrays backed by OCaml's Bigarray, with a familiar NumPy-like API. Create arrays, slice them, broadcast operations, do linear algebra, all with OCaml's type safety catching your dimension mismatches at compile time.

The key difference from NumPy: we use a pluggable backend architecture. Today that means a pure OCaml CPU backend. Tomorrow it means Metal and CUDA support without changing your code.

## Current Status

Nx works today. You can use it for real numerical computing on CPU. The API covers the essentials:

- Array creation and manipulation
- Broadcasting and element-wise operations  
- Reductions (sum, mean, max, etc.)
- Basic linear algebra (matrix multiply, transpose)
- I/O support for .npy files

What's missing: advanced linear algebra (decompositions, solvers), sparse arrays, and GPU backends. These are coming post-v1.

## Learn More

- [Getting Started](/docs/nx/getting-started/) - Installation and first steps
- [NumPy Comparison](/docs/nx/numpy-comparison/) - Coming from Python
- [API Reference](https://ocaml.org/p/nx/latest/doc/Nx/index.html) - Complete API docs (when released)