# nx

Nx provides n-dimensional arrays with NumPy-like semantics and OCaml's type safety. It is the numerical foundation for the entire Raven ecosystem.

## Features

- **19 data types** — float16 through float64, int4 through int64, complex64/128, bool
- **Broadcasting** — automatic shape matching for binary operations
- **Views** — reshape, transpose, and slice without copying data
- **Linear algebra** — matmul, solve, cholesky, QR, SVD, eigendecomposition
- **FFT** — full suite of discrete Fourier transforms
- **Signal processing** — convolution, correlation, filtering
- **I/O** — read and write images (PNG, JPEG), NumPy files (.npy, .npz)
- **Pluggable backends** — default C backend, extensible architecture

## Quick Start

```ocaml
open Nx

let () =
  (* Create and manipulate arrays *)
  let x = linspace Float32 0. 10. 5 in
  let y = mul x x in
  Printf.printf "x = "; print_data x;
  Printf.printf "y = x² = "; print_data y;

  (* Matrix operations *)
  let a = rand Float32 [|3; 3|] in
  let b = rand Float32 [|3; 3|] in
  let c = matmul a b in
  Printf.printf "matmul shape: [|%d; %d|]\n" (dim 0 c) (dim 1 c)
```

## Next Steps

- [Getting Started](/docs/nx/getting-started/) — installation, dtypes, slicing, broadcasting
- [Array Operations](/docs/nx/array-operations/) — reshaping, views, joining, splitting
- [Linear Algebra](/docs/nx/linear-algebra/) — decompositions, solvers, FFT
- [Input/Output](/docs/nx/io/) — images, npy, npz files
- [NumPy Comparison](/docs/nx/numpy-comparison/) — side-by-side reference
