# nx

Nx provides n-dimensional arrays for OCaml. An array is a value: an operation
gives a new array and never changes its operands. Each array has a dtype, a
shape and a placement on a set of devices. It is the numerical foundation of
the Raven ecosystem.

## Features

- **21 data types**: floats from 4 to 64 bits, integers from 4 to 64 bits,
  complex64 and complex128, bool and bit
- **Broadcasting**: binary operations stretch axes of extent 1
- **Views**: reshape, transpose and slice without copying where strides allow
- **Linear algebra**: matmul, solve, Cholesky, QR, LU, SVD, eigendecomposition
- **FFT**: complex, real and Hermitian transforms, cosine and sine transforms
- **Signal processing**: convolution, correlation, short-time transforms
- **Random numbers**: samplers that draw from explicit keys or a scope's key
- **I/O**: images (PNG, JPEG), NumPy files (.npy, .npz), SafeTensors and GGUF
- **Devices**: arrays lie on the host or on a set of devices, and an operation
  computes where its operands lie

## Quick Start

```ocaml
let () =
  (* Five evenly spaced float32 values from 0 to 10, and their squares. *)
  let x = Nx.linspace Nx.float32 0. 10. 5 in
  let y = Nx.Infix.(x * x) in
  Nx.print x;
  Nx.print y;

  (* Two independent keys from seed 0, one per random matrix. *)
  let keys = Nx.Rng.split (Nx.Rng.key 0) in
  let a = Nx.Rng.uniform ~key:keys.(0) Nx.float32 [| 3; 3 |] in
  let b = Nx.Rng.uniform ~key:keys.(1) Nx.float32 [| 3; 3 |] in
  let c = Nx.matmul a b in
  Format.printf "matmul shape: %a@." Nx.pp_shape (Nx.shape c)
```

It prints:

```
float32 [5] [0, 2.5, 5, 7.5, 10]
float32 [5] [0, 6.25, 25, 56.25, 100]
matmul shape: [3; 3]
```

`Nx.linspace Nx.float32 0. 10. 5` takes the dtype `Nx.float32`, the first and
last values, and the number of values. Like every array made from numbers, it
is a host array. Inside `Nx.Infix.( … )`, `*` multiplies arrays element by element;
outside, it is OCaml's integer product. `Nx.print` prints an array's dtype,
shape and elements.

`Nx.Rng.key 0` is the key of the seed 0, and `Nx.Rng.split` derives two
independent keys from it. `Nx.Rng.uniform ~key dtype shape` draws values in
[0, 1) from its key, so the same key always gives the same matrix.
`Nx.matmul` is the matrix product, and `Nx.pp_shape` prints a shape.

## Next Steps

- [Getting Started](01-getting-started.md): installation, dtypes, slicing, broadcasting
- [Array Operations](02-array-operations.md): reshaping, views, joining, splitting
- [Linear Algebra](03-linear-algebra.md): decompositions, solvers, FFT
- [Input/Output](04-io.md): images, NumPy, SafeTensors and GGUF files, and saving and loading values
- [NumPy Comparison](05-numpy-comparison.md): side-by-side reference
- [Structures](06-structures.md): records of arrays that transformations, optimizers and saving walk
