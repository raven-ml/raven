# Nx

N-dimensional arrays for OCaml.

Nx is the core component of the Raven ecosystem. An array is a value with a
dtype, a shape and a placement on a set of devices; an operation gives a new
array and never changes its operands. OCaml's type system keeps dtypes and
device sets apart.

## Features

- Multi-dimensional arrays of arbitrary rank
- 21 data types: floats from 4 to 64 bits (float4_e2m1fn, float8, float16,
  bfloat16, float32, float64), signed and unsigned integers from 4 to 64 bits,
  complex64, complex128, bool and bit
- Slicing, reshaping, and broadcasting as views where strides allow, without
  copying
- Element-wise and scalar operations, reductions and scans
- Linear algebra (`matmul`, `solve`, `cholesky`, `qr`, `svd`, `eigh`, ...)
  and Fourier transforms
- Random numbers from explicit keys
- Host computation by C kernels; arrays on device sets, each operation
  computing where its operands lie
- I/O support: images (PNG, JPEG), NumPy files (.npy, .npz), SafeTensors and
  GGUF
- Integration with the Raven ecosystem: `rune`, `kaun`, `hugin`, `quill`, etc.

## Quick Start

```ocaml
(* Create a 2x3 array *)
let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]

(* Fill an array with ones *)
let b = Nx.full Nx.float32 [| 2; 3 |] 1.0

(* Element-wise addition, as a function or inside a local open *)
let c = Nx.add a b
let c' = Nx.Infix.(a + b)

(* Matrix multiplication *)
let x = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |]
let y = Nx.create Nx.float32 [| 3; 2 |] [| 7.; 8.; 9.; 10.; 11.; 12. |]
let z = Nx.matmul x y

(* Reduction: sum across an axis *)
let s = Nx.sum ~axes:[ 1 ] x
```

## Contributing

See the [Raven monorepo README](../README.md) for contribution guidelines.

## License

ISC License. See [LICENSE](../../LICENSE) for details.
