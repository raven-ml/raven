# Ndarray

N-dimensional array library for OCaml.

Ndarray is the core component of the Raven ecosystem, providing efficient
numerical computation with multi-device support. It offers NumPy-like
functionality with the benefits of OCaml's strong static type system.

## Features

- Multi-dimensional arrays (tensors) with arbitrary rank
- Support for data types: float16, float32, float64, int8, int16, int32,
  int64, uint8, uint16, complex32, complex64
- Flexible memory layouts: C-contiguous and strided
- Zero-copy slicing, reshaping, and broadcasting
- Element-wise and scalar operations (add, sub, mul, div, map, etc.)
- Linear algebra routines (`dot`, matrix multiplication, transpose,
  sum, mean, argmax, etc.)
- Optimized CPU backend; pure OCaml interface leveraging Bigarray
- Seamless integration with the Raven ecosystem: `ndarray-io`,
  `ndarray-cv`, `quill`, `hugin`, etc.

## Quick Start

```ocaml
open Ndarray

(* Create a 2x3 tensor *)
let a = create float32 [|2;3|] [|1.; 2.; 3.; 4.; 5.; 6.|]

(* Fill a tensor with ones *)
let b = full float32 [|2;3|] 1.0

(* Element-wise addition *)
let c = add a b

(* Matrix multiplication *)
let x = create float32 [|2;3|] [|1.;2.;3.;4.;5.;6.|]
let y = create float32 [|3;2|] [|7.;8.;9.;10.;11.;12.|]
let z = dot x y

(* Reduction: sum across an axis *)
let s = sum ~axes:[|1|] x
```

## Contributing

See the [Raven monorepo README](../README.md) for contribution guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
