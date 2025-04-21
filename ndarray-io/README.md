# ndarray-io

Input/Output library for Ndarray

ndarray-io provides functions to load and save Ndarray tensors in image formats,
NumPy formats (`.npy`, `.npz`), and more.

[![Build Status](https://github.com/raven-ml/raven/actions/workflows/ci.yml/badge.svg)](https://github.com/raven-ml/raven/actions/workflows/ci.yml)
[![Opam Version](https://img.shields.io/opam/v/ndarray-io.svg)](https://opam.ocaml.org/packages/ndarray-io)

## Features

- Image I/O: `load_image`, `save_image` (PNG, JPEG, BMP, TGA, GIF)
- NumPy `.npy` format: `load_npy`, `save_npy`
- NumPy `.npz` archives: `load_npz`, `load_npz_member`, `save_npz`
- `packed_ndarray` for runtime-detected dtypes
- Conversion utilities: `to_float32`, `to_uint8`, etc.

## Installation

**Prerequisites**
- OCaml >= 5.0.0
- dune >= 3.17

**Via opam**
```bash
opam install ndarray-io
```

**From source**
```bash
cd ndarray-io
dune build
dune runtest
```

## Quick Start

```ocaml
#require "ndarray";;
#require "ndarray-io";;
open Ndarray
open Ndarray_io

(* Image I/O *)
let img = load_image "photo.png"        (* uint8 tensor [|H;W;C|] or [|H;W|] *)
save_image img "copy.png"

(* NumPy I/O *)
let P arr = load_npy "array.npy"       (* packed ndarray *)
save_npy arr "out.npy"

(* NPZ archive *)
let archive = load_npz "bundle.npz"
match Hashtbl.find_opt archive "data" with
| Some (P a) -> (* use `a` : packed ndarray *)
| None -> failwith "data not found"

(* Converting packed arrays *)
let float_arr = to_float32 (load_npy "array.npy")
```

## Documentation

Full API docs: <https://raven-ml.github.io/raven/ndarray-io/>

Generate locally:
```bash
cd ndarray-io
dune build @doc
dune exec -- odoc serve _build/default/_doc/_html
```

## Testing

```bash
dune runtest
```

## Contributing

See the [Raven monorepo README](../README.md) for guidelines.

## License

ISC License. See [LICENSE](../LICENSE) for details.
