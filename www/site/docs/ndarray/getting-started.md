# Getting Started with ndarray

This guide will help you get up and running with ndarray, the foundational array library of the Raven ecosystem.

## Installation

Install ndarray using OPAM:

```bash
opam install ndarray
```

Or build from source:

```bash
git clone https://github.com/raven-ml/raven
cd raven
dune build ndarray
```

## Your First Array

Let's create your first ndarray:

```ocaml
open Ndarray

(* Create a 3x3 matrix of ones *)
let a = Ndarray.ones [3; 3] float32

(* Create a matrix with random values *)
let b = Ndarray.random [3; 3] float32

(* Matrix multiplication *)
let c = Ndarray.matmul a b

(* Element-wise operations *)
let d = Ndarray.add c (Ndarray.scalar 2.0 float32)
```

## Basic Operations

ndarray supports all the fundamental operations you'd expect:

### Array Creation
- `zeros`, `ones`, `random` - Create arrays with specific values
- `linspace`, `arange` - Create sequences
- `from_array` - Convert from OCaml arrays

### Indexing and Slicing
- Index individual elements: `arr.[0; 1]`
- Slice arrays: `arr.[0..2; ..]`
- Boolean indexing for filtering

### Mathematical Operations
- Element-wise: `add`, `mul`, `sin`, `cos`, etc.
- Linear algebra: `matmul`, `dot`, `inv`, `svd`
- Reductions: `sum`, `mean`, `max`, `argmax`

## Next Steps

- Read the [Array Creation guide](/docs/ndarray/arrays/) for comprehensive array creation methods
- Learn about [Broadcasting](/docs/ndarray/broadcasting/) for operations between different shaped arrays
- Explore [Linear Algebra](/docs/ndarray/linalg/) operations for scientific computing
- Check out the [API Reference](/docs/ndarray/api/) for complete documentation