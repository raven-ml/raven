# Getting Started with nx

This guide shows you how to use nx for numerical computing in OCaml.

## Installation

Nx isn't released yet. When it is, you'll install it with:

```bash
opam install nx
```

For now, build from source:

```bash
git clone https://github.com/raven-ml/raven
cd raven
dune build nx
```

## First Steps

Here's a simple example that actually works with nx:

```ocaml
open Nx

(* Create arrays *)
let a = ones float32 [|3; 3|]
let b = rand float32 [|3; 3|]

(* Matrix multiplication *)
let c = matmul a b

(* Element-wise operations *)
let d = add c (scalar float32 2.0)

(* Print the result *)
print_tensor d
```

## Key Concepts

**Types matter.** Every array has a dtype (like `float32` or `int64`). No automatic casting, if you want to convert types, use `astype`.

**Shapes are arrays.** When specifying dimensions, use `[|3; 3|]` not `[3; 3]`. This is OCaml's array literal syntax.

**Slicing uses functions.** Instead of NumPy's `arr[0:2, :]`, nx uses:
```ocaml
let slice = get_slice [R [0; 2]; All] arr
```

**Broadcasting works like NumPy.** Arrays with compatible shapes can be used together:
```ocaml
let matrix = rand float32 [|3; 4|]
let row = rand float32 [|1; 4|]
let result = add matrix row  (* broadcasts row to each matrix row *)
```

## Common Operations

```ocaml
(* Creation *)
let x = zeros float32 [|2; 3|]
let y = ones int32 [|5|]
let z = full float64 [|3; 3|] 3.14
let seq = arange 0 10
let points = linspace float32 0. 1. 100

(* Manipulation *)
let reshaped = reshape [|6|] x
let transposed = transpose matrix
let concatenated = concatenate ~axis:0 [x; y]

(* Math *)
let sum_all = sum arr
let mean_axis0 = mean ~axes:[|0|] arr
let maximum = max arr
let product = mul a b  (* element-wise *)
let dot_product = matmul a b  (* matrix multiply *)

(* I/O *)
save_npy "data.npy" arr
let loaded = load_npy float32 "data.npy"
```

## Next Steps

Check out the [NumPy Comparison](/docs/nx/numpy-comparison/) if you're coming from Python. The comparison shows real, working code for both libraries.

When nx is released, full API documentation will be available. For now, the source code in `nx/lib/nx.mli` is your best reference.