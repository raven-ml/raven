# `01-creating-arrays`

Build arrays from scratch — constants, ranges, grids, and custom data. This
example walks through the most common ways to create arrays in Nx.

```bash
dune exec nx/examples/01-creating-arrays/main.exe
```

## What You'll Learn

- Choosing a dtype (`float32`, `float64`, `int32`)
- Filling arrays with constants: `zeros`, `ones`, `full`
- Generating ranges: `arange`, `linspace`, `logspace`
- Building arrays from OCaml data: `create`, `init`
- Diagonal and special matrices: `identity`, `eye`, `tril`, `triu`
- Coordinate grids with `meshgrid`

## Key Functions

| Function                       | Purpose                                |
| ------------------------------ | -------------------------------------- |
| `zeros dtype shape`            | Array of all zeros                     |
| `ones dtype shape`             | Array of all ones                      |
| `full dtype shape value`       | Array filled with a value              |
| `arange dtype start stop step` | Integer-stepped range (exclusive stop) |
| `linspace dtype start stop n`  | Evenly spaced floats                   |
| `logspace dtype start stop n`  | Logarithmically spaced values          |
| `create dtype shape data`      | Array from an OCaml array              |
| `init dtype shape f`           | Array from a function of indices       |
| `identity dtype n`             | n×n identity matrix                    |
| `eye ?k dtype n`               | Ones on the k-th diagonal              |
| `meshgrid x y`                 | Coordinate grids from 1D arrays        |
| `tril m` / `triu m`            | Lower / upper triangular part          |

## Output Walkthrough

When you run this example, you'll see arrays printed in a compact format:

```
zeros (2×3):
[[0, 0, 0],
 [0, 0, 0]]

arange 0..9:
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

5×5 multiplication table:
[[1, 2, 3, 4, 5],
 [2, 4, 6, 8, 10],
 [3, 6, 9, 12, 15],
 [4, 8, 12, 16, 20],
 [5, 10, 15, 20, 25]]
```

The multiplication table is built with `init`, which calls a function with the
index array for each element:

```ocaml
init int32 [| 5; 5 |] (fun idx ->
    Int32.of_int ((idx.(0) + 1) * (idx.(1) + 1)))
```

`meshgrid` builds a pair of 2D coordinate grids from two 1D arrays — useful
for evaluating functions over a grid:

```
meshgrid X:
[[0, 1, 2],
 [0, 1, 2]]
meshgrid Y:
[[0, 0, 0],
 [1, 1, 1]]
```

## Dtypes

Every array has a dtype that determines the element type and precision. The
first argument to most creation functions is the dtype:

```ocaml
zeros float32 [| 2; 3 |]   (* 32-bit floats *)
ones float64 [| 3 |]       (* 64-bit floats *)
arange int32 0 10 1        (* 32-bit integers *)
```

Nx supports 18 dtypes including `Float16`, `BFloat16`, `Complex128`, `Bool`,
and various integer widths.

## Try It

1. Create a 10-element `linspace` from -1.0 to 1.0 and print it.
2. Use `init` to build a 4×4 matrix where each element is the sum of its row
   and column index.
3. Try `eye ~k:(-1) float64 4` to see the subdiagonal.

## Next Steps

Continue to [02-infix-and-arithmetic](../02-infix-and-arithmetic/) to learn how
the Infix module makes array math read like algebra.
