# `01-creating-arrays`

Build arrays from scratch: constants, ranges, grids, and custom data. This
example walks through the most common ways to create arrays in Nx. Each
creation makes its array on the host.

```bash
dune exec packages/nx/examples/01-creating-arrays/main.exe
```

## What You'll Learn

- Choosing a dtype (`float32`, `float64`, `int32`)
- Filling arrays with constants: `zeros`, `ones`, `full`
- Generating ranges: `arange`, `linspace`, `logspace`
- Building arrays from OCaml data: `create`, `init`
- Diagonal and special matrices: `eye`, `tril`, `triu`
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
| `eye ?k dtype n`               | Ones on the k-th diagonal              |
| `meshgrid ~indexing [x; y]`    | Coordinate grids from 1D arrays        |
| `tril m` / `triu m`            | Lower / upper triangular part          |

## Output Walkthrough

When you run this example, you'll see each array printed with its dtype and
shape, then its elements as nested rows:

```
zeros (2×3):
float32 [2; 3] [[0, 0, 0], [0, 0, 0]]

arange 0..9:
int32 [10] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

5×5 multiplication table:
int32 [5; 5]
  [[1, 2, 3, 4, 5],
   [2, 4, 6, 8, 10],
   [3, 6, 9, 12, 15],
   [4, 8, 12, 16, 20],
   [5, 10, 15, 20, 25]]
```

The multiplication table is built with `init`, which calls a function with the
index array for each element:

```ocaml
Nx.init Nx.int32 [| 5; 5 |] (fun idx ->
    Int32.of_int ((idx.(0) + 1) * (idx.(1) + 1)))
```

`meshgrid` builds one 2D coordinate grid per 1D array, useful for evaluating
functions over a grid. With `` ~indexing:`Xy ``, the first array runs along
columns and the second along rows, as a plane's x and y do:

```ocaml
Nx.meshgrid ~indexing:`Xy [ xs; ys ]
```

```
meshgrid X:
float64 [2; 3] [[0, 1, 2], [0, 1, 2]]
meshgrid Y:
float64 [2; 3] [[0, 0, 0], [1, 1, 1]]
```

## Dtypes

Every array has a dtype that determines the element type and precision. The
first argument to most creation functions is the dtype:

```ocaml
Nx.zeros Nx.float32 [| 2; 3 |]   (* 32-bit floats *)
Nx.ones Nx.float64 [| 3 |]       (* 64-bit floats *)
Nx.arange Nx.int32 0 10 1        (* 32-bit integers *)
```

Nx has 21 dtypes, among them `float16`, `bfloat16`, `complex128`, `bool`,
and integers from 4 to 64 bits.

## Try It

1. Create a 10-element `linspace` from -1.0 to 1.0 and print it.
2. Use `init` to build a 4×4 matrix where each element is the sum of its row
   and column index.
3. Try `Nx.eye ~k:(-1) Nx.float64 4` to see the subdiagonal.

## Next Steps

Continue to [02-infix-and-arithmetic](../02-infix-and-arithmetic/) to learn how
operators make array math read like algebra.
