# Getting Started

This guide covers installation, data types, array creation, slicing, broadcasting, and basic operations.

## Installation

<!-- $MDX skip -->
```bash
opam install nx
```

Or build from source:

<!-- $MDX skip -->
```bash
git clone https://github.com/raven-ml/raven
cd raven && dune build packages/nx
```

Add to your `dune` file:

<!-- $MDX skip -->
```dune
(executable
 (name main)
 (libraries nx))
```

## Creating Arrays

```ocaml
open Nx

let () =
  (* From explicit values: provide dtype, shape, and flat data *)
  let a = create Float32 [|2; 3|] [|1.; 2.; 3.; 4.; 5.; 6.|] in
  print_data a;

  (* Filled arrays *)
  let z = zeros Float32 [|3; 3|] in
  let o = ones Int32 [|5|] in
  let f = full Float64 [|2; 2|] 3.14 in
  ignore (z, o, f);

  (* Ranges and sequences *)
  let r = arange Int32 0 10 1 in          (* [0, 1, ..., 9] *)
  let l = linspace Float32 0. 1. 5 in     (* 5 points in [0, 1] *)
  ignore (r, l);

  (* Random arrays *)
  let x = rand Float32 [|3; 4|] in
  let y = randn Float32 [|3; 4|] in
  ignore (x, y);

  (* Special matrices *)
  let i = eye Float32 3 in               (* 3×3 identity *)
  print_data i
```

## Data Types

Every array has a `dtype` that determines its element type. Common dtypes:

| Dtype | OCaml type | Typical use |
|-------|-----------|-------------|
| `Float32` | `float` | Neural networks, images |
| `Float64` | `float` | Scientific computing |
| `Int32` | `int32` | Integer data, indices |
| `Int64` | `int64` | Large integers |
| `Bool` | `bool` | Masks, conditions |
| `Complex128` | `Complex.t` | Signal processing |

Nx does not automatically cast between types. Convert explicitly with `astype`:

```ocaml
open Nx

let () =
  let x = create Int32 [|3|] [|1l; 2l; 3l|] in
  let y = astype Float32 x in
  print_data y   (* [1. 2. 3.] as float32 *)
```

## Array Properties

```ocaml
open Nx

let () =
  let x = rand Float32 [|2; 3; 4|] in
  Printf.printf "shape: [|%s|]\n"
    (Array.to_list (shape x) |> List.map string_of_int |> String.concat "; ");
  Printf.printf "ndim: %d\n" (ndim x);         (* 3 *)
  Printf.printf "size: %d\n" (size x);          (* 24 *)
  Printf.printf "dtype: %s\n" (Dtype.to_string (dtype x))
```

## Element-wise Operations

Binary operations work element-wise and support broadcasting:

```ocaml
open Nx

let () =
  let a = create Float32 [|3|] [|1.; 2.; 3.|] in
  let b = create Float32 [|3|] [|4.; 5.; 6.|] in

  let _ = add a b in       (* [5. 7. 9.] *)
  let _ = mul a b in       (* [4. 10. 18.] *)
  let _ = sub a b in       (* [-3. -3. -3.] *)
  let _ = div a b in       (* [0.25 0.4 0.5] *)

  (* Scalar operations *)
  let _ = add a (scalar Float32 10.) in   (* [11. 12. 13.] *)

  (* Math functions *)
  let _ = sin a in
  let _ = exp a in
  let _ = sqrt (abs a) in
  ()
```

## Reductions

```ocaml
open Nx

let () =
  let x = create Float32 [|2; 3|] [|1.; 2.; 3.; 4.; 5.; 6.|] in

  (* Reduce all elements *)
  Printf.printf "sum = %.1f\n" (item [] (sum x));
  Printf.printf "mean = %.1f\n" (item [] (mean x));

  (* Reduce along an axis *)
  let col_sums = sum ~axes:[0] x in    (* sum each column *)
  print_data col_sums;   (* [5. 7. 9.] *)

  let row_sums = sum ~axes:[1] x in    (* sum each row *)
  print_data row_sums    (* [6. 15.] *)
```

## Slicing and Indexing

### Basic indexing

```ocaml
open Nx

let () =
  let x = create Int32 [|3; 3|] [|1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l; 9l|] in

  (* Get a row *)
  let row = get [1] x in           (* [4, 5, 6] *)
  print_data row;

  (* Get a scalar *)
  let v = item [1; 2] x in        (* 6l *)
  Printf.printf "x[1,2] = %ld\n" v
```

### Advanced slicing

```ocaml
open Nx

let () =
  let x = create Int32 [|4; 4|]
    [|1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l;
      9l; 10l; 11l; 12l; 13l; 14l; 15l; 16l|] in

  (* Range: rows 0 to 2 (exclusive), all columns *)
  let sub = slice [R (0, 2); A] x in
  print_data sub;

  (* Single index on one axis, range on another *)
  let row1_cols = slice [I 1; R (0, 3)] x in
  print_data row1_cols;

  (* Gather specific indices *)
  let picked = slice [L [0; 3]; L [1; 2]] x in
  print_data picked
```

Index types: `I i` (single index), `R (start, stop)` (half-open range), `Rs (start, stop, step)` (strided range), `L indices` (gather), `A` (all), `N` (new axis).

## Broadcasting

Operations automatically broadcast arrays with compatible shapes. Dimensions are aligned from the right, and each pair must be equal or one must be 1:

```ocaml
open Nx

let () =
  let matrix = ones Float32 [|3; 4|] in
  let row = create Float32 [|1; 4|] [|10.; 20.; 30.; 40.|] in
  let result = add matrix row in    (* row added to every row *)
  print_data result
```

## Matrix Multiplication

```ocaml
open Nx

let () =
  let a = rand Float32 [|3; 4|] in
  let b = rand Float32 [|4; 2|] in
  let c = matmul a b in
  Printf.printf "(%d×%d) × (%d×%d) = (%d×%d)\n"
    (dim 0 a) (dim 1 a) (dim 0 b) (dim 1 b) (dim 0 c) (dim 1 c)
```

## Next Steps

- [Array Operations](/docs/nx/array-operations/) — reshaping, views, joining, transposing
- [Linear Algebra](/docs/nx/linear-algebra/) — decompositions, solvers, FFT
- [NumPy Comparison](/docs/nx/numpy-comparison/) — side-by-side reference if you're coming from Python
