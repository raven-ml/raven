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

## Operators and Opens

The examples on these pages name every function with its module, `Nx.zeros`.
The arithmetic operators on arrays, `+`, `-`, `*`, `/`, `**` and their scalar
forms such as `+$`, live in `Nx.Infix`, so `open Nx` keeps OCaml's integer
arithmetic. A local open, `Nx.Infix.(a * b + c)`, applies them to arrays
inside the parentheses alone, and `Nx.(Infix.(sin x * x +$ 1.))` also
reaches `Nx`'s functions. Inside such an open, integer arithmetic is
`Stdlib.( + )` ([nx.mli, Operators](../lib/nx.mli)).

## Creating Arrays

An array made from a dtype, a shape and numbers is a host array: it lies in
the host's memory, and the host computes on it.

```ocaml
let () =
  (* From explicit values: provide dtype, shape, and flat data *)
  let a = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  Nx.print a;

  (* Filled arrays *)
  let z = Nx.zeros Nx.float32 [| 3; 3 |] in
  let o = Nx.ones Nx.int32 [| 5 |] in
  let f = Nx.full Nx.float64 [| 2; 2 |] 3.14 in
  ignore (z, o, f);

  (* Ranges and sequences *)
  let r = Nx.arange Nx.int32 0 10 1 in        (* [0, 1, ..., 9] *)
  let l = Nx.linspace Nx.float32 0. 1. 5 in   (* 5 points in [0, 1] *)
  ignore (r, l);

  (* Random arrays, drawn from the keys of a scope rooted at seed 42 *)
  Nx.Rng.with_key (Nx.Rng.key 42) (fun () ->
      let x = Nx.Rng.uniform Nx.float32 [| 3; 4 |] in
      let y = Nx.Rng.normal Nx.float32 [| 3; 4 |] in
      ignore (x, y));

  (* Special matrices *)
  let i = Nx.eye Nx.float32 3 in              (* 3×3 identity *)
  Nx.print i
```

A sampler such as `Nx.Rng.uniform` draws from the key it is given as `~key`,
or else from the next key of the enclosing `Nx.Rng.with_key` scope, so a
program that fixes its keys draws the same numbers on every run.

## Data Types

Every array has a `dtype` that determines its element type. Common dtypes:

| Dtype | OCaml type | Typical use |
|-------|-----------|-------------|
| `Nx.float32` | `float` | Neural networks, images |
| `Nx.float64` | `float` | Scientific computing |
| `Nx.int32` | `int32` | Integer data |
| `Nx.int64` | `int64` | Indices, large integers |
| `Nx.bool` | `bool` | Masks, conditions |
| `Nx.complex128` | `Complex.t` | Signal processing |

The operands of an operation share a dtype by their types, so Nx never
converts one implicitly. Convert explicitly with `cast`:

```ocaml
let () =
  let x = Nx.create Nx.int32 [| 3 |] [| 1l; 2l; 3l |] in
  let y = Nx.cast Nx.float32 x in
  Nx.print y   (* float32 [3] [1, 2, 3] *)
```

## Array Properties

```ocaml
let () =
  let x = Nx.Rng.uniform Nx.float32 [| 2; 3; 4 |] in
  Format.printf "shape: %a@." Nx.pp_shape (Nx.shape x);   (* [2; 3; 4] *)
  Printf.printf "ndim: %d\n" (Nx.ndim x);                  (* 3 *)
  Printf.printf "size: %d\n" (Nx.numel x);                 (* 24 *)
  Format.printf "dtype: %a@." Nx.pp_dtype (Nx.dtype x)     (* float32 *)
```

## Element-wise Operations

Binary operations work element-wise and support broadcasting:

```ocaml
let () =
  let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Nx.create Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in

  let _ = Nx.add a b in       (* [5, 7, 9] *)
  let _ = Nx.mul a b in       (* [4, 10, 18] *)
  let _ = Nx.sub a b in       (* [-3, -3, -3] *)
  let _ = Nx.div a b in       (* [0.25, 0.4, 0.5] *)

  (* The same with operators, inside a local open *)
  let _ = Nx.Infix.((a + b) * a) in (* [5, 14, 27] *)

  (* Scalar operations *)
  let _ = Nx.add a (Nx.scalar Nx.float32 10.) in   (* [11, 12, 13] *)
  let _ = Nx.Infix.(a +$ 10.) in                         (* the same *)

  (* Math functions *)
  let _ = Nx.sin a in
  let _ = Nx.exp a in
  let _ = Nx.sqrt (Nx.abs a) in
  ()
```

## Reductions

```ocaml
let () =
  let x = Nx.create Nx.float32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in

  (* Reduce all elements to a 0-d array, read by [item []] *)
  Printf.printf "sum = %.1f\n" (Nx.item [] (Nx.sum x));
  Printf.printf "mean = %.1f\n" (Nx.item [] (Nx.mean x));

  (* Reduce along an axis *)
  let col_sums = Nx.sum ~axes:[ 0 ] x in    (* sum each column *)
  Nx.print col_sums;                         (* [5, 7, 9] *)

  let row_sums = Nx.sum ~axes:[ 1 ] x in    (* sum each row *)
  Nx.print row_sums                          (* [6, 15] *)
```

## Slicing and Indexing

### Basic indexing

```ocaml
let () =
  let x =
    Nx.create Nx.int32 [| 3; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l; 9l |]
  in

  (* Get a row *)
  let row = Nx.get [ 1 ] x in           (* [4, 5, 6] *)
  Nx.print row;

  (* Get a scalar *)
  let v = Nx.item [ 1; 2 ] x in         (* 6l *)
  Printf.printf "x[1,2] = %ld\n" v
```

### Advanced slicing

The index constructors, such as `R` and `A`, are `Nx`'s, so a slice is
written inside a local open:

```ocaml
let () =
  let x =
    Nx.create Nx.int32 [| 4; 4 |]
      [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l;
         9l; 10l; 11l; 12l; 13l; 14l; 15l; 16l |]
  in

  (* Range: rows 0 to 2 (exclusive), all columns *)
  let sub = Nx.(slice [ R (0, 2); A ] x) in
  Nx.print sub;

  (* Single index on one axis, range on another *)
  let row1_cols = Nx.(slice [ I 1; R (0, 3) ] x) in
  Nx.print row1_cols;

  (* Gather specific indices *)
  let picked = Nx.(slice [ L [ 0; 3 ]; L [ 1; 2 ] ] x) in
  Nx.print picked
```

Index types: `I i` (single index), `R (start, stop)` (half-open range), `Rs (start, stop, step)` (strided range), `L indices` (gather), `T positions` (gather at positions held in an array), `M mask` (gather where a boolean mask holds), `A` (all), `N` (new axis).

## Broadcasting

Operations automatically broadcast arrays with compatible shapes. Dimensions are aligned from the right, and each pair must be equal or one must be 1:

```ocaml
let () =
  let matrix = Nx.ones Nx.float32 [| 3; 4 |] in
  let row = Nx.create Nx.float32 [| 1; 4 |] [| 10.; 20.; 30.; 40. |] in
  let result = Nx.add matrix row in    (* row added to every row *)
  Nx.print result
```

## Matrix Multiplication

```ocaml
let () =
  let a = Nx.Rng.uniform Nx.float32 [| 3; 4 |] in
  let b = Nx.Rng.uniform Nx.float32 [| 4; 2 |] in
  let c = Nx.matmul a b in
  Printf.printf "(%d×%d) × (%d×%d) = (%d×%d)\n"
    (Nx.dim 0 a) (Nx.dim 1 a) (Nx.dim 0 b) (Nx.dim 1 b) (Nx.dim 0 c) (Nx.dim 1 c)
```

## Next Steps

- [Array Operations](02-array-operations.md): reshaping, views, joining, transposing
- [Linear Algebra](03-linear-algebra.md): decompositions, solvers, FFT
- [NumPy Comparison](05-numpy-comparison.md): side-by-side reference if you're coming from Python
