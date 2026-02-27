# Array Operations

This guide covers reshaping, broadcasting, joining, slicing, and the view model that underlies Nx's efficiency.

## Views and Copies

Many Nx operations return **views** — tensors that share the underlying buffer with the original but have different shape, strides, or offset. Views are O(1) and allocate no new data.

View-producing operations: `reshape`, `transpose`, `slice`, `squeeze`, `unsqueeze`, `flip`, `get`, `moveaxis`, `swapaxes`.

Copy-producing operations: `contiguous`, `copy`, `concatenate`, `stack`, `pad`, element-wise operations.

Use `is_c_contiguous` to check whether elements are laid out contiguously in row-major order, and `contiguous` to force a copy when needed:

<!-- $MDX skip -->
```ocaml
let t = Nx.transpose x in
Nx.is_c_contiguous t       (* often false *)
let t' = Nx.contiguous t   (* force a contiguous copy *)
```

## Reshaping

### reshape

Change the shape without changing the data order. The total number of elements must match. Use `-1` to infer one dimension:

```ocaml
open Nx

let () =
  let x = create Int32 [|6|] [|1l; 2l; 3l; 4l; 5l; 6l|] in
  let a = reshape [|2; 3|] x in
  let b = reshape [|3; -1|] x in   (* -1 inferred as 2 *)
  print_data a;
  print_data b
```

### flatten and unflatten

`flatten` collapses dimensions into one. `unflatten` expands a dimension back:

<!-- $MDX skip -->
```ocaml
let x = Nx.zeros Nx.Float32 [|2; 3; 4|] in
Nx.flatten x |> Nx.shape                     (* [|24|] *)
Nx.flatten ~start_dim:1 x |> Nx.shape        (* [|2; 12|] *)

let y = Nx.zeros Nx.Float32 [|2; 12|] in
Nx.unflatten 1 [|3; 4|] y |> Nx.shape        (* [|2; 3; 4|] *)
```

### squeeze and unsqueeze

Remove or add dimensions of size 1:

```ocaml
open Nx

let () =
  let x = ones Float32 [|1; 3; 1; 4|] in
  let a = squeeze x in                    (* [|3; 4|] *)
  let b = squeeze ~axes:[0] x in          (* [|3; 1; 4|] *)
  Printf.printf "squeeze all: %dx%d\n" (dim 0 a) (dim 1 a);
  Printf.printf "squeeze [0]: %dx%dx%d\n" (dim 0 b) (dim 1 b) (dim 2 b);

  let y = create Float32 [|3|] [|1.; 2.; 3.|] in
  let c = unsqueeze ~axes:[0; 2] y in     (* [|1; 3; 1|] *)
  Printf.printf "unsqueeze: %dx%dx%d\n" (dim 0 c) (dim 1 c) (dim 2 c)
```

## Broadcasting

Binary operations automatically broadcast operands. Dimensions are aligned from the right, and each pair must be equal or one must be 1:

```ocaml
open Nx

let () =
  (* Add a row vector to every row of a matrix *)
  let matrix = ones Float32 [|3; 4|] in
  let row = create Float32 [|1; 4|] [|10.; 20.; 30.; 40.|] in
  let result = add matrix row in
  print_data result;

  (* Add a column vector to every column *)
  let col = create Float32 [|3; 1|] [|100.; 200.; 300.|] in
  let result2 = add matrix col in
  print_data result2
```

You can also broadcast explicitly:

<!-- $MDX skip -->
```ocaml
let x = Nx.broadcast_to [|3; 3|] (Nx.create Nx.Float32 [|1; 3|] [|1.; 2.; 3.|])
(* Repeats the row 3 times without copying data *)
```

### Broadcasting rules

Shapes are compatible when, aligned from the right, every dimension pair is either equal or one of them is 1. The result shape takes the maximum at each position.

```
[|   3; 4|]  +  [|1; 4|]  →  [|3; 4|]   ✓
[|2; 3; 4|]  +  [|   4|]  →  [|2; 3; 4|] ✓
[|   3; 4|]  +  [|3; 1|]  →  [|3; 4|]   ✓
[|      3|]  +  [|   4|]  →  error        ✗
```

## Transposing and Permuting

### transpose

Reverse dimensions (no copy):

```ocaml
open Nx

let () =
  let x = create Int32 [|2; 3|] [|1l; 2l; 3l; 4l; 5l; 6l|] in
  let t = transpose x in
  print_data t
  (* [[1, 4],
      [2, 5],
      [3, 6]] *)
```

Specify a permutation for higher-rank tensors:

<!-- $MDX skip -->
```ocaml
(* Permute [batch; height; width; channels] to [batch; channels; height; width] *)
let nhwc_to_nchw x = Nx.transpose ~axes:[0; 3; 1; 2] x
```

### moveaxis and swapaxes

Move or swap individual dimensions:

<!-- $MDX skip -->
```ocaml
Nx.moveaxis 0 2 x    (* move axis 0 to position 2 *)
Nx.swapaxes 1 2 x    (* swap axes 1 and 2 *)
```

### flip

Reverse elements along axes:

<!-- $MDX skip -->
```ocaml
Nx.flip ~axes:[1] x     (* mirror columns *)
Nx.flip x                (* reverse all dimensions *)
```

## Indexing and Slicing

### get

Index from the outermost dimension inward. Returns a sub-tensor (view):

```ocaml
open Nx

let () =
  let x = create Int32 [|2; 3|] [|1l; 2l; 3l; 4l; 5l; 6l|] in
  let row = get [1] x in      (* second row: [4, 5, 6] *)
  print_data row
```

### item

Extract a scalar value:

<!-- $MDX skip -->
```ocaml
let v = Nx.item [1; 2] matrix    (* element at row 1, column 2 *)
```

### slice

Advanced indexing with range and index specifications:

```ocaml
open Nx

let () =
  let x = create Int32 [|3; 3|] [|1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l; 9l|] in

  (* R (start, stop): half-open range *)
  let rows_0_1 = slice [R (0, 2); A] x in
  print_data rows_0_1;

  (* I i: single index (reduces dimension) *)
  let col_1 = slice [A; I 1] x in
  print_data col_1;

  (* L [indices]: gather specific indices *)
  let corners = slice [L [0; 2]; L [0; 2]] x in
  print_data corners
```

Index types:
- `I i` — single index (reduces dimension)
- `R (start, stop)` — half-open range
- `Rs (start, stop, step)` — strided range
- `L indices` — gather listed indices
- `A` — all elements (default for trailing axes)
- `N` — insert new axis of size 1

## Joining and Splitting

### concatenate

Join tensors along an existing axis:

```ocaml
open Nx

let () =
  let a = ones Float32 [|2; 3|] in
  let b = zeros Float32 [|2; 3|] in
  let c = concatenate ~axis:0 [a; b] in   (* [|4; 3|] *)
  Printf.printf "concat axis 0: %dx%d\n" (dim 0 c) (dim 1 c);
  let d = concatenate ~axis:1 [a; b] in   (* [|2; 6|] *)
  Printf.printf "concat axis 1: %dx%d\n" (dim 0 d) (dim 1 d)
```

Shorthands: `vstack` (axis 0), `hstack` (axis 1), `dstack` (axis 2).

### stack

Join tensors along a **new** axis:

```ocaml
open Nx

let () =
  let a = create Float32 [|3|] [|1.; 2.; 3.|] in
  let b = create Float32 [|3|] [|4.; 5.; 6.|] in
  let c = stack ~axis:0 [a; b] in   (* [|2; 3|] *)
  print_data c
```

### split

Split a tensor into equal parts along an axis:

<!-- $MDX skip -->
```ocaml
let parts = Nx.split ~axis:0 2 x    (* split into 2 along axis 0 *)
```

## Tiling and Repeating

### tile

Replicate the tensor according to a repeat pattern:

<!-- $MDX skip -->
```ocaml
(* Tile a [2; 3] tensor 2x along rows, 3x along columns → [4; 9] *)
Nx.tile [|2; 3|] x
```

### repeat

Repeat elements along a single axis:

<!-- $MDX skip -->
```ocaml
(* Repeat each element 3 times along axis 0 *)
Nx.repeat ~axis:0 3 x
```

### pad

Pad with a constant value:

<!-- $MDX skip -->
```ocaml
(* Pad: 1 before and 2 after along axis 0, 0 and 1 along axis 1 *)
Nx.pad [|(1, 2); (0, 1)|] 0. x
```

## Next Steps

- [Linear Algebra](/docs/nx/linear-algebra/) — matrix operations, decompositions, FFT
- [Input/Output](/docs/nx/io/) — reading and writing images, npy, npz files
- [NumPy Comparison](/docs/nx/numpy-comparison/) — side-by-side reference
