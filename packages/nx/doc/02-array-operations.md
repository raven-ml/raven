# Array Operations

This guide covers reshaping, broadcasting, joining, slicing, and the views that let Nx rearrange arrays without copying them.

The examples name every function with its module. The index constructors, such as `R` and `A`, are `Nx`'s too, so a slice is written inside a local open, `Nx.(slice [ R (0, 2); A ] x)`.

## Views and Copies

A **movement** rearranges an array's elements without computing on them. Many movements return a **view**: an array over its operand's memory, with another shape, strides or offset, made in constant time without copying any element. An array never changes, so a view and a copy read the same elements. Whether a movement makes a view is a cost, and never changes a result.

Views: `transpose`, `squeeze`, `unsqueeze`, `flip`, `moveaxis`, `swapaxes`, `broadcast_to`, `sliding_window`, `split`, `get`, and `slice` with `I`, `R`, `Rs`, `A` and `N`. `reshape`, `flatten`, `tile` and `repeat` make a view where strides express the result, and a copy otherwise.

Copies: `copy`, `concatenate`, `stack`, `pad`, `roll`, `slice` with `L`, `T`, `D` and `M`, and element-wise operations. `copy` stores an array afresh, in memory of its own.

## Reshaping

### reshape

Change the shape without changing the data order. The total number of elements must match. Use `-1` to infer one dimension:

```ocaml
let () =
  let x = Nx.create Nx.int32 [| 6 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
  let a = Nx.reshape [| 2; 3 |] x in
  let b = Nx.reshape [| 3; -1 |] x in   (* -1 inferred as 2 *)
  Nx.print a;
  Nx.print b
```

### flatten

`flatten` merges a run of axes into one. `reshape` splits an axis back, given
the shape it should have:

```ocaml
let () =
  let x = Nx.zeros Nx.float32 [| 2; 3; 4 |] in
  ignore (Nx.flatten x |> Nx.shape);                  (* [|24|] *)
  ignore (Nx.flatten ~start_dim:1 x |> Nx.shape);     (* [|2; 12|] *)

  let y = Nx.zeros Nx.float32 [| 2; 12 |] in
  ignore (Nx.reshape [| 2; 3; 4 |] y |> Nx.shape)     (* [|2; 3; 4|] *)
```

A pattern names the axes instead. A group on its left side splits an axis
into the axes it names, and `~sizes` gives the extents the shape leaves
unknown:

```ocaml
let () =
  let y = Nx.zeros Nx.float32 [| 2; 12 |] in
  let split = Nx.Pattern.v "b (h w) -> b h w" in
  ignore (Nx.rearrange ~sizes:[ ("h", 3) ] split y |> Nx.shape)  (* [|2; 3; 4|] *)
```

### squeeze and unsqueeze

Remove or add dimensions of size 1:

```ocaml
let () =
  let x = Nx.ones Nx.float32 [| 1; 3; 1; 4 |] in
  let a = Nx.squeeze x in                    (* [|3; 4|] *)
  let b = Nx.squeeze ~axes:[ 0 ] x in        (* [|3; 1; 4|] *)
  Printf.printf "squeeze all: %dx%d\n" (Nx.dim 0 a) (Nx.dim 1 a);
  Printf.printf "squeeze [0]: %dx%dx%d\n" (Nx.dim 0 b) (Nx.dim 1 b) (Nx.dim 2 b);

  let y = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let c = Nx.unsqueeze ~axes:[ 0; 2 ] y in   (* [|1; 3; 1|] *)
  Printf.printf "unsqueeze: %dx%dx%d\n" (Nx.dim 0 c) (Nx.dim 1 c) (Nx.dim 2 c)
```

## Broadcasting

Binary operations automatically broadcast operands. Dimensions are aligned from the right, and each pair must be equal or one must be 1:

```ocaml
let () =
  (* Add a row vector to every row of a matrix *)
  let matrix = Nx.ones Nx.float32 [| 3; 4 |] in
  let row = Nx.create Nx.float32 [| 1; 4 |] [| 10.; 20.; 30.; 40. |] in
  let result = Nx.add matrix row in
  Nx.print result;

  (* Add a column vector to every column *)
  let col = Nx.create Nx.float32 [| 3; 1 |] [| 100.; 200.; 300. |] in
  let result2 = Nx.add matrix col in
  Nx.print result2
```

You can also broadcast explicitly:

```ocaml
let x =
  Nx.broadcast_to [| 3; 3 |]
    (Nx.create Nx.float32 [| 1; 3 |] [| 1.; 2.; 3. |])
(* Repeats the row 3 times without copying data *)
```

### Broadcasting rules

Shapes are compatible when, aligned from the right, every dimension pair is either equal or one of them is 1, and missing leading dimensions count as 1. At each position the result takes the extent that is not 1, or 1.

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
let () =
  let x = Nx.create Nx.int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
  let t = Nx.transpose x in
  Nx.print t
  (* int32 [3; 2] [[1, 4], [2, 5], [3, 6]] *)
```

Specify a permutation for higher-rank arrays:

```ocaml
(* Permute [batch; height; width; channels] to [batch; channels; height; width] *)
let nhwc_to_nchw x = Nx.transpose ~axes:[0; 3; 1; 2] x
```

### moveaxis and swapaxes

Move or swap individual dimensions:

<!-- $MDX skip -->
```ocaml
let moved = Nx.moveaxis 0 2 x      (* move axis 0 to position 2 *)
let swapped = Nx.swapaxes 1 2 x    (* swap axes 1 and 2 *)
```

### flip

Reverse elements along axes:

<!-- $MDX skip -->
```ocaml
let mirrored = Nx.flip ~axes:[1] x    (* mirror columns *)
let reversed = Nx.flip x              (* reverse all dimensions *)
```

## Indexing and Slicing

### get

Index from the outermost dimension inward. Returns a sub-array (view):

```ocaml
let () =
  let x = Nx.create Nx.int32 [| 2; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l |] in
  let row = Nx.get [ 1 ] x in      (* second row: [4, 5, 6] *)
  Nx.print row
```

`get [i; j] x` is `slice [I i; I j] x`.

### item

Extract a scalar value:

<!-- $MDX skip -->
```ocaml
let v = Nx.item [1; 2] matrix    (* element at row 1, column 2 *)
```

### slice

An index selects positions along one axis and replaces the axis by the
index's shape; `N` adds an axis of extent 1, and a mask `M` covers as many
axes as it has. `I i` is one position, so its axis goes:

```ocaml
let () =
  let x =
    Nx.create Nx.int32 [| 3; 3 |] [| 1l; 2l; 3l; 4l; 5l; 6l; 7l; 8l; 9l |]
  in

  (* R (start, stop): half-open range *)
  let rows_0_1 = Nx.(slice [ R (0, 2); A ] x) in
  Nx.print rows_0_1;

  (* I i: single index (reduces dimension) *)
  let col_1 = Nx.(slice [ A; I 1 ] x) in
  Nx.print col_1;

  (* L [indices]: gather specific indices *)
  let corners = Nx.(slice [ L [ 0; 2 ]; L [ 0; 2 ] ] x) in
  Nx.print corners
```

Index types:
- `I i`: one position (removes the axis)
- `L positions`: listed positions
- `T p`: positions held in the int64 array `p`, whose shape replaces the axis
- `R (start, stop)`: half-open range
- `Rs (start, stop, step)`: strided range
- `A`: all elements (default for trailing axes)
- `M mask`: positions where a boolean mask holds
- `D (start, len)`: `len` positions from a start held in a 0-d array
- `N`: insert new axis of size 1

A position written in the program (`I`, `L`) raises outside its axis, a range
is cut to it, and a position held in an array (`T`) reads zero outside it.

## Joining and Splitting

### concatenate

Join arrays along an existing axis:

```ocaml
let () =
  let a = Nx.ones Nx.float32 [| 2; 3 |] in
  let b = Nx.zeros Nx.float32 [| 2; 3 |] in
  let c = Nx.concatenate ~axis:0 [ a; b ] in   (* [|4; 3|] *)
  Printf.printf "concat axis 0: %dx%d\n" (Nx.dim 0 c) (Nx.dim 1 c);
  let d = Nx.concatenate ~axis:1 [ a; b ] in   (* [|2; 6|] *)
  Printf.printf "concat axis 1: %dx%d\n" (Nx.dim 0 d) (Nx.dim 1 d)
```

### stack

Join arrays along a **new** axis:

```ocaml
let () =
  let a = Nx.create Nx.float32 [| 3 |] [| 1.; 2.; 3. |] in
  let b = Nx.create Nx.float32 [| 3 |] [| 4.; 5.; 6. |] in
  let c = Nx.stack ~axis:0 [ a; b ] in   (* [|2; 3|] *)
  Nx.print c
```

### split

Split an array into runs of consecutive elements along an axis, as equal as
its extent allows:

<!-- $MDX skip -->
```ocaml
let parts = Nx.split ~axis:0 2 x    (* split into 2 along axis 0 *)
```

## Tiling and Repeating

### tile

Replicate the array according to a repeat pattern:

<!-- $MDX skip -->
```ocaml
(* Tile a [2; 3] array 2x along rows, 3x along columns → [4; 9] *)
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

- [Linear Algebra](03-linear-algebra.md): matrix operations, decompositions, FFT
- [Input/Output](04-io.md): reading and writing images, npy, npz files
- [NumPy Comparison](05-numpy-comparison.md): side-by-side reference
