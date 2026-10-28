# `04-reshaping-and-broadcasting`

Change array shapes and let broadcasting align dimensions automatically. This
example reshapes a flat signal into frames, centers data by subtracting column
means, and builds an outer product, all without explicit loops.

```bash
dune exec packages/nx/examples/04-reshaping-and-broadcasting/main.exe
```

## What You'll Learn

- Reshaping flat arrays into multi-dimensional frames with `reshape`
- Flattening back to 1D with `flatten`
- Transposing rows and columns
- Joining arrays with `stack` (new axis) and `concatenate` (existing axis)
- Broadcasting: how `keepdims` enables operations on different-shaped arrays
- Building outer products via broadcasting
- Adding and removing dimensions with `unsqueeze` and `squeeze`

## Key Functions

| Function              | Purpose                                                |
| --------------------- | ------------------------------------------------------ |
| `reshape shape t`     | Change array shape (total elements must match)         |
| `flatten t`           | Collapse all dimensions into 1D                        |
| `transpose t`         | Reverse all axes (swap rows and columns)               |
| `stack ~axis ts`      | Join arrays along a new axis                           |
| `concatenate ~axis ts`| Join arrays along an existing axis                     |
| `unsqueeze ~axes t`   | Insert size-1 dimensions at specified positions        |
| `squeeze t`           | Remove all size-1 dimensions                           |
| `mean ~keepdims:true` | Reduce while keeping axis as size 1 (for broadcasting) |

## Output Walkthrough

Reshape a flat 12-element signal into a 3×4 matrix of frames:

```ocaml
let signal = Nx.arange_f Nx.float64 0.0 12.0 1.0 in
let frames = Nx.reshape [| 3; 4 |] signal
```

```
Flat signal (12 samples):
float64 [12] [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

Reshaped into 3 frames of 4:
float64 [3; 4] [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]]
```

### Broadcasting in action

Subtracting column means from data. The `keepdims:true` parameter gives
the mean shape `[1; 3]` instead of `[3]`, which broadcasts against `[4; 3]`:

```ocaml
let col_means = Nx.mean ~axes:[ 0 ] ~keepdims:true data in
let centered = Nx.(data - col_means)
```

### Outer product via broadcasting

Reshape vectors into compatible shapes and multiply, with no loop:

```ocaml
let outer = Nx.(reshape [| 4; 1 |] x * reshape [| 1; 3 |] y)
```

```
Outer product (x × y):
float64 [4; 3] [[10, 20, 30], [20, 40, 60], [30, 60, 90], [40, 80, 120]]
```

## Broadcasting Rules

Shapes are aligned at their last axes, and missing leading axes count as 1.
Two aligned extents are compatible when they are equal or one of them is 1,
and an extent of 1 stretches to match the other. A reduction with
`keepdims:true` keeps its reduced axes as extents of 1, so its result lines up
with the array it came from.

## Try It

1. Reshape the signal into `[4; 3]` instead of `[3; 4]` and compare with the
   transpose of the original frames.
2. Stack three 1D arrays of different values with `stack`, then compute
   row-wise means using `mean ~axes:[1]`.
3. Compute an outer product of two vectors of different lengths (e.g., 5 and 3)
   using `reshape` and broadcasting.

## Next Steps

Continue to [05-reductions-and-statistics](../05-reductions-and-statistics/) to
learn how to summarize data with aggregations along any axis.
