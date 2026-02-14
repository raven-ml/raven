# `04-reshaping-and-broadcasting`

Change array shapes and let broadcasting align dimensions automatically. This
example reshapes a flat signal into frames, centers data by subtracting column
means, and builds an outer product — all without explicit loops.

```bash
dune exec nx/examples/04-reshaping-and-broadcasting/main.exe
```

## What You'll Learn

- Reshaping flat arrays into multi-dimensional frames with `reshape`
- Flattening back to 1D with `flatten`
- Transposing rows and columns
- Stacking arrays vertically and horizontally: `vstack`, `hstack`
- Broadcasting: how `keepdims` enables operations on different-shaped arrays
- Building outer products via broadcasting
- Adding and removing dimensions with `expand_dims` and `squeeze`

## Key Functions

| Function              | Purpose                                                |
| --------------------- | ------------------------------------------------------ |
| `reshape shape t`     | Change array shape (total elements must match)         |
| `flatten t`           | Collapse all dimensions into 1D                        |
| `transpose t`         | Reverse all axes (swap rows and columns)               |
| `vstack ts`           | Stack arrays vertically (along axis 0)                 |
| `hstack ts`           | Stack arrays horizontally (along axis 1)               |
| `expand_dims axes t`  | Insert size-1 dimensions at specified positions        |
| `squeeze t`           | Remove all size-1 dimensions                           |
| `mean ~keepdims:true` | Reduce while keeping axis as size 1 (for broadcasting) |

## Output Walkthrough

Reshape a flat 12-element signal into a 3×4 matrix of frames:

```ocaml
let signal = arange_f float64 0.0 12.0 1.0 in
let frames = reshape [| 3; 4 |] signal
```

```
Flat signal (12 samples):
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

Reshaped into 3 frames of 4:
[[0, 1, 2, 3],
 [4, 5, 6, 7],
 [8, 9, 10, 11]]
```

### Broadcasting in action

Subtracting column means from data. The `keepdims:true` parameter gives
the mean shape `[1; 3]` instead of `[3]`, which broadcasts against `[4; 3]`:

```ocaml
let col_means = mean ~axes:[ 0 ] ~keepdims:true data in
let centered = data - col_means
```

### Outer product via broadcasting

Reshape vectors into compatible shapes and multiply — no loops needed:

```ocaml
let outer = reshape [| 4; 1 |] x * reshape [| 1; 3 |] y
```

```
Outer product (x × y):
[[10, 20, 30],
 [20, 40, 60],
 [30, 60, 90],
 [40, 80, 120]]
```

## Broadcasting Rules

Two dimensions are compatible for broadcasting when they are either:
1. Equal, or
2. One of them is 1

When dimensions differ, the size-1 dimension is stretched to match. This is
why `keepdims:true` is essential for reductions used in arithmetic.

## Try It

1. Reshape the signal into `[4; 3]` instead of `[3; 4]` and compare with the
   transpose of the original frames.
2. Stack three 1D arrays of different values with `vstack`, then compute
   row-wise means using `mean ~axes:[1]`.
3. Compute an outer product of two vectors of different lengths (e.g., 5 and 3)
   using `reshape` and broadcasting.

## Next Steps

Continue to [05-reductions-and-statistics](../05-reductions-and-statistics/) to
learn how to summarize data with aggregations along any axis.
