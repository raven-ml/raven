# `07-linear-algebra`

Solve systems, decompose matrices, and fit models: linear algebra made
practical. This example covers matrix multiplication, linear solves, least
squares fitting, eigendecomposition, and SVD.

```bash
dune exec packages/nx/examples/07-linear-algebra/main.exe
```

## What You'll Learn

- Matrix multiplication with `*@` and dot products with `dot`
- Solving linear systems with `solve`
- Computing inverses, determinants, and norms
- Fitting a line to data with least squares (`lstsq`)
- Eigendecomposition of symmetric matrices (`eigh`)
- Singular value decomposition and reconstruction (`svd`)

## Key Functions

A function whose result is real, such as a norm, eigenvalues of a symmetric
matrix or singular values, takes that result's dtype first.

| Function                 | Purpose                                            |
| ------------------------ | -------------------------------------------------- |
| `a *@ b`                 | Matrix multiplication                              |
| `dot u v`                | Vector dot product                                 |
| `solve a b`              | Solve the linear system Ax = b                     |
| `inv m`                  | Matrix inverse                                     |
| `det m`                  | Determinant                                        |
| `matrix_norm dtype m`    | Matrix norm (Frobenius by default)                 |
| `lstsq dtype a b`        | Least squares solution to overdetermined system    |
| `eigh dtype m`           | Eigenvalues and eigenvectors of a symmetric matrix |
| `svd dtype m`            | Singular value decomposition (U, S, Vt)            |
| `diag v`                 | Create diagonal matrix from a vector               |

## Output Walkthrough

### Matrix multiplication

```ocaml
let a = Nx.create Nx.float64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
let b = Nx.create Nx.float64 [| 3; 2 |] [| 7.; 8.; 9.; 10.; 11.; 12. |] in
Nx.(a *@ b)    (* [2; 2] result *)
```

### Solving linear systems

`solve` finds x with Ax = b:

```ocaml
let x = Nx.solve coeff rhs    (* x ≈ [2; 3; -1] *)
```

```
x: float64 [3] [2.0000000000000004, 2.9999999999999996, -0.9999999999999991]
```

### Inverse verification

```ocaml
let m_inv = Nx.inv m in
Nx.(m *@ m_inv)    (* ≈ identity matrix *)
```

### Least squares fitting

Build a design matrix [x, 1] and solve for slope and intercept:

```ocaml
let design =
  Nx.concatenate ~axis:1 [ x_col; Nx.ones Nx.float64 [| 6; 1 |] ]
in
let coeffs, _, _, _ = Nx.lstsq Nx.float64 design y_col in
(* m ≈ 1.96, c ≈ 1.08 *)
```

### SVD decomposition and reconstruction

`svd` gives the reduced factors by default: for a 3×2 matrix, U is 3×2, S
holds 2 singular values and Vt is 2×2.

```ocaml
let u_mat, s_vec, vt = Nx.svd Nx.float64 data in
let reconstructed = Nx.(u_mat *@ diag s_vec *@ vt)
(* reconstructed ≈ original *)
```

## Try It

1. Solve a different 3×3 system and verify the solution by computing
   `Nx.(coeff *@ x)`: it should match the right-hand side.
2. Extend the least squares example to fit a quadratic by adding an x^2
   column to the design matrix.
3. Use SVD for low-rank approximation: zero out the smallest singular value,
   reconstruct, and compare to the original.

## Next Steps

Continue to [08-signal-processing](../08-signal-processing/) to apply
frequency analysis with FFT.
