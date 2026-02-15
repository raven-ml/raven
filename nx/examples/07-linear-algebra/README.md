# `07-linear-algebra`

Solve systems, decompose matrices, and fit models — linear algebra made
practical. This example covers matrix multiplication, linear solves, least
squares fitting, eigendecomposition, and SVD.

```bash
dune exec nx/examples/07-linear-algebra/main.exe
```

## What You'll Learn

- Matrix multiplication with `@@` and dot products with `<.>`
- Solving linear systems with `/@`
- Computing inverses, determinants, and norms
- Fitting a line to data with least squares (`lstsq`)
- Eigendecomposition of symmetric matrices (`eigh`)
- Singular value decomposition and reconstruction (`svd`)

## Key Functions

| Function    | Purpose                                            |
| ----------- | -------------------------------------------------- |
| `a @@ b`    | Matrix multiplication                              |
| `u <.> v`   | Vector dot product                                 |
| `a /@ b`    | Solve linear system Ax = b                         |
| `inv m`     | Matrix inverse                                     |
| `det m`     | Determinant                                        |
| `norm m`    | Matrix norm (Frobenius by default)                 |
| `lstsq a b` | Least squares solution to overdetermined system    |
| `eigh m`    | Eigenvalues and eigenvectors of a symmetric matrix |
| `svd m`     | Singular value decomposition (U, S, Vt)            |
| `diag v`    | Create diagonal matrix from a vector               |

## Output Walkthrough

### Matrix multiplication

```ocaml
let a = create float64 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
let b = create float64 [| 3; 2 |] [| 7.; 8.; 9.; 10.; 11.; 12. |] in
a @@ b    (* [2; 2] result *)
```

### Solving linear systems

The `/@` operator solves Ax = b for x:

```ocaml
let x = coeff /@ rhs    (* x = [2; 3; -1] *)
```

### Inverse verification

```ocaml
let m_inv = inv m in
m @@ m_inv    (* ≈ identity matrix *)
```

### Least squares fitting

Build a design matrix [x, 1] and solve for slope and intercept:

```ocaml
let design = hstack [ x_col; ones float64 [| 6; 1 |] ] in
let coeffs, _, _, _ = lstsq design y_col in
(* m ≈ 1.97, c ≈ 1.03 *)
```

### SVD decomposition and reconstruction

```ocaml
let u_mat, s_vec, vt = svd data in
let reconstructed = u_mat.${[ A; R (0, 2) ]} @@ diag s_vec @@ vt
(* reconstructed ≈ original *)
```

## Try It

1. Solve a different 3×3 system and verify the solution by computing
   `coeff @@ x` — it should match the right-hand side.
2. Extend the least squares example to fit a quadratic by adding an x^2
   column to the design matrix.
3. Use SVD for low-rank approximation: zero out the smallest singular value,
   reconstruct, and compare to the original.

## Next Steps

Continue to [08-signal-processing](../08-signal-processing/) to apply
frequency analysis with FFT.
