# Linear Algebra

Nx provides linear algebra over batches of matrices and Fourier transforms. This guide covers the most commonly used operations.

A function whose result is real while its operand may be complex, such as singular values, the eigenvalues of a Hermitian matrix, a norm or a log-determinant, takes that result's dtype first: `Nx.svd Nx.float64 a`. So does a function whose result is complex while its operand may be real, such as `Nx.eig` or `Nx.rfft`.

## Matrix Multiplication

### matmul

General matrix multiplication supporting batched inputs:

```ocaml
let () =
  let a = Nx.Rng.uniform Nx.float32 [| 3; 4 |] in
  let b = Nx.Rng.uniform Nx.float32 [| 4; 2 |] in
  let c = Nx.matmul a b in   (* [|3; 2|] *)
  Printf.printf "result shape: [|%d; %d|]\n" (Nx.dim 0 c) (Nx.dim 1 c)
```

`matmul` supports batched matrix multiplication: leading dimensions are broadcast.

```ocaml
let () =
  (* Batched: [|batch; m; k|] × [|batch; k; n|] → [|batch; m; n|] *)
  let a = Nx.Rng.uniform Nx.float32 [| 10; 3; 4 |] in
  let b = Nx.Rng.uniform Nx.float32 [| 10; 4; 2 |] in
  let c = Nx.matmul a b in   (* [|10; 3; 2|] *)
  ignore c
```

Inside a local open, `Nx.(a *@ b)` is `Nx.matmul a b`.

### Related products

| Function | Purpose |
|----------|---------|
| `dot` | Sum over the last axis of one operand and an axis of the other |
| `vdot` | Complex-conjugate inner product of flattened operands |
| `inner` | Inner product over last axes |
| `outer` | Outer product of flattened operands |
| `tensordot` | Contraction over specified axes |
| `einsum` | Products and sums named by an axis pattern |
| `kron` | Kronecker product |
| `cross` | Cross product of 3-element vectors |
| `multi_dot` | Chain of matrix products, in the cheapest order |

### einsum

A pattern names each operand's axes. Names are words separated by spaces. A name in the result keeps its axis, and the names after `|` are summed; each summed name appears in two operands or more and not in the result:

<!-- $MDX skip -->
```ocaml
(* Matrix multiplication: j is summed *)
let c = Nx.einsum (Nx.Pattern.v "i j, j k -> i k | j") [ a; b ]

(* Batch matrix multiply: b is in both operands and the result *)
let c = Nx.einsum (Nx.Pattern.v "b i j, b j k -> b i k | j") [ a; b ]

(* Matrix times matrix times vector, in the order of fewest products *)
let y = Nx.einsum (Nx.Pattern.v "i j, j k, k -> i | j k") [ a; b; v ]
```

A name appears once in each operand, so a diagonal is read with `Nx.trace m` or `Nx.diagonal m`. A pattern of one operand rearranges it, with `Nx.rearrange`:

<!-- $MDX skip -->
```ocaml
(* Transpose *)
let t = Nx.rearrange (Nx.Pattern.v "i j -> j i") m
```

`Nx.Pattern.v` parses a pattern each time it is called, so a pattern used in a loop is bound once, outside it.

## Decompositions

### Cholesky

Factor a symmetric positive-definite matrix: A = L·Lᵀ

<!-- $MDX skip -->
```ocaml
let l = Nx.cholesky a                        (* lower triangular by default *)
let u = Nx.cholesky ~triangle:Nx.Upper a        (* upper triangular *)
```

### QR

Factor A = Q·R where Q is orthogonal and R is upper triangular:

<!-- $MDX skip -->
```ocaml
let q, r = Nx.qr a                           (* reduced by default *)
let q_full, r_full = Nx.qr ~factors:Nx.Complete a
```

### LU

Factor A with partial pivoting into a row order `perm`, L lower triangular
with a unit diagonal and U upper triangular, so that the rows of A taken in
the order `perm` are L·U. `det`, `slogdet`, `solve` and `inv` are computed from
this factorization:

<!-- $MDX skip -->
```ocaml
let perm, l, u = Nx.lu a
```

### SVD

Singular value decomposition A = U·Σ·Vᴴ, the singular values in the dtype given first:

<!-- $MDX skip -->
```ocaml
let u, s, vh = Nx.svd Nx.float64 a                  (* reduced factors *)
let _, s_only, _ = Nx.svd ~vectors:false Nx.float64 a  (* singular values only *)
```

### Eigendecomposition

<!-- $MDX skip -->
```ocaml
(* General: complex eigenvalues and eigenvectors *)
let eigenvalues, eigenvectors = Nx.eig Nx.complex128 a
let eigenvalues_only, _ = Nx.eig ~vectors:false Nx.complex128 a

(* Symmetric/Hermitian: real eigenvalues, ascending *)
let eigenvalues, eigenvectors = Nx.eigh Nx.float64 a
let eigenvalues_only, _ = Nx.eigh ~vectors:false Nx.float64 a
```

## Solving Linear Systems

### solve

Solve A·x = b for x:

<!-- $MDX skip -->
```ocaml
let x = Nx.solve a b
```

### solve_triangular

Solve a triangular system directly, skipping the factorization. Only the
triangle named by `triangle` is read:

<!-- $MDX skip -->
```ocaml
let x = Nx.solve_triangular ~triangle:Nx.Upper r b
```

### lstsq

The least-squares solution of least norm, for a matrix of any shape and rank,
with the residuals, the rank and the singular values:

<!-- $MDX skip -->
```ocaml
let x, residuals, rank, sv = Nx.lstsq Nx.float64 a b
```

### inv and pinv

Matrix inverse and pseudo-inverse:

<!-- $MDX skip -->
```ocaml
let a_inv = Nx.inv a          (* square; NaN for a singular matrix *)
let a_pinv = Nx.pinv a        (* works for any shape *)
```

## When a Matrix Fails

No linear-algebra function raises on the values of a matrix. A matrix on which a function is undefined, such as one that is not positive-definite given to `cholesky`, one with a zero pivot given to `solve` or `inv`, or one holding NaN or an infinity given to `svd`, `eig` or `eigh`, has NaN in every element of its results. Each matrix of a batch is computed on its own, so one failing matrix leaves the others unchanged, and any one element of a matrix's result tells whether it failed:

<!-- $MDX skip -->
```ocaml
let l = Nx.cholesky covariances in                       (* [|batch; n; n|] *)
let ok = Nx.all ~axes:[ -2; -1 ] (Nx.isfinite l) in     (* [|batch|] *)
```

A caller that wants an exception checks, naming the matrix that failed:

<!-- $MDX skip -->
```ocaml
Nx.check Nx.Ptree.unit ok () (fun i () ->
    Failure (Printf.sprintf "covariance %d is not positive-definite" i.(0)))
```

Outside a function's stated precondition, NaN or an infinity in an operand propagates through the arithmetic to some elements of the results.

## Norms and Properties

### Norms

`vector_norm` takes the norm of the vectors along some axes, all of them by default, and `matrix_norm` that of each matrix of a batch. Both take the result's dtype first:

<!-- $MDX skip -->
```ocaml
(* Vector norms *)
let l2 = Nx.vector_norm Nx.float64 v                    (* L2 by default *)
let l1 = Nx.vector_norm Nx.float64 ~ord:1. v            (* L1 norm *)
let linf = Nx.vector_norm Nx.float64 ~ord:infinity v    (* max absolute value *)

(* Matrix norms *)
let fro = Nx.matrix_norm Nx.float64 m                   (* Frobenius by default *)
let spectral = Nx.matrix_norm Nx.float64 ~ord:`Two m    (* largest singular value *)

(* Along specific axes *)
let row_norms = Nx.vector_norm Nx.float64 ~axes:[ 1 ] m (* per-row L2 norm *)
```

### Other properties

<!-- $MDX skip -->
```ocaml
let d = Nx.det m                              (* determinant *)
let sign, logdet = Nx.slogdet Nx.float64 m    (* sign, and log |det| in float64 *)
let tr = Nx.trace m                           (* sum of diagonal elements *)
let r = Nx.matrix_rank m                      (* numerical rank, int64 *)
let c = Nx.cond Nx.float64 m                  (* condition number *)
let diag = Nx.diagonal m                      (* extract diagonal *)
```

## FFT

Nx provides complex, real and Hermitian discrete Fourier transforms, and the cosine and sine transforms. A transform runs along `~axes`, the last axis by default, so a transform along several axes is the same function with more axes.

### Basic FFT

<!-- $MDX skip -->
```ocaml
(* 1-D complex FFT and inverse *)
let spectrum = Nx.fft x
let reconstructed = Nx.ifft spectrum

(* 2-D FFT, along the last two axes *)
let spectrum_2d = Nx.fft ~axes:[ -2; -1 ] image

(* N-D FFT *)
let spectrum_nd = Nx.fft ~axes:[ 0; 1; 2 ] volume
```

### Real FFT

For real-valued inputs, `rfft` keeps only the bins from 0 to n/2 of the last
axis, since the others are their conjugates:

<!-- $MDX skip -->
```ocaml
(* The first argument picks the output dtype, independent of the input's.
   Pair float32 with complex64 to keep a single-precision spectrum. *)
let spectrum = Nx.rfft Nx.complex64 signal      (* n/2+1 complex outputs *)
let signal_back = Nx.irfft Nx.float32 spectrum  (* back to real *)

let spectrum_2d = Nx.rfft Nx.complex64 ~axes:[ -2; -1 ] image
let spectrum_nd = Nx.rfft Nx.complex64 ~axes:[ 0; 1 ] data
```

Without `~s`, `irfft` of m bins gives 2 (m - 1) points; `~s:[ n ]` asks for
the n points of an odd-length signal.

### Frequency axes

Frequency axes are constructed, so they take their dtype first. Ask for the same precision you transformed at and the frequency bins line up with the spectrum without a cast:

<!-- $MDX skip -->
```ocaml
let freqs = Nx.fftfreq Nx.float32 n    (* frequency bins for fft *)
let rfreqs = Nx.rfftfreq Nx.float32 n  (* frequency bins for rfft *)
let shifted = Nx.fftshift spectrum     (* shift zero-frequency to center *)
```

### Magnitude and phase

A spectrum is complex. `magnitude` and `angle` extract its polar components as
float arrays, and `complex` reassembles one. The first argument picks the
result dtype, since leaving the complex domain has to name what to land in;
this is why `magnitude` is not called `abs`, which preserves the dtype:

<!-- $MDX skip -->
```ocaml
let mags = Nx.magnitude Nx.float64 spectrum  (* |z| per bin *)
let phase = Nx.angle Nx.float64 spectrum     (* arg z, in [-π, π] *)

(* rebuild a spectrum after reshaping its magnitude *)
let rebuilt =
  Nx.complex Nx.complex128
    ~re:Nx.(mul mags (cos phase))
    ~im:Nx.(mul mags (sin phase))
```

`real`, `imag`, and `conjugate` give the cartesian view of the same array.

## Next Steps

- [Array Operations](02-array-operations.md): reshaping, broadcasting, slicing
- [Input/Output](04-io.md): reading and writing files
- [NumPy Comparison](05-numpy-comparison.md): side-by-side reference
