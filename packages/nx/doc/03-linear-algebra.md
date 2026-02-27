# Linear Algebra

Nx provides a comprehensive linear algebra suite and FFT operations. This guide covers the most commonly used operations.

## Matrix Multiplication

### matmul

General matrix multiplication supporting batched inputs:

```ocaml
open Nx

let () =
  let a = rand Float32 [|3; 4|] in
  let b = rand Float32 [|4; 2|] in
  let c = matmul a b in   (* [|3; 2|] *)
  Printf.printf "result shape: [|%d; %d|]\n" (dim 0 c) (dim 1 c)
```

`matmul` supports batched matrix multiplication: leading dimensions are broadcast.

<!-- $MDX skip -->
```ocaml
(* Batched: [|batch; m; k|] × [|batch; k; n|] → [|batch; m; n|] *)
let a = Nx.rand Nx.Float32 [|10; 3; 4|] in
let b = Nx.rand Nx.Float32 [|10; 4; 2|] in
let c = Nx.matmul a b   (* [|10; 3; 2|] *)
```

### Related products

| Function | Purpose |
|----------|---------|
| `dot` | Inner product (flattened inputs) |
| `vdot` | Complex-conjugate inner product |
| `inner` | Inner product over last axes |
| `outer` | Outer product of 1-D tensors |
| `tensordot` | Contraction over specified axes |
| `einsum` | Einstein summation notation |
| `kron` | Kronecker product |
| `cross` | Cross product of 3-element vectors |

### einsum

Einstein summation provides a compact notation for many tensor operations:

<!-- $MDX skip -->
```ocaml
(* Matrix multiplication: ij,jk->ik *)
let c = Nx.einsum "ij,jk->ik" [|a; b|]

(* Batch matrix multiply: bij,bjk->bik *)
let c = Nx.einsum "bij,bjk->bik" [|a; b|]

(* Trace: ii-> *)
let tr = Nx.einsum "ii->" [|m|]

(* Transpose: ij->ji *)
let t = Nx.einsum "ij->ji" [|m|]
```

## Decompositions

### Cholesky

Factor a symmetric positive-definite matrix: A = L·Lᵀ

<!-- $MDX skip -->
```ocaml
let l = Nx.cholesky a               (* lower triangular by default *)
let u = Nx.cholesky ~upper:true a   (* upper triangular *)
```

### QR

Factor A = Q·R where Q is orthogonal and R is upper triangular:

<!-- $MDX skip -->
```ocaml
let q, r = Nx.qr a                        (* reduced by default *)
let q_full, r_full = Nx.qr ~mode:`Complete a
```

### SVD

Singular value decomposition A = U·Σ·Vᵀ:

<!-- $MDX skip -->
```ocaml
let u, s, vt = Nx.svd a
let s_only = Nx.svdvals a   (* singular values only, more efficient *)
```

### Eigendecomposition

<!-- $MDX skip -->
```ocaml
(* General: returns complex eigenvalues and eigenvectors *)
let eigenvalues, eigenvectors = Nx.eig a
let eigenvalues_only = Nx.eigvals a

(* Symmetric/Hermitian: returns real eigenvalues *)
let eigenvalues, eigenvectors = Nx.eigh a
let eigenvalues_only = Nx.eigvalsh a
```

## Solving Linear Systems

### solve

Solve A·x = b for x:

<!-- $MDX skip -->
```ocaml
let x = Nx.solve a b
```

### lstsq

Least-squares solution (for overdetermined systems):

<!-- $MDX skip -->
```ocaml
let x, residuals, rank, sv = Nx.lstsq a b
```

### inv and pinv

Matrix inverse and pseudo-inverse:

<!-- $MDX skip -->
```ocaml
let a_inv = Nx.inv a          (* requires square, non-singular *)
let a_pinv = Nx.pinv a        (* works for any shape *)
```

## Norms and Properties

### norm

Compute various matrix and vector norms:

<!-- $MDX skip -->
```ocaml
(* Vector norms *)
let l2 = Nx.norm v                            (* L2 by default *)
let l1 = Nx.norm ~ord:(`Float 1.) v           (* L1 norm *)
let linf = Nx.norm ~ord:`Inf v                (* max absolute value *)

(* Matrix norms *)
let fro = Nx.norm ~ord:`Fro m                 (* Frobenius norm *)

(* Along specific axes *)
let row_norms = Nx.norm ~axis:[1] m           (* per-row L2 norm *)
```

### Other properties

<!-- $MDX skip -->
```ocaml
let d = Nx.det m                    (* determinant *)
let sd = Nx.slogdet m              (* sign and log-determinant *)
let tr = Nx.trace m                (* sum of diagonal elements *)
let r = Nx.matrix_rank m           (* numerical rank *)
let c = Nx.cond m                  (* condition number *)
let diag = Nx.diagonal m           (* extract diagonal *)
```

## FFT

Nx provides the full suite of discrete Fourier transforms.

### Basic FFT

<!-- $MDX skip -->
```ocaml
(* 1-D complex FFT and inverse *)
let spectrum = Nx.fft x
let reconstructed = Nx.ifft spectrum

(* 2-D FFT *)
let spectrum_2d = Nx.fft2 image

(* N-D FFT *)
let spectrum_nd = Nx.fftn ~axes:[0; 1; 2] volume
```

### Real FFT

For real-valued inputs, `rfft` is more efficient — it exploits conjugate symmetry and returns only the positive-frequency half:

<!-- $MDX skip -->
```ocaml
let spectrum = Nx.rfft signal           (* n/2+1 complex outputs *)
let signal_back = Nx.irfft spectrum     (* back to real *)

let spectrum_2d = Nx.rfft2 image
let spectrum_nd = Nx.rfftn ~axes:[0; 1] data
```

### Frequency axes

<!-- $MDX skip -->
```ocaml
let freqs = Nx.fftfreq n          (* frequency bins for fft *)
let rfreqs = Nx.rfftfreq n        (* frequency bins for rfft *)
let shifted = Nx.fftshift spectrum (* shift zero-frequency to center *)
```

## Next Steps

- [Array Operations](/docs/nx/array-operations/) — reshaping, broadcasting, slicing
- [Input/Output](/docs/nx/io/) — reading and writing files
- [NumPy Comparison](/docs/nx/numpy-comparison/) — side-by-side reference
