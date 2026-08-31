(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Linear algebra as unrolled Tolk compositions.

    QR, triangular solves, and Cholesky have no single-Uop Tolk form: their
    classic implementations iterate over data, and the Uop vocabulary has no
    host control flow. Their iteration counts are fixed by the input shapes
    alone, though, so {!qr}, {!solve_triangular}, and {!cholesky} unroll them
    at graph-construction time into ordinary compositions of {!Op},
    {!Elementwise}, {!Movement}, and {!Reduce}. The lowering sees a plain
    static graph and compiles it for every Tolk device; graph size grows
    linearly in the matrix dimension.

    Conventions follow LAPACK — the Householder reflector sign, zero-tail
    columns taking no reflector, and the reflector tails stored on the
    subdiagonal — so results match the classic algorithms up to
    floating-point association. *)

val qr : reduced:bool -> Tensor.t -> Tensor.t * Tensor.t
(** [qr ~reduced a] is the Householder QR factorization of [a] as [(q, r)]
    with [a ≍ q·r] and orthonormal [q] (columns of [q] orthonormal when [a]
    has full rank).

    [a] has shape [batch @ [m; n]]. With [reduced:true] (the thin
    factorization) [q] is [batch @ [m; min(m, n)]] and [r] is
    [batch @ [min(m, n); n]]; with [reduced:false] the factors are
    [batch @ [m; m]] and [batch @ [m; n]]. A column whose subdiagonal tail is
    already zero takes no reflector, so its [r] diagonal entry keeps its sign.

    Raises [Invalid_argument] if [a] has fewer than 2 dimensions or a
    non-float dtype. *)

val solve_triangular :
  upper:bool ->
  transpose:bool ->
  unit_diag:bool ->
  Tensor.t -> Tensor.t -> Tensor.t
(** [solve_triangular ~upper ~transpose ~unit_diag a b] solves the triangular
    system [a·x = b] for [x] by forward substitution — unrolled, one row
    (block) per step, into matmuls and element-wise arithmetic.

    [a] has shape [batch @ [n; n]] and is read as upper triangular when
    [upper] is [true] and lower triangular otherwise; the other triangle is
    never read, so it may hold arbitrary values. [transpose] solves
    [aᵀ·x = b] instead, and [unit_diag] treats the diagonal of [a] as ones
    without reading it. [b] is one right-hand-side vector of shape
    [batch @ [n]] or [nrhs] right-hand sides stacked as [batch @ [n; nrhs]];
    the result has the shape of [b].

    Wide right-hand sides (at least {!block_threshold} columns on a matrix of
    more than [2 * block_rows] rows) solve block-by-block — {!block_rows}-row
    blocks, each diagonal block inverted once, one GEMM per block against the
    rows solved so far — which runs the substitution as real GEMMs instead of
    one thin matmul per row.

    Raises [Invalid_argument] if [a] has a non-float dtype. A singular
    [a] (a zero diagonal entry with [unit_diag:false]) yields infinities
    rather than an error — detecting it needs host control flow. *)

val cholesky : upper:bool -> Tensor.t -> Tensor.t
(** [cholesky ~upper a] is the Cholesky factorization of the
    symmetric-positive-definite [a]: the lower triangular [l] with
    [a ≍ l·lᵀ], or the upper triangular [u] with [a ≍ uᵀ·u] when [upper] is
    [true]. One column of the factor is unrolled per step.

    [a] has shape [batch @ [n; n]]. The per-element arithmetic is the classic
    unblocked algorithm's, so results agree with blocked implementations up
    to floating-point association.

    Raises [Invalid_argument] if [a] has fewer than 2 dimensions or a
    non-float dtype. A non-positive-definite [a] yields nans rather than an
    error — detecting it needs host control flow. *)

val block_rows : int
(** Row-block size of the blocked triangular solve. *)
