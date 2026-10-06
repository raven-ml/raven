(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Matrix products and factorizations as UOps.

    Each function takes the nodes of an operation's operands, of the shapes and
    dtypes nx gives them, and is the node that computes its result as nx
    documents it. The last two axes hold the matrices, and the leading ones are
    batch axes.

    A factorization takes a number of steps that its shapes fix, and none
    raises: a matrix that fails, as each function states, has results whose
    every element is NaN, as nx documents. [float16] computes at [float32] and
    rounds each result once.

    The functions that take [~device] hold their step once, in a loop that
    runs on [device], the device of their result ({!Loop.repeat}): their graphs
    do not grow with the number of steps. *)

open Tolk

val matmul : Ops.t -> Ops.t -> Ops.t
(** [matmul a b] is the product of the matrices of [a] and [b], their batch axes
    broadcast. The operands are converted to {!Lower_reduce.accumulator}'s type
    before they are multiplied, so that the products of narrow floats are exact,
    and each output is their sum ({!Lower_reduce.reduce}), converted once to the
    operands' dtype: floats from [+0.], integers modularly, and booleans as
    whether any product holds. *)

val cholesky : upper:bool -> Ops.t -> Ops.t
(** [cholesky ~upper a] is the lower-triangular [L] with [a = L Lᵀ], reading
    only the lower triangle of [a], or [Lᵀ] under [upper]. A matrix with a pivot
    that is not positive, or is NaN, fails: it is not positive-definite. *)

val qr : device:Ops.device -> reduced:bool -> Ops.t -> Ops.t * Ops.t
(** [qr ~device ~reduced a] is [(q, r)] with [a = q r], [q] orthogonal and [r]
    upper triangular, by Householder reflections. For [a] of [m] rows, [n]
    columns and [k = min m n], [q] has [k] columns and [r] [k] rows under
    [reduced], and [q] is square otherwise. The signs of the factors are
    unspecified. A column's norm is the square root of its sum of squares, so
    the factors lose accuracy where squares of elements overflow or fall below
    the normal range. *)

val lu : device:Ops.device -> Ops.t -> Ops.t * Ops.t * Ops.t
(** [lu ~device a] is [(lu, pivots, perm)], the factorization of [a] with partial
    pivoting as {!Nx_backend.S.lu} writes it: both factors packed in [a]'s
    shape, the [int64] row interchanged at each step, and the [int64] row order
    they produce. The pivot of a column is its first element of largest
    magnitude on or below the diagonal; an element below the diagonal that is
    NaN is never the pivot. A zero pivot leaves its column unscaled. *)

val svd :
  device:Ops.device -> full_matrices:bool -> Ops.t -> Ops.t * Ops.t * Ops.t
(** [svd ~device ~full_matrices a] is [(u, s, vt)] with [a = u diag(s) vt], by
    one-sided Jacobi rotations after a QR factorization: the [float64] singular
    values [s] in descending order, a zero one [+0.], and [u] and [vt] with
    orthonormal columns and rows, square under [full_matrices] and of [min m n]
    columns and rows otherwise. It rotates for a number of sweeps that grows as
    the logarithm of [min m n], and has [qr]'s range of accuracy. The signs of
    the vectors are unspecified. A matrix holding NaN or an infinity, or whose
    rotated columns' inner products off the diagonal are not within [16 k] units
    of roundoff of their norm, [k = min m n], fails. *)

val eigh : device:Ops.device -> vectors:bool -> Ops.t -> Ops.t * Ops.t option
(** [eigh ~device ~vectors a] is [(w, v)] with [a v = v diag(w)], reading only
    the lower triangle of [a]: the [float64] eigenvalues [w] in ascending order
    and, under [vectors], the orthonormal eigenvectors [v] as columns, by
    two-sided Jacobi rotations. For [n] rows it rotates for [⌈p ⌈log2 n⌉ / 10⌉]
    sweeps of the pairs of rows, [p] the precision in bits at which it computes,
    enough for the slowest spectra, of repeated eigenvalues, to reach [n] units
    of roundoff of [a]'s norm. Every element of [v diag(w) vᵀ] and [vᵀ v] is
    then within some ten [n] units of roundoff of [a]'s and the identity's. The
    signs of the vectors, and the basis of the space of a repeated eigenvalue,
    are unspecified. A matrix whose lower triangle holds NaN or an infinity, or
    whose rotated elements off the diagonal are not within [16 n] units of
    roundoff of its norm, fails. *)

val solve_triangular :
  device:Ops.device ->
  upper:bool ->
  transpose:bool ->
  unit_diag:bool ->
  Ops.t ->
  Ops.t ->
  Ops.t
(** [solve_triangular ~device ~upper ~transpose ~unit_diag a b] is the [x] with
    [a x = b], or [aᵀ x = b] under [transpose], reading only the triangle of [a]
    that [upper] names and taking its diagonal as ones under [unit_diag]. [b] is
    a vector of [a]'s batch shape and size, or right-hand sides as columns. A
    matrix with a zero on the diagonal it reads fails. *)
