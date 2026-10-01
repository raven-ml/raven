(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Matrix products and factorizations as UOps.

    Each function takes the nodes of an operation's operands, of the shapes and
    dtypes nx gives them, and is the node that computes its result as nx
    documents it. The last two axes hold the matrices, and the leading ones are
    batch axes.

    A factorization takes a number of steps that its shapes fix, and none fails:
    where nx raises [Linalg_error], the result holds the non-finite values its
    steps produce, as each function states. [float16] computes at [float32] and
    rounds each result once. *)

open Tolk_next

val matmul : Ops.t -> Ops.t -> Ops.t
(** [matmul a b] is the product of the matrices of [a] and [b], their batch axes
    broadcast. The operands are converted to {!Lower_reduce.accumulator}'s type
    before they are multiplied, so that the products of narrow floats are exact,
    and each output is their sum ({!Lower_reduce.reduce}), converted once to the
    operands' dtype: floats from [+0.], integers modularly, and booleans as
    whether any product holds. *)

val cholesky : upper:bool -> Ops.t -> Ops.t
(** [cholesky ~upper a] is the lower-triangular [L] with [a = L Lᵀ], reading
    only the lower triangle of [a], or [Lᵀ] under [upper]. A pivot that is not
    positive, where nx raises, is NaN: the column it heads and every later one
    are NaN on and below the diagonal. *)

val qr : reduced:bool -> Ops.t -> Ops.t * Ops.t
(** [qr ~reduced a] is [(q, r)] with [a = q r], [q] orthogonal and [r] upper
    triangular, by Householder reflections. For [a] of [m] rows, [n] columns and
    [k = min m n], [q] has [k] columns and [r] [k] rows under [reduced], and [q]
    is square otherwise. The signs of the factors are unspecified. A column's
    norm is the square root of its sum of squares, so the factors lose accuracy
    where squares of elements overflow or fall below the normal range. *)

val lu : Ops.t -> Ops.t * Ops.t * Ops.t
(** [lu a] is [(lu, pivots, perm)], the factorization of [a] with partial
    pivoting as {!Nx_backend.S.lu} writes it: both factors packed in [a]'s
    shape, the [int64] row interchanged at each step, and the [int64] row order
    they produce. The pivot of a column is its first element of largest
    magnitude on or below the diagonal; an element below the diagonal that is
    NaN is never the pivot. A zero pivot leaves its column unscaled. *)

val svd : full_matrices:bool -> Ops.t -> Ops.t * Ops.t * Ops.t
(** [svd ~full_matrices a] is [(u, s, vt)] with [a = u diag(s) vt], by one-sided
    Jacobi rotations after a QR factorization: the [float64] singular values [s]
    in descending order, a zero one [+0.], and [u] and [vt] with orthonormal
    columns and rows, square under [full_matrices] and of [min m n] columns and
    rows otherwise. It rotates for a number of sweeps that grows as the
    logarithm of [min m n], and has [qr]'s range of accuracy. The signs of the
    vectors are unspecified. *)

val solve_triangular :
  upper:bool -> transpose:bool -> unit_diag:bool -> Ops.t -> Ops.t -> Ops.t
(** [solve_triangular ~upper ~transpose ~unit_diag a b] is the [x] with
    [a x = b], or [aᵀ x = b] under [transpose], reading only the triangle of [a]
    that [upper] names and taking its diagonal as ones under [unit_diag]. [b] is
    a vector of [a]'s batch shape and size, or right-hand sides as columns. A
    zero on the diagonal, where nx raises, makes the solution non-finite from
    its row on, in the order of substitution. *)
