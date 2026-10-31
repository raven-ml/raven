(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tangents of the matrix routines of {!Nx.Prim.type-linalg}.

    Each function but {!solve_triangular} takes a routine's results on a
    primal [a] and the tangent [da] of [a], of [a]'s shape and dtype, and
    returns the tangents of the routine's results, in the routine's order,
    each of its result's shape and dtype. {!lu} returns the packed factors'
    tangent alone, since the positions have none. Like the routines, they
    compute on each matrix of a stack, the last two axes. Where two
    eigenvalues or two singular values are equal, the tangents of the vectors
    are infinite or NaN. A [name] is the calling operation's, and starts the
    message of an [Invalid_argument] the function raises. *)

val cholesky :
  Nx_kernel.triangle ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t
(** [cholesky t y da] is the tangent of the factor [y] that [Cholesky] gives
    under [t]. It reads [da]'s lower triangle and the real parts of its
    diagonal alone, as the routine reads [a]. *)

val lu :
  ('v, 's, 'd) Nx.t ->
  (int64, Nx.int64_elt, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t
(** [lu packed perm da] is the tangent of [Lu]'s packed factors [packed],
    whose rows are [a]'s permuted by [perm], packed as [packed] is: [L]'s
    tangent below the diagonal, [U]'s on and above it. [U] must have no zero
    pivot. A zero pivot makes the tangent NaN. *)

val qr :
  string ->
  Nx_kernel.factors ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t * ('v, 's, 'd) Nx.t
(** [qr name factors q r da] is the tangents of [q] and [r]. [a] must have
    full column rank, or for a wide [a] its first [m] columns must, [m] its
    rows. A zero on [r]'s diagonal makes the tangents NaN.

    Raises [Invalid_argument] for [Complete] factors of a tall [a], whose
    columns of [q] past [a]'s are any completion. *)

val svd_values :
  (float, 'r) Nx.dtype ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  (float, 'r, 'd) Nx.t
(** [svd_values dtype u vh da] is the tangent of the singular values of [a],
    of [dtype], from [u] and [vh], which must be [a]'s [Reduced] factors. *)

val svd :
  string ->
  Nx_kernel.factors ->
  (float, 'r) Nx.dtype ->
  (float, 'r, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  (float, 'r, 'd) Nx.t * ('v, 's, 'd) Nx.t * ('v, 's, 'd) Nx.t
(** [svd name factors dtype s u vh da] is the tangents of [s], [u] and [vh].
    Each right singular vector's tangent is orthogonal to it, and each left
    one's carries the phase that keeps [a = u diag s vh]. For a tall or wide
    [a], the tangents are also infinite or NaN where a singular value is
    zero.

    Raises [Invalid_argument] for [Complete] factors of a matrix that is not
    square. *)

val eigh_values :
  (float, 'r) Nx.dtype ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  (float, 'r, 'd) Nx.t
(** [eigh_values dtype v da] is the tangent of the eigenvalues of [a], of
    [dtype], from [a]'s eigenvectors [v]. It reads [da] as [Eigh] reads
    [a]. *)

val eigh :
  (float, 'r) Nx.dtype ->
  (float, 'r, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  (float, 'r, 'd) Nx.t * ('v, 's, 'd) Nx.t
(** [eigh dtype w v da] is the tangents of the eigenvalues [w] and the
    eigenvectors [v], each eigenvector's tangent orthogonal to it. It reads
    [da] as [Eigh] reads [a]. *)

val eig_values :
  (Complex.t, 'c, 'd) Nx.t -> ('v, 's, 'd) Nx.t -> (Complex.t, 'c, 'd) Nx.t
(** [eig_values v da] is the tangent of the eigenvalues of [a] from its
    eigenvectors [v], which must be invertible. *)

val eig :
  (Complex.t, 'c, 'd) Nx.t ->
  (Complex.t, 'c, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  (Complex.t, 'c, 'd) Nx.t * (Complex.t, 'c, 'd) Nx.t
(** [eig w v da] is the tangents of the eigenvalues [w] and the
    eigenvectors [v], which must be invertible. Each eigenvector's tangent is
    orthogonal to it, so that it keeps its unit norm. *)

val solve_triangular :
  by:string ->
  Nx_kernel.triangle ->
  adjoint:bool ->
  unit_diagonal:bool ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t ->
  ('v, 's, 'd) Nx.t option ->
  ('v, 's, 'd) Nx.t option ->
  ('v, 's, 'd) Nx.t
(** [solve_triangular ~by t ~adjoint ~unit_diagonal a x da db] is the tangent
    of the solution [x] that [Solve_triangular] gives for [a] and [b], from
    the tangents [da] of [a] and [db] of [b], [None] where zero. It reads
    [da] as the routine reads [a], and solves through {!Nx.Prim.eval}
    [~by]. One of [da] and [db] must be [Some]. *)
