(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** nx's matrix routines as tolk graphs.

    {!op} computes each routine of {!Nx.Prim.linalg} on the matrices of a
    [float32] or [float64] operand, their last two axes of [m] rows and [n]
    columns, [k = min m n], each matrix on its own, at the operand's dtype.
    Each result has the shape and the meaning {!Nx_kernel.S} states, and a
    matrix that fails, as it states, is NaN in its own results alone.

    A routine repeats one step, a column of a factorisation or a round of
    rotations, in a tolk loop on the graph's device: its graph holds the step
    at most three times whatever the matrix's size, so that compiling it costs
    the same at every size.

    {b Numerics.} The bounds below hold to first order in the dtype's unit
    roundoff [u], barring overflow and underflow, with
    [γ(c) = c u / (1 - c u)], [|·|] taken elementwise and [‖·‖_F] the Frobenius
    norm. [R] is the number of rounds of rotations of [k] columns: [S] sweeps
    of [k - 1] rounds, [k] for an odd [k], none for [k < 2], with
    [S = ⌈p ⌈log₂ k⌉ / 10⌉] for [Eigh] and [S = ⌈p (⌈log₂ k⌉ + 1) / 10⌉] for
    [Svd], [p] the dtype's precision in bits.
    - [Cholesky]: [L Lᵀ = a + Δ] with [|Δ| <= γ(n + 1) |L| |Lᵀ|], [a] the
      symmetric matrix of [a]'s lower triangle. [Upper]'s factor is [Lower]'s
      transpose, bit for bit.
    - [Lu]: [L U = P a + Δ] with [|Δ| <= γ(k) |L| |U|], [P] the rows'
      permutation. The pivots follow {!Nx_kernel.S.lu}'s rule over the
      computed columns.
    - [Solve_triangular]: [|op a x - b| <= γ(n) |op a| |x|] for each column of
      [b], [op a] the triangle read, transposed under [adjoint], ones on its
      diagonal under [unit_diagonal].
    - [Qr]: by Householder reflections. With [g = γ(k (2m + 10))], each column
      [a_j] has [‖a_j - q r_j‖₂ <= g ‖a_j‖₂], and each element of [qᵀ q - I]
      is at most [g]. [r]'s diagonal has either sign.
    - [Eigh] and [Eigh_values]: by two-sided Jacobi rotations of the symmetric
      matrix of [a]'s lower triangle. Each eigenvalue is within
      [γ(16 (R + n)) ‖a‖_F] of the exact one and, with
      [g = γ(16 (R + n) (1 + √n))], [‖a - v diag w vᵀ‖_F <= g ‖a‖_F] and
      [‖vᵀ v - I‖_F <= g]. A matrix whose read triangle holds NaN or an
      infinity fails, as does one whose rotated elements off the diagonal
      have a norm above [16 n u ‖a‖_F]: the rotations did not converge.
    - [Svd] and [Svd_values]: by one-sided Jacobi rotations of the [r] of
      [a]'s QR, [aᵀ]'s for [m < n]. With [p = max m n] and
      [g = γ(16 (R + p²) (k + 2) √k)], [‖a - u diag s vh‖_F <= g ‖a‖_F] over
      [u]'s first [k] columns and [vh]'s first [k] rows,
      [‖uᵀ u - I‖_F <= g] and [‖vh vhᵀ - I‖_F <= g], and each singular value
      is within [3 g ‖a‖_F] of the exact one. A matrix that holds NaN or an
      infinity fails, as does one with rotated columns [c_i] and [c_j], of
      norms above [8 R (1 + √k) u ‖a‖_F], whose [|c_iᵀ c_j|] exceeds
      [16 k u ‖c_i‖₂ ‖c_j‖₂]: the rotations did not converge. [Svd_values]
      computes as [Svd] does.

    The rotations' bounds are worst cases that grow with [R], as each rotation
    of two columns moves them by at most [8 u] of their norm. *)

open Tolk

val op :
  device:Ops.device ->
  ('v 's 'd. ('v, 's, 'd) Nx.t -> Ops.t) ->
  ('d, 'r) Nx.Prim.linalg ->
  Ops.t array
(** [op ~device node l] is the node of each result of [l] in position order,
    its operand [x] read from [node x], computed on [device].

    Raises [Invalid_argument] naming the routine for a complex operand and for
    [Eig] and [Eig_values]. *)
