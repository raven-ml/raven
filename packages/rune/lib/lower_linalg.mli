(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** nx's matrix routines as tolk graphs.

    {!op} computes each routine of {!Nx.Prim.linalg} on each matrix of a
    [float32] or [float64] operand, at the operand's dtype. A matrix is the
    operand's last two axes, of [m] rows and [n] columns, and [k = min m n].
    Each matrix is computed on its own. Each result has the shape and the
    meaning {!Nx_kernel.S} states. A matrix that fails, as {!Nx_kernel.S}
    defines failure, makes only its own results NaN.

    A routine repeats one step, a column of a factorisation or a round of
    rotations, in a tolk loop on the graph's device. Its graph holds the step at
    most three times whatever the matrix's size, so compiling it costs the same
    at every size.

    {b Numerics.} These bounds implement the matrix rows of
    {!Rune.section-numerics}. They hold to first order in the dtype's unit
    roundoff [u], barring overflow and underflow, with [γ(c) = c u / (1 - c u)]
    ({!Nx_kernel.section-bounds}), [|·|] taken elementwise and [‖·‖_F] the
    Frobenius norm. [R] is the number of rounds of rotations of [k] columns. It
    is [S] sweeps of [k - 1] rounds each, or of [k] rounds for an odd [k], and
    [0] for [k < 2], where [S = ⌈b ⌈log₂ k⌉ / 10⌉] for [Eigh],
    [S = ⌈b (⌈log₂ k⌉ + 1) / 10⌉] for [Svd], and [b] is the dtype's precision in
    bits.
    - [Cholesky]: [L Lᵀ = a + Δ] with [|Δ| <= γ(n + 1) |L| |Lᵀ|], where [a]
      stands for the symmetric matrix of its lower triangle. [Upper]'s factor is
      [Lower]'s transpose, bit for bit.
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
      matrix of [a]'s lower triangle, scaled by a power of two. Each
      eigenvalue is within [γ(16 (R + n)) ‖a‖_F] of the exact one and, with
      [g = γ(16 (R + n) (1 + √n))], [‖a - v diag w vᵀ‖_F <= g ‖a‖_F] and
      [‖vᵀ v - I‖_F <= g]. A matrix fails where its read triangle holds NaN or
      an infinity, or where its rotated elements off the diagonal have a norm
      above [16 n u ‖a‖_F], which shows its rotations have not converged.
    - [Svd] and [Svd_values]: by one-sided Jacobi rotations of the [r] of [a]'s
      QR, [aᵀ]'s for [m < n], scaled by a power of two. With [p = max m n] and
      [g = γ(8 k (p (p + 6) + 8 R))], [‖a - u diag s vh‖_F <= g ‖a‖_F] over
      [u]'s first [k] columns and [vh]'s first [k] rows, [‖uᵀ u - I‖_F <= g] and
      [‖vh vhᵀ - I‖_F <= g], and each singular value is within [3 g ‖a‖_F] of
      the exact one. From [k = p = 68] in [float32] and [k = p = 82313] in
      [float64], and at smaller [k] for [p > k], [8 k (p (p + 6) + 8 R) u] is at
      least [1/2], and the bound is vacuous. A matrix fails where it holds NaN
      or an infinity, or where two rotated columns [c_i] and [c_j] of norms
      above [8 R (1 + √k) u ‖a‖_F] have [|c_iᵀ c_j|] above
      [16 k u ‖c_i‖₂ ‖c_j‖₂], which shows its rotations have not converged.
      [Svd_values] computes as [Svd] does.

    The rotations' bounds are worst cases that grow with [R]. *)

open Tolk

val op :
  device:Ops.device ->
  ('v 's 'd. ('v, 's, 'd) Nx.t -> Ops.t) ->
  ('d, 'r) Nx.Prim.linalg ->
  Ops.t array
(** [op ~device node l] is the node of each result of [l], in position order,
    computed on [device]. It reads each operand [x] from [node x].

    Raises [Invalid_argument] naming the routine for a complex operand and for
    [Eig] and [Eig_values]. *)
