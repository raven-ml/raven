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

    A routine repeats one step, a column of a factorisation, in a tolk loop
    on the device of its operand's storage: its graph holds the step once,
    whatever the matrix's size, and compiling it costs one step.

    {b Numerics.} The bounds below hold to first order in the dtype's unit
    roundoff [u], barring overflow and underflow, with
    [γ(c) = c u / (1 - c u)] and [|·|] taken elementwise.
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
      is at most [g]. [r]'s diagonal has either sign. *)

open Tolk

val op :
  ('v 's 'd. ('v, 's, 'd) Nx.t -> Ops.t) ->
  ('d, 'r) Nx.Prim.linalg ->
  Ops.t array
(** [op node l] is the node of each result of [l] in position order, its
    operand [x] read from [node x].

    Raises [Invalid_argument] naming the routine for a complex operand and for
    [Svd_values], [Svd], [Eigh_values], [Eigh], [Eig] and [Eig_values], and
    naming [Linalg] for an operand of a loop that
    reads no storage, whose device it cannot name. *)
