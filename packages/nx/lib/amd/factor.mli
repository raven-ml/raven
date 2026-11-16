(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The factorisations' launches of factor.hip, one call a routine.

    A plan takes a call nx.amd found nothing to refuse in, of a dtype
    {!Nx_kernel.full_precision} takes, and lowers it to one {!Plan.t}: a copy of
    an operand its steps do not read in place, then the steps and the
    contraction's products ({!Contract.product}), the routine's own matrices
    and its steps' scratch in the workspace. Each element
    of [cholesky], [lu] and [solve_triangular] takes the operations of the
    unblocked right-looking algorithm, so their bits do not depend on the
    blocking; no result's bits depend on the batch. A call of no element plans
    [None]. *)

val cholesky :
  Nx_kernel.triangle -> dst:Nx_array.any -> Nx_array.any -> Plan.t option
(** [cholesky t ~dst a] plans {!Nx_kernel.S.cholesky}. *)

val lu :
  lu:Nx_array.any ->
  pivots:Nx_array.any ->
  perm:Nx_array.any ->
  Nx_array.any ->
  Plan.t option
(** [lu ~lu ~pivots ~perm a] plans {!Nx_kernel.S.lu}. *)

val qr : q:Nx_array.any -> r:Nx_array.any -> Nx_array.any -> Plan.t option
(** [qr ~q ~r a] plans {!Nx_kernel.S.qr}, its factors as [q]'s shape states. *)

val solve_triangular :
  Nx_kernel.triangle ->
  adjoint:bool ->
  unit_diagonal:bool ->
  dst:Nx_array.any ->
  Nx_array.any ->
  Nx_array.any ->
  Plan.t option
(** [solve_triangular t ~adjoint ~unit_diagonal ~dst a b] plans
    {!Nx_kernel.S.solve_triangular}. *)
