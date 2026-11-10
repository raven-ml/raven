(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The factorisations' launches of factor.cu and of the contraction's
    products, one submission a call.

    A plan takes a call nx.cuda found nothing to refuse in, of a dtype
    {!Nx_kernel.full_precision} takes, and lowers it to one {!Contract.call}:
    a copy of an operand its steps do not read in place, then the steps and
    products, the routine's own matrices in buffers of the call's own, its
    steps' scratch in the workspace. Each element of [cholesky], [lu] and
    [solve_triangular] takes the operations of the unblocked right-looking
    algorithm, so their bits do not depend on the blocking; no result's
    bits depend on the batch. *)

val cholesky :
  Nx_kernel.triangle -> dst:Nx_array.any -> Nx_array.any -> Bytes.t Launch.plan
(** [cholesky t ~dst a] plans {!Nx_kernel.S.cholesky}. *)

val lu :
  lu:Nx_array.any ->
  pivots:Nx_array.any ->
  perm:Nx_array.any ->
  Nx_array.any ->
  Bytes.t Launch.plan
(** [lu ~lu ~pivots ~perm a] plans {!Nx_kernel.S.lu}. *)

val qr : q:Nx_array.any -> r:Nx_array.any -> Nx_array.any -> Bytes.t Launch.plan
(** [qr ~q ~r a] plans {!Nx_kernel.S.qr}, its factors as [q]'s shape
    states. *)

val solve_triangular :
  Nx_kernel.triangle ->
  adjoint:bool ->
  unit_diagonal:bool ->
  dst:Nx_array.any ->
  Nx_array.any ->
  Nx_array.any ->
  Bytes.t Launch.plan
(** [solve_triangular t ~adjoint ~unit_diagonal ~dst a b] plans
    {!Nx_kernel.S.solve_triangular}. *)
