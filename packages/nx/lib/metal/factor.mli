(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The factorisations' launches of factor.metal, one call a routine.

    A plan takes a call nx.metal found nothing to refuse in and lowers it to one
    {!Plan.t} of factor.metal's steps, each step's units split into launches
    that each keep within {!Plan.allowed}. Each element of [cholesky], [lu] and
    [solve_triangular] takes the operations of the unblocked right-looking
    algorithm; no result's bits depend on the batch. A call of no element plans
    [None]. A call is declined at a dtype other than [float32] and [complex64],
    and for an operand whose batch axes do not merge into one (nx_metal.mli's
    Matrices).

    Each plan takes [holds], where [holds k] is the threads Metal lets a group
    of [k]'s pipeline have ({!Loop.map}), and the call's facts ({!Operands}):
    its destinations, then its operands, in the order of {!Nx_kernel.S}'s
    arguments. *)

val cholesky :
  holds:(Kernels.t -> int) ->
  Nx_kernel.triangle ->
  Operands.t ->
  (Plan.t option, Nx_array.answer) result
(** [cholesky ~holds t xs] plans {!Nx_kernel.S.cholesky}. *)

val lu :
  holds:(Kernels.t -> int) ->
  Operands.t ->
  (Plan.t option, Nx_array.answer) result
(** [lu ~holds xs] plans {!Nx_kernel.S.lu}. *)

val qr :
  holds:(Kernels.t -> int) ->
  Operands.t ->
  (Plan.t option, Nx_array.answer) result
(** [qr ~holds xs] plans {!Nx_kernel.S.qr}, its factors as [q]'s shape states.
*)

val solve_triangular :
  holds:(Kernels.t -> int) ->
  Nx_kernel.triangle ->
  adjoint:bool ->
  unit_diagonal:bool ->
  Operands.t ->
  (Plan.t option, Nx_array.answer) result
(** [solve_triangular ~holds t ~adjoint ~unit_diagonal xs] plans
    {!Nx_kernel.S.solve_triangular}. *)

(**/**)

val bound_lu : rows:int option -> unit
(** [bound_lu ~rows:(Some r)] makes an LU panel of [m] rows over several groups
    take [⌈m / r⌉] groups of at most [r] rows in place of its rule's, so that
    tests reach a level search, several rows a thread and a partial last group
    with small matrices; [bound_lu ~rows:None] restores the rule. It applies to
    calls planned after it returns, and changes no bit. For tests.

    Raises [Invalid_argument] if [r < 1]. *)

val bound_divide : below:int -> unit
(** [bound_divide ~below] makes the LU's and the triangular solve's launches
    divide their units in 64 bits where a launch's last unit or its divisor
    reaches [below], in place of [2{^32}], past which the 32-bit division by a
    divisor fails, so that tests reach both divisions and a launch across the
    bound with small matrices. It applies to calls planned after it returns, and
    changes no bit. For tests.

    Raises [Invalid_argument] if [below] is not in \[1; [2{^32}]\]. *)
