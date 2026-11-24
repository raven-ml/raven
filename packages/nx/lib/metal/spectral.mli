(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The spectral routines' launches of spectral.metal, one call a routine.

    A plan takes a call nx.metal found nothing to refuse in and lowers it to one
    {!Plan.t} of spectral.metal's steps over the operand, read in place where it
    is C-contiguous, else copied into the workspace by {!Contract.copy}, and
    each matrix's workspace, each step's units split into launches that each
    keep within {!Plan.allowed}. No result's bits depend on the batch, nor on
    where a launch's units start or end. A call of no matrix, or for [eigh] and
    [eig] of order [0], plans [None]. A call is declined at a dtype other than
    [float32] and [complex64], of an operand whose batch axes do not merge into
    one strided axis, where a single unit of a step passes the budget or Metal
    grants a kernel fewer threads than its group, and past the paths
    nx_metal.mli's Matrices states.

    Each plan takes [holds], where [holds k] is the threads Metal lets a group
    of [k]'s pipeline have ({!Loop.map}), and the call's facts ({!Operands}):
    its destinations, then its operand. *)

val eigh :
  holds:(Kernels.t -> int) ->
  Operands.t ->
  (Plan.t option, Nx_array.answer) result
(** [eigh ~holds xs] plans {!Nx_kernel.S.eigh}: [w], then [v] where the call
    stores it. *)

val svd :
  holds:(Kernels.t -> int) ->
  Operands.t ->
  (Plan.t option, Nx_array.answer) result
(** [svd ~holds xs] plans {!Nx_kernel.S.svd}: [s], then [u] and [vh] where the
    call stores them, its factors as their shapes state. *)

val eig :
  holds:(Kernels.t -> int) ->
  Operands.t ->
  (Plan.t option, Nx_array.answer) result
(** [eig ~holds xs] plans {!Nx_kernel.S.eig}: [w], then [v] where the call
    stores it. *)

(**/**)

val svd_of_facts :
  holds:(Kernels.t -> int) ->
  s:Operands.operand ->
  ?vectors:Operands.operand * Operands.operand ->
  Operands.operand ->
  (Plan.t option, Nx_array.answer) result
(** [svd_of_facts ~holds ~s ~vectors a] is {!svd}'s plan of a call whose [s],
    [u] and [vh], and [a], have those facts. It reads no buffer, so tests plan
    calls of any extents. *)

val bound_iterations :
  sweeps:int -> steps:int -> qr:int -> passes:int -> secular:int -> unit
(** [bound_iterations ~sweeps ~steps ~qr ~passes ~secular] is
    {!Nx_metal.bound_iterations}. *)
