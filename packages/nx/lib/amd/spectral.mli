(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The spectral routines' launches of spectral.hip and the contraction's
    products, one call.

    A plan takes a call nx.amd found nothing to refuse in, of a dtype
    {!Nx_kernel.full_precision} takes, and plans it as one {!Plan.t}: a
    copy of an operand that is not C-contiguous into workspace, then the
    steps and {!Contract.product}s over each matrix's workspace. No
    result's bits depend on the batch. A call of no matrix, or for [eigh]
    and [eig] of order [0], plans [None]. *)

val eigh :
  w:Nx_array.any ->
  ?v:Nx_array.any ->
  Nx_array.any ->
  Plan.t option
(** [eigh ~w ~v a] plans {!Nx_kernel.S.eigh}. *)

val svd :
  s:Nx_array.any ->
  ?vectors:Nx_array.any * Nx_array.any ->
  Nx_array.any ->
  Plan.t option
(** [svd ~s ~vectors:(u, vh) a] plans {!Nx_kernel.S.svd}, its factors as
    [u]'s and [vh]'s shapes state. *)

val eig :
  w:Nx_array.any ->
  ?v:Nx_array.any ->
  Nx_array.any ->
  Plan.t option
(** [eig ~w ~v a] plans {!Nx_kernel.S.eig}. *)

(**/**)

val bound_iterations :
  sweeps:int -> steps:int -> qr:int -> rotations:int -> unit
(** [bound_iterations ~sweeps ~steps ~qr ~rotations] is
    {!Nx_amd.bound_iterations}. *)
