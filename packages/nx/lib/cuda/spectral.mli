(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The spectral routines' launches of spectral.cu and of the
    contraction's products, one submission a call.

    A plan takes a call nx.cuda found nothing to refuse in, of a dtype
    {!Nx_kernel.full_precision} takes, and lowers it to one
    {!Contract.call}: a copy of an operand that is not C-contiguous, then
    the steps and products over each matrix's workspace, laid out as
    kernels.h states, in a buffer of the call's own. No result's bits
    depend on the batch. *)

val eigh :
  w:Nx_array.any -> ?v:Nx_array.any -> Nx_array.any -> Bytes.t Launch.plan
(** [eigh ~w ~v a] plans {!Nx_kernel.S.eigh}. *)

val svd :
  s:Nx_array.any ->
  ?vectors:Nx_array.any * Nx_array.any ->
  Nx_array.any ->
  Bytes.t Launch.plan
(** [svd ~s ~vectors:(u, vh) a] plans {!Nx_kernel.S.svd}, its factors as
    [u]'s and [vh]'s shapes state. *)

val eig :
  w:Nx_array.any -> ?v:Nx_array.any -> Nx_array.any -> Bytes.t Launch.plan
(** [eig ~w ~v a] plans {!Nx_kernel.S.eig}. *)
