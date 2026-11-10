(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A transform's launches of fft.cu.

    A plan takes a transform nx.cuda found nothing to refuse in and lowers it
    to launches of fft and fft_real, their parameters and the workspace
    they share. A line's arithmetic depends on its points and its
    precision alone, so a result's bits do not depend on its layout, its
    batch or the launches it runs as. *)

type cache
(** The type for a device's plans of lines, kept from call to call. *)

val cache : Rig.t -> cache
(** [cache d] keeps plans on [d]. *)

type params
(** The type for a launch's parameters. *)

val c2c :
  cache ->
  Nx_kernel.direction ->
  axes:int iarray ->
  dst:(Complex.t, 's) Nx_array.t ->
  (Complex.t, 's) Nx_array.t ->
  params Launch.plan
(** [c2c c d ~axes ~dst x] plans {!Nx_kernel.S.c2c} of operands it refuses
    nothing in, as one call. Raises [Out_of_memory] where the host has no
    memory for a plan. *)

val r2c :
  cache ->
  axes:int iarray ->
  dst:(Complex.t, 'c) Nx_array.t ->
  (float, 'r) Nx_array.t ->
  params Launch.plan
(** [r2c c ~axes ~dst x] plans {!Nx_kernel.S.r2c} of a last axis of at least
    one point, as {!c2c} does. *)

val c2r :
  cache ->
  axes:int iarray ->
  dst:(float, 'r) Nx_array.t ->
  (Complex.t, 'c) Nx_array.t ->
  params Launch.plan
(** [c2r c ~axes ~dst x] plans {!Nx_kernel.S.c2r}, as {!c2c} does. *)

val write : params -> Rig.Submission.Run.t -> Rig.Submission.block -> unit
(** [write p r at] stores [p] as the parameters of [r]'s block [at]. *)
