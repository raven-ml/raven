(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** A transform's launches of fft.hip.

    A plan takes a transform nx.amd found nothing to refuse in and lowers it
    to launches of fft and fft_real, their words, the workspace they share
    and a {!Plan.Constant} of each line length's tables, which the door
    makes once it accepts the call. A line's arithmetic depends on its
    points and its precision alone, so a result's bits do not depend on
    its layout, its batch or the launches it runs as. *)

val c2c :
  Nx_kernel.direction ->
  axes:int iarray ->
  dst:(Complex.t, 's) Nx_array.t ->
  (Complex.t, 's) Nx_array.t ->
  Plan.t option
(** [c2c d ~axes ~dst x] plans {!Nx_kernel.S.c2c} of operands it refuses
    nothing in, as one call, [None] where it has no launch. A table's
    bytes raise [Out_of_memory] where the host has no memory for them. *)

val r2c :
  axes:int iarray ->
  dst:(Complex.t, 'c) Nx_array.t ->
  (float, 'r) Nx_array.t ->
  Plan.t option
(** [r2c ~axes ~dst x] plans {!Nx_kernel.S.r2c} of a last axis of at least
    one point, as {!c2c} does. *)

val c2r :
  axes:int iarray ->
  dst:(float, 'r) Nx_array.t ->
  (Complex.t, 'c) Nx_array.t ->
  Plan.t option
(** [c2r ~axes ~dst x] plans {!Nx_kernel.S.c2r}, as {!c2c} does. *)
