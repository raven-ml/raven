(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Draws of {!Nx.Rng} compiled at a placement, against nx's eager draws on
    the host.

    rune.mli's Numerics section states a compiled draw eager's, bit for bit,
    on a faithful device. The samplers, their cases and covers are
    {!Nx_rng_support}'s. *)

val at :
  'd Nx.Placement.t -> (int -> 'd Nx.Placement.t) -> Nx_rng_support.runner
(** [at whole cut] compiles a draw with its key at [whole] and each argument
    of rank [r] at [cut r], with no call evaluated. *)

val law :
  (module Nx.Devices) ->
  dtypes:Nx_rng_support.dtype list ->
  shapes:Nx_rng_support.shape list ->
  Nx_rng_support.sampler ->
  Nx_rng_support.case ->
  unit
(** [law set ~dtypes ~shapes s c] is {!Nx_rng_support.law} for draws compiled
    on [set] with no call evaluated, the key and arguments whole on each
    device: each gives the bits of the draw computed eagerly on the host. *)
