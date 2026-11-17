(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Draws of {!Nx.Rng} compiled on a set, against nx's eager draws on the
    host.

    rune.mli's Numerics section states a compiled draw eager's, bit for bit,
    but for {!Nx.Rng.dirichlet}, whose row sum is a fold. The samplers, their
    cases and covers are {!Nx_rng_support}'s. *)

val law :
  (module Nx.Devices) ->
  dtypes:Nx_rng_support.dtype list ->
  shapes:Nx_rng_support.shape list ->
  Nx_rng_support.sampler ->
  Nx_rng_support.case ->
  unit
(** [law set ~dtypes ~shapes s c] is {!Nx_rng_support.law} for draws compiled
    on [set] with no call evaluated: each gives the bits of the draw computed
    eagerly on the host, a {!Nx.Rng.dirichlet} draw instead within
    {!Nx_rng_support.bound}, the bound rune.mli's Numerics states. *)
