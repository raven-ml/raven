(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Sharding transformations for device partitions and per-thread fragments. *)

val multi_pm : Tolk_uop.Uop.t -> Tolk_uop.Uop.t option
(** [multi_pm node] lowers operations on UNSHARD values to their local shards.
    The axes and owning ranges determine shard sizes; symbolic dimensions are
    retained. CALL bodies are rewritten recursively. Returns [None] when no
    rule applies. *)

val late_allreduce : int
(** [late_allreduce] is [LATE_ALLREDUCE] as read at startup (default [1]).
    Nonzero leaves ALLREDUCE nodes for {!lower_allreduces}; zero expands
    them while lowering shards. Zero also expands an allreduce that a
    reshard would lower to a reduce-scatter, which then moves the
    allreduce's bytes. *)

val lower_allreduces : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [lower_allreduces root] makes each ALLREDUCE that {!multi_pm} left in
    [root] a precompiled collective call. An allreduce whose only consumer
    keeps each device's own block along one axis (a reshard of the reduced
    value, possibly through the casts of [ALLREDUCE_CAST]) becomes a
    reduce-scatter that computes just those blocks; any other becomes one
    allreduce, whatever its number of consumers.

    Raises [Invalid_argument] naming the call if an ALLREDUCE sits in the
    body of a non-precompiled call. *)
