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
