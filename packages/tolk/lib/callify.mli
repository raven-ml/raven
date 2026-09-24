(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Storage interface normalization for schedule caching. *)

val transform_to_call : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [transform_to_call sink] collects effects from a bufferized tensor graph,
    replaces bound inputs with positional formals, and canonicalizes anonymous
    allocation identities within each call scope. *)

val is_store_after : Tolk_uop.Uop.t -> bool
(** [is_store_after u] identifies a materialized tensor effect, excluding
    scalar bindings and anonymous call results. *)

val contiguous_view : Tolk_uop.Uop.t -> Tolk_uop.Uop.t option
(** [contiguous_view u] folds a contiguous movement view of bound storage
    without discarding pending effects. *)
