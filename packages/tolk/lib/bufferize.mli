(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Persistent tensor storage and post-realization identities. *)

val run : Tolk_uop.Uop.t -> Tolk_uop.Uop.t * (int, Tolk_uop.Uop.t) Hashtbl.t
(** [run sink] binds output and persistent allocations, retaining the effects
    needed to populate them in the returned sink. The map, keyed by original
    node tags, gives the storage identities that live tensors become after
    those effects execute. *)
