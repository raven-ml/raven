(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Memory access validation for UOp specs.

    Deterministic out-of-bounds checking only: with [CHECK_OOB] disabled every
    memory access is accepted, image accesses (three-dimensional buffers whose
    last dimension is four) bypass bounds checks, buffer sizes come from the
    access target's shape, and simple boolean gates can refine symbolic index
    bounds. General solver-backed validation is intentionally absent. *)

val is_index_source : Uop.t -> bool
(** [is_index_source src] is [true] iff [src] is a memory-source node,
    i.e. {!Ops.Index}, {!Ops.Shrink}, or one {!Ops.Cast} over either form.
    It does not check bounds; see {!validate_index_source}. *)

val validate_index_source : ?gate:Uop.t -> Uop.t -> bool
(** [validate_index_source ?gate src] is [true] iff [src] is an accepted
    memory-source node for a load or store.

    Accepted sources are {!Ops.Index}, {!Ops.Shrink}, or one {!Ops.Cast} over
    either form. When [CHECK_OOB] is enabled, the underlying index must be
    statically in bounds, bypassed by a hard-to-model form (a bitcast or stack
    in the index expression), protected by a statically false gate, or proven in
    bounds by the deterministic gate refinement implemented here. *)
