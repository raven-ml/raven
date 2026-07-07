(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Memory access validation for UOp specs.

    This module mirrors tinygrad's [uop/validate.py] boundary for
    out-of-bounds checking. It contains deterministic checks only: disabled
    [CHECK_OOB] accepts memory accesses, image pointers bypass bounds checks,
    buffer sizes come from explicit shape metadata when present, and simple
    boolean gates can refine symbolic index bounds. General solver-backed
    validation is intentionally absent. *)

val is_index_source : Uop.t -> bool
(** [is_index_source src] is [true] iff [src] is a memory-source node,
    i.e. {!Ops.Index}, {!Ops.Shrink}, or one {!Ops.Cast} over either form.
    It does not check bounds; see {!validate_index_source}. *)

val validate_index_source : ?gate:Uop.t -> Uop.t -> bool
(** [validate_index_source ?gate src] is [true] iff [src] is an accepted
    memory-source node for a load or store.

    Accepted sources are {!Ops.Index}, {!Ops.Shrink}, or one {!Ops.Cast} over
    either form. When [CHECK_OOB] is enabled, the underlying index must be
    statically in bounds, bypassed by tinygrad's hard-to-model forms, protected
    by a statically false gate, or proven in bounds by the deterministic gate
    refinement implemented here. *)
