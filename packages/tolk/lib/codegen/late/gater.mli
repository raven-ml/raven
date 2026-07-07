(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Late gate relocation.

    Port of tinygrad [codegen/late/gater.py].

    This pass moves invalid-index gates onto memory operations after late
    decompositions, and folds a [where] around an already-gated load into the
    load's alternate value. *)

val pm_move_gates_from_index : Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [pm_move_gates_from_index root] rewrites load/store indexes of the form
    [where gate idx Invalid] to an ungated index plus an explicit load/store
    gate. Gated loads selected by the same surrounding [where] are rebuilt
    with the other [where] branch as their alternate value. *)
