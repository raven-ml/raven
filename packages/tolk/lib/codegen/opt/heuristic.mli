(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** Heuristic-based kernel optimizations.

    Applies a sequence of hand-coded optimization steps to a kernel scheduler:
    tensor cores, image upcasts, matvec detection, grouping, masked upcasts,
    broadcast-based upcasts, reduce unrolling, local groups, and threading.

    *)

val nolocals_var : int Helpers.Context_var.t
(** Runtime override for [NOLOCALS] environment variable. *)

val hand_coded_optimizations : Postrange.t -> Postrange.t
(** [hand_coded_optimizations k] applies heuristic-based optimizations to the
    kernel scheduler [k]. Returns the (possibly mutated) scheduler. *)
