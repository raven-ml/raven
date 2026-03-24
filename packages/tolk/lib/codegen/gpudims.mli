(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** GPU dimension mapping.

    Maps logical kernel ranges to physical GPU grid dimensions
    ({!Tolk_ir.Kernel.view.Special} nodes) via grouping, splitting, and
    contraction.

    The pass replaces {!Tolk_ir.Kernel.view.Range} nodes of Global, Thread,
    Warp, Local, and Group_reduce kinds with SPECIAL hardware index nodes,
    adjusting for renderer grid size limits. For threaded backends, a single
    [core_id] variable replaces all global ranges. Missing local ranges on
    global stores are gated with validity masks. *)

val get_grouped_dims :
  string -> Tolk_ir.Kernel.t array -> int list option -> reverse:bool -> Tolk_ir.Kernel.t list
(** [get_grouped_dims prefix dims max_sizes ~reverse] maps logical [dims] to
    physical SPECIAL dimension nodes.

    [dims] are kernel expression nodes (typically constant-valued, but
    symbolic expressions are accepted for the grouping path).

    [prefix] is ["gidx"], ["lidx"], or ["idx"]. [max_sizes] constrains physical
    dimensions (or [None] for no constraint). When [reverse], dims are reversed
    before mapping and the result reversed back.

    This is the core grouping/splitting/contraction logic used by
    {!pm_add_gpudims}. *)

val pm_add_gpudims : Renderer.t -> Tolk_ir.Kernel.t -> Tolk_ir.Kernel.t
(** [pm_add_gpudims renderer root] replaces GPU-mappable ranges in [root] with
    SPECIAL dimension nodes sized to the renderer's grid limits.

    Returns [root] unchanged when the kernel has no GPU-mappable ranges or
    already contains SPECIAL nodes. *)
