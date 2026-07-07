(*---------------------------------------------------------------------------
  Copyright (c) 2024 the tiny corp. MIT License (see LICENSE-tinygrad).
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: MIT AND ISC
  ---------------------------------------------------------------------------*)

(** GPU dimension mapping.

    Maps logical kernel ranges to physical GPU grid dimensions
    ({!Tolk_uop.Ops.Special} nodes) via grouping, splitting, and
    contraction.

    The pass replaces {!Tolk_uop.Ops.Range} nodes of Global, Thread,
    Warp, Local, and Group_reduce kinds with SPECIAL hardware index nodes,
    adjusting for renderer grid size limits. Threaded backends require exactly
    one global range and no local ranges; that range is replaced by a
    [core_id] variable. Missing local ranges on global stores are gated with
    validity masks. *)

type dim_kind =
  | Group_id
  | Local_id
  | Global_idx
(** Physical SPECIAL family to emit for grouped dimensions. *)

val get_grouped_dims :
  dim_kind ->
  Tolk_uop.Uop.t array ->
  int list option ->
  reverse:bool ->
  Tolk_uop.Uop.t list
(** [get_grouped_dims kind dims max_sizes ~reverse] maps logical [dims]
    to physical SPECIAL dimension nodes.

    [kind] chooses whether physical dimensions are emitted as group IDs, local
    IDs, or flat global IDs. [max_sizes] constrains physical dimensions
    ([None] for no constraint). When present, it is the physical renderer axis
    limit list used by tinygrad's GPU dimension mapper; renderer-backed calls
    pass the backend grid or workgroup limits, normally three GPU axes. When
    [reverse], dims are reversed before mapping and the result reversed back.

    Raises [Failure] if dims cannot be grouped or split to fit
    [max_sizes]. *)

val pm_add_gpudims : Renderer.t -> Tolk_uop.Uop.t -> Tolk_uop.Uop.t
(** [pm_add_gpudims renderer root] replaces GPU-mappable ranges in [root]
    with SPECIAL dimension nodes sized to the renderer's grid limits.

    Returns [root] unchanged when the kernel has no GPU-mappable ranges
    or already contains SPECIAL nodes. *)
