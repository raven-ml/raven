(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. ISC License.

  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Canonical GPU workitem dimensions.

    Tinygrad represents GPU workitem ids in the UOp graph as raw
    {!Tolk_uop.Ops.Special} names: [gidxN] for group ids, [lidxN] for local
    ids, and [idxN] for flat global ids. This module is the typed view used by
    codegen and renderers at the edges where those names have GPU launch
    meaning. *)

type t =
  | Group_id of int
      (** Workgroup id on axis [N], encoded as [gidxN]. *)
  | Local_id of int
      (** Local workitem id on axis [N], encoded as [lidxN]. *)
  | Global_idx of int
      (** Flat global workitem id on axis [N], encoded as [idxN]. *)
(** The type for canonical GPU workitem dimensions. *)

val axis : t -> int
(** [axis dim] is the axis carried by [dim]. *)

val to_special_name : t -> string
(** [to_special_name dim] is the raw {!Tolk_uop.Ops.Special} name for [dim]. *)

val of_special_name : string -> t option
(** [of_special_name name] decodes a canonical GPU SPECIAL name. *)

val compare : t -> t -> int
(** [compare] orders group ids before local ids before flat global ids, then by
    axis. *)

val equal : t -> t -> bool
(** [equal a b] is [true] iff [compare a b = 0]. *)

val compare_special_name : string -> string -> int
(** [compare_special_name a b] orders canonical names with {!compare}. Unknown
    names sort after canonical names and are then ordered lexicographically. *)
