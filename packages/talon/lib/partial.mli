(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Aggregates that stream, lowered to ordinary aggregates.

    An aggregate of reductions that compose over batches is the ordinary
    aggregate of its {e state}: fields of one row per group, held rows of one
    batch per row, the state rows first so that a group's rows fold in their
    order. A compaction merges the state and the held rows with reductions over
    the fields, and {!finish} computes the outputs from the fields. *)

type t = {
  schema : Schema.t;
      (** The held and state rows: the keys, the fields, and {!row_field}. *)
  held : (string * Expr.packed) list;
      (** Over the input: the keys, then each field's held value, the fields
          in their reductions' evaluation order. The run adds {!row_field}. *)
  merge : (string * Expr.packed) list;
      (** Over [schema], grouped by the keys: the merged fields, in the order
          of [held], and {!row_field}. *)
  finish : (string * Expr.packed) list;
      (** Over [schema]: the keys and the outputs. *)
  folds : bool;
      (** Whether an output is a [var] or [std], whose state cannot take a
          traced batch. *)
}

val row_field : string
(** [row_field] is the field of each row's first input row. *)

val v :
  by:string list -> (string * Expr.packed) list -> Schema.t -> t option
(** [v ~by outputs schema] is the lowering of an aggregate of [outputs] over
    [schema], [None] where its outputs block ({!Talon.Query.aggregate}). *)
