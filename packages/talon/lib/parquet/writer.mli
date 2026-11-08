(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Parquet's writer: the talon types as a Parquet schema, shredding a column
    into its levels and values, the pages and statistics of a column chunk,
    row groups, and the footer. {!Talon_parquet} states what it writes. *)

val written : Talon.Type.any -> Talon.Type.any
(** [written t] is the type a column of type [t] writes as, and reads back as
    when it is flat: [t] with its extensions replaced by their storage, its
    categoricals by [string], its seconds by milliseconds, its durations by
    [int64] and its tensors by nested lists.

    Raises [Invalid_argument] if [t] holds a record without fields or a tensor
    of a dtype with no talon scalar type. *)

val schema_elements : (string * Talon.Type.any) list -> Meta.element array
(** [schema_elements columns] is the Parquet schema of [columns], root first. *)

val encode :
  codec:Meta.codec ->
  row_group:int ->
  columns:(string * Talon.Type.any) list ->
  Talon.Query.t ->
  Bytesrw.Bytes.Writer.t ->
  (unit, Talon.Error.t) result
(** [encode ~codec ~row_group ~columns q w] writes the rows of [q], whose schema
    is [columns], as a Parquet file to [w], in row groups of [row_group] rows.

    Raises [Invalid_argument] if [row_group] is not positive. *)
