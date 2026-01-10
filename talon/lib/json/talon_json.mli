(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** JSON I/O operations for Talon dataframes *)

val to_string : ?orient:[ `Records | `Columns ] -> Talon.t -> string
(** [to_string ?orient df] converts to JSON string.
    - orient: `Records for row-oriented, `Columns for column-oriented (default
      `Records)

    Null values are represented as JSON null. *)

val from_string :
  ?orient:[ `Records | `Columns ] ->
  ?dtype_spec:
    (string * [ `Float32 | `Float64 | `Int32 | `Int64 | `Bool | `String ]) list ->
  string ->
  Talon.t
(** [from_string ?orient json] creates from JSON string. JSON null values become
    None/NaN appropriately. *)

val to_file : ?orient:[ `Records | `Columns ] -> Talon.t -> string -> unit
(** [to_file ?orient df file] writes dataframe to JSON file. *)

val from_file :
  ?orient:[ `Records | `Columns ] ->
  ?dtype_spec:
    (string * [ `Float32 | `Float64 | `Int32 | `Int64 | `Bool | `String ]) list ->
  string ->
  Talon.t
(** [from_file ?orient file] reads dataframe from JSON file. *)
