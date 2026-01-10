(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** CSV I/O operations for Talon dataframes *)

val read :
  ?sep:char ->
  ?header:bool ->
  ?na_values:string list ->
  ?dtype_spec:
    (string * [ `Float32 | `Float64 | `Int32 | `Int64 | `Bool | `String ]) list ->
  string ->
  Talon.t
(** [read ?sep ?header ?na_values ?dtype_spec file] reads CSV file.
    - sep: delimiter (default ',')
    - header: first row contains column names (default true)
    - na_values: strings to interpret as null (default
      ["", "NA", "N/A", "null", "NULL"])
    - dtype_spec: column types (auto-detected if not provided)

    Null handling:
    - Empty strings and na_values become None for strings, NaN for floats
    - Invalid numeric values become NaN/None
    - Auto-detection treats columns with nulls as nullable *)

val write :
  ?sep:char -> ?header:bool -> ?na_repr:string -> Talon.t -> string -> unit
(** [write ?sep ?header ?na_repr df file] writes to CSV.
    - sep: delimiter (default ',')
    - header: write column names as first row (default true)
    - na_repr: string representation of nulls (default "")

    Null values are written as na_repr (empty string by default). *)

val to_string :
  ?sep:char -> ?header:bool -> ?na_repr:string -> Talon.t -> string
(** [to_string ?sep ?header ?na_repr df] converts dataframe to CSV string. *)

val from_string :
  ?sep:char ->
  ?header:bool ->
  ?na_values:string list ->
  ?dtype_spec:
    (string * [ `Float32 | `Float64 | `Int32 | `Int64 | `Bool | `String ]) list ->
  string ->
  Talon.t
(** [from_string ?sep ?header ?na_values ?dtype_spec csv_string] parses CSV
    string. *)
