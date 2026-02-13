(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** CSV codec for Talon dataframes.

    {[
      (* From string *)
      let df = Talon_csv.of_string csv_text

      (* From file (streaming) *)
      let df = Talon_csv.read "data.csv"

      (* To file (streaming) *)
      Talon_csv.write "out.csv" df
    ]} *)

type dtype_spec =
  (string * [ `Float32 | `Float64 | `Int32 | `Int64 | `Bool | `String ]) list
(** Column type specifications. Columns not listed are auto-detected. *)

val of_string :
  ?sep:char ->
  ?names:string list ->
  ?na_values:string list ->
  ?dtype_spec:dtype_spec ->
  string ->
  Talon.t
(** [of_string s] parses CSV text into a dataframe.
    The first row is used as column names unless [names] is provided,
    in which case all rows are treated as data.

    @param sep delimiter character (default [','])
    @param names explicit column names; when given, all rows are data
    @param na_values strings treated as null (default
    [\[""; "NA"; "N/A"; "null"; "NULL"; "nan"; "NaN"\]])
    @param dtype_spec explicit column types; unspecified columns are
    auto-detected *)

val to_string : ?sep:char -> ?na_repr:string -> Talon.t -> string
(** [to_string df] serializes a dataframe to CSV text.
    The first row of the output is the column names.

    @param sep delimiter character (default [','])
    @param na_repr string for null values (default [""]) *)

val read :
  ?sep:char ->
  ?names:string list ->
  ?na_values:string list ->
  ?dtype_spec:dtype_spec ->
  string ->
  Talon.t
(** [read path] reads a CSV file into a dataframe, streaming line by line.

    @param sep delimiter character (default [','])
    @param names explicit column names; when given, all rows are data
    @param na_values strings treated as null
    @param dtype_spec explicit column types *)

val write : ?sep:char -> ?na_repr:string -> string -> Talon.t -> unit
(** [write path df] writes a dataframe to a CSV file, streaming row by row.

    @param sep delimiter character (default [','])
    @param na_repr string for null values (default [""]) *)
