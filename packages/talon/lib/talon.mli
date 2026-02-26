(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Talon - A dataframe library for OCaml

    Talon provides efficient tabular data manipulation with heterogeneous column
    types, inspired by pandas and polars. Built on top of the Nx tensor library,
    it offers type-safe operations with comprehensive null handling.

    Dataframes are immutable collections of named columns with equal length.
    Each column can contain different types of data with explicit null
    semantics:
    - Numeric tensors (via Nx): float32/float64, int32/int64
    - String option arrays with explicit null support
    - Boolean option arrays with explicit null support

    {1 Key Concepts}

    {2 Null Handling}

    Talon provides first-class null semantics with explicit null masks for
    numeric columns, ensuring accurate tracking of missing values:

    {3 Null Representation}

    - {b Numeric columns}: An optional boolean mask tracks null entries
      explicitly. When no mask exists the column has no missing values; numeric
      payloads (including NaN or extremal integers) are treated as regular data.
    - {b String/Boolean columns}: [None] values represent nulls explicitly

    {3 Creating Nullable Columns}

    Use the [_opt] constructors to create columns with explicit null support:
    {[
      (* Nullable numeric columns *)
      Col.float64_opt
        [| Some 1.0; None; Some 3.0 |]
        Col.int32_opt
        [| Some 42l; None; Some 100l |]
        (* String/bool columns preserve None directly *)
        Col.string_opt
        [| Some "hello"; None; Some "world" |]
        Col.bool_opt
        [| Some true; None; Some false |]
    ]}

    {3 Accessing Values with Null Awareness}

    Use option-based accessors to surface nulls as [None]:
    {[
      (* Row-wise option accessors *)
      Row.float64_opt "score" (* Returns None for nulls *) Row.int32_opt
        "count" (* Distinguishes None from Int32.min_int *)
        (* Extract as option arrays *)
        to_float64_options df "score" (* float option array *)
    ]}

    {3 Null Propagation}

    - Operations propagate nulls: [null + x = null]
    - Aggregations skip nulls by default (configurable with [skipna] parameter)
    - Mask-aware aggregations properly exclude masked entries from computations

    {2 Type Safety}

    The library maintains type information through GADTs and provides
    type-specific aggregation modules ([Agg.Float], [Agg.Int], etc.) that ensure
    operations are only applied to compatible column types.

    {2 Performance}

    Operations leverage vectorized Nx tensor computations where possible.
    Row-wise operations use an applicative interface that compiles to efficient
    loops. Use [with_columns_map] for computing multiple columns in a single
    pass.

    {1 Quick Start}

    {[
      open Talon

      (* Create a dataframe from columns *)
      let df =
        create
          [
            ("name", Col.string_list [ "Alice"; "Bob"; "Charlie" ]);
            ("age", Col.int32_list [ 25l; 30l; 35l ]);
            ("score", Col.float64_list [ 85.5; 92.0; 78.5 ]);
            ("active", Col.bool_list [ true; false; true ]);
          ]

      (* Filter rows where age > 25 *)
      let adults =
        filter_by df Row.(map (int32 "age") ~f:(fun age -> age > 25l))

      (* Aggregations - explicit about expected types *)
      let total_score =
        Agg.Float.sum df "score" (* 256.0 - works on any numeric *)

      let avg_age = Agg.Int.mean df "age" (* 30.0 - returns float *)
      let max_name = Agg.String.max df "name" (* Some "Charlie" *)

      (* Column operations preserve dtype *)
      let cumulative =
        Agg.cumsum df "score" (* Returns packed_column with float32 *)

      let age_diff = Agg.diff df "age" () (* Returns packed_column with int32 *)

      (* Extract column as array for external processing *)
      let scores_array = to_float32_array df "score"

      (* Group by a computed key *)
      let by_category =
        group_by df
          Row.(
            map (float32 "score") ~f:(fun s ->
                if s >= 90.0 then "A" else if s >= 80.0 then "B" else "C"))
    ]} *)

type t
(** Type of dataframe.

    Dataframes are immutable tabular data structures with named, typed columns.
    All columns in a dataframe have the same length (number of rows).

    Implementation: Internally uses a list of (name, column) pairs for ordering
    and a hash table for O(1) column lookup by name. *)

type 'a row
(** Type for row-wise computations.

    This abstract type represents a computation that can be applied to each row
    of a dataframe to produce a value of type ['a]. Row computations are lazy
    and only executed when the dataframe is processed.

    Row computations form an applicative functor, allowing composition of
    independent computations from multiple columns. *)

(** {1 Column Operations}

    Columns are the fundamental data containers in Talon dataframes. Each column
    stores homogeneous data with consistent null handling. *)

module Col : sig
  (** Column creation and manipulation for heterogeneous data types.

      Columns are the fundamental building blocks of dataframes, each storing a
      homogeneous array of values with consistent null handling. *)

  type t =
    | P : ('a, 'b) Nx.dtype * ('a, 'b) Nx.t * bool array option -> t
    | S : string option array -> t
    | B : bool option array -> t
        (** Heterogeneous column representation with explicit null support.

            Variants:
            - [P (dtype, tensor, mask)]: Numeric data stored as 1D Nx tensors
              with an optional null mask indicating which entries are missing
            - [S arr]: String data with explicit [None] for nulls
            - [B arr]: Boolean data with explicit [None] for nulls

            Null representation:
            - Float columns: the optional mask determines which rows are null.
              The underlying tensor may contain any float values (including
              NaN), but these are treated as data unless masked out.
            - Integer columns: the optional mask determines null rows. Extreme
              integer values such as [Int32.min_int] remain valid data unless
              explicitly masked.
            - String/Boolean columns: [None] values indicate nulls

            Invariants:
            - Numeric tensors must be 1D
            - All values in a column have the same length
            - Null semantics are preserved across operations

            Performance:
            - Numeric operations leverage vectorized Nx computations
            - String/Boolean operations use standard OCaml array operations *)

  (** {3 From arrays (non-nullable)}

      Create columns from arrays without introducing null masks. Values are
      taken literally; to represent missing data, use the [_opt] constructors
      instead. *)

  val float32 : float array -> t
  (** [float32 arr] creates a float32 column from array.

      The resulting column has no null mask. All values, including [nan], are
      treated as regular data. Use [float32_opt] to create a nullable column.

      Time complexity: O(n) where n is array length. *)

  val float64 : float array -> t
  (** [float64 arr] creates a float64 column from array.

      The resulting column has no null mask. All values, including [nan], are
      treated as regular data. Use [float64_opt] for nullable columns.

      Time complexity: O(n) where n is array length. *)

  val int32 : int32 array -> t
  (** [int32 arr] creates an int32 column from array.

      The resulting column has no null mask. Values equal to [Int32.min_int] are
      treated as ordinary data. Use [int32_opt] to represent missing values.

      Time complexity: O(n) where n is array length. *)

  val int64 : int64 array -> t
  (** [int64 arr] creates an int64 column from array.

      The resulting column has no null mask. Values equal to [Int64.min_int]
      remain ordinary data. Use [int64_opt] to represent missing values.

      Time complexity: O(n) where n is array length. *)

  val bool : bool array -> t
  (** [bool arr] creates a non-nullable boolean column from array.

      All values are wrapped as [Some value], creating a column with no nulls.
      Use [bool_opt] if you need explicit null support.

      Time complexity: O(n) where n is array length. *)

  val string : string array -> t
  (** [string arr] creates a non-nullable string column from array.

      All values are wrapped as [Some value], creating a column with no nulls.
      Use [string_opt] if you need explicit null support.

      Time complexity: O(n) where n is array length. *)

  (** {3 From option arrays (nullable)}

      Create columns from option arrays with explicit null representation.
      Numeric types attach a null mask (while storing placeholder values in the
      tensor), whereas string/bool types preserve the option structure. *)

  val float32_opt : float option array -> t
  (** [float32_opt arr] creates a nullable float32 column.

      [None] values are recorded in the null mask. Placeholder [nan] values are
      stored in the tensor but callers must rely on the mask (via option
      accessors or [Agg] helpers) to detect nulls.

      Example:
      {[
        let col = Col.float32_opt [| Some 1.0; None; Some 3.14 |] in
        assert (Col.null_count col = 1)
      ]}

      Time complexity: O(n) where n is array length. *)

  val float64_opt : float option array -> t
  (** [float64_opt arr] creates a nullable float64 column.

      [None] values are recorded in the null mask. Placeholder [nan] values are
      stored in the tensor but callers must rely on the mask to detect nulls.

      Time complexity: O(n) where n is array length. *)

  val int32_opt : int32 option array -> t
  (** [int32_opt arr] creates a nullable int32 column.

      [None] values are recorded in the null mask. The tensor stores a
      placeholder value ([Int32.min_int]) for efficiency, but the mask is
      authoritative when checking for nulls.

      Example:
      {[
        let col = Col.int32_opt [| Some 42l; None; Some (-1l) |] in
        assert (Col.null_count col = 1)
      ]}

      Time complexity: O(n) where n is array length. *)

  val int64_opt : int64 option array -> t
  (** [int64_opt arr] creates a nullable int64 column.

      [None] values are recorded in the null mask. The tensor stores a
      placeholder value ([Int64.min_int]) for efficiency, but the mask is
      authoritative when checking for nulls.

      Time complexity: O(n) where n is array length. *)

  val bool_opt : bool option array -> t
  (** [bool_opt arr] creates a nullable boolean column.

      Unlike numeric columns, boolean columns preserve the option type directly
      for more precise null semantics.

      Time complexity: O(1) - no array copying required. *)

  val string_opt : string option array -> t
  (** [string_opt arr] creates a nullable string column.

      The option array is used directly without conversion, preserving exact
      null semantics.

      Time complexity: O(1) - no array copying required. *)

  (** {3 From lists}

      Convenience functions for creating columns from lists. Equivalent to
      creating arrays with [Array.of_list] then using the array functions. *)

  val float32_list : float list -> t
  (** [float32_list lst] creates a float32 column from list.

      Time complexity: O(n) where n is list length. *)

  val float64_list : float list -> t
  (** [float64_list lst] creates a float64 column from list.

      Time complexity: O(n) where n is list length. *)

  val int32_list : int32 list -> t
  (** [int32_list lst] creates an int32 column from list.

      Time complexity: O(n) where n is list length. *)

  val int64_list : int64 list -> t
  (** [int64_list lst] creates an int64 column from list.

      Time complexity: O(n) where n is list length. *)

  val bool_list : bool list -> t
  (** [bool_list lst] creates a boolean column from list.

      Time complexity: O(n) where n is list length. *)

  val string_list : string list -> t
  (** [string_list lst] creates a string column from list.

      Time complexity: O(n) where n is list length. *)

  val null_mask : t -> bool array option
  (** [null_mask col] returns the explicit null mask tracked for numeric columns
      constructed via nullable builders.

      Returns [Some mask] when an explicit mask exists, [None] otherwise. *)

  (** {3 From tensors}

      Direct integration with Nx tensors for efficient column creation. *)

  val of_tensor : ('a, 'b) Nx.t -> t
  (** [of_tensor t] creates a column from a 1D tensor.

      The tensor's dtype is preserved in the resulting column. The column is
      treated as non-nullable: existing payload values (including NaNs or
      extremal integers) remain regular data. Use the [_opt] builders to attach
      null masks.

      @raise Invalid_argument if tensor is not 1D.

      Time complexity: O(1) - tensor is used directly without copying. *)

  (** {3 Null handling}

      Functions for detecting and manipulating null values in columns. Null
      semantics vary by column type but are handled consistently. *)

  val has_nulls : t -> bool
  (** [has_nulls col] returns true if column contains any null values.

      Checks for nulls according to column type:
      - Numeric columns: consult the null mask (if present)
      - String/Boolean columns: scan for [None] values

      Time complexity: O(n) in worst case (scans entire column). *)

  val null_count : t -> int
  (** [null_count col] returns the number of null values in column.

      Counts nulls according to column type. More efficient than scanning with
      [has_nulls] if you need the exact count.

      Time complexity: O(n) - must scan entire column. *)

  val drop_nulls : t -> t
  (** [drop_nulls col] returns a new column with null values removed.

      Creates a new column with shorter length containing only non-null values.
      The column type is preserved (numeric columns remain numeric, etc.).

      Example:
      {[
        let col = Col.float32_opt [| Some 1.0; None; Some 3.0 |] in
        let clean = Col.drop_nulls col in
        (* clean now contains [1.0; 3.0] *)
        assert (Col.null_count clean = 0)
      ]}

      Time complexity: O(n) where n is the original column length. *)

  val fill_nulls_float32 : t -> value:float -> t
  (** [fill_nulls_float32 col ~value] replaces null values with the given float
      value.

      Works only on float32 columns. NaN values are treated as nulls and
      replaced with the specified value.

      @param value The replacement value for null entries
      @raise Invalid_argument if column is not float32 type

      Time complexity: O(n) where n is the column length. *)

  val fill_nulls_float64 : t -> value:float -> t
  (** [fill_nulls_float64 col ~value] replaces null values with the given float
      value.

      Works only on float64 columns. NaN values are treated as nulls and
      replaced with the specified value.

      @param value The replacement value for null entries
      @raise Invalid_argument if column is not float64 type

      Time complexity: O(n) where n is the column length. *)

  val fill_nulls_int32 : t -> value:int32 -> t
  (** [fill_nulls_int32 col ~value] replaces null values with the given int32
      value.

      Works only on int32 columns. Int32.min_int values are treated as nulls and
      replaced with the specified value.

      @param value The replacement value for null entries
      @raise Invalid_argument if column is not int32 type

      Time complexity: O(n) where n is the column length. *)

  val fill_nulls_int64 : t -> value:int64 -> t
  (** [fill_nulls_int64 col ~value] replaces null values with the given int64
      value.

      Works only on int64 columns. Int64.min_int values are treated as nulls and
      replaced with the specified value.

      @param value The replacement value for null entries
      @raise Invalid_argument if column is not int64 type

      Time complexity: O(n) where n is the column length. *)

  val fill_nulls_string : t -> value:string -> t
  (** [fill_nulls_string col ~value] replaces null values with the given string
      value.

      Works only on string columns. None values are treated as nulls and
      replaced with the specified value.

      @param value The replacement value for null entries
      @raise Invalid_argument if column is not string type

      Time complexity: O(n) where n is the column length. *)

  val fill_nulls_bool : t -> value:bool -> t
  (** [fill_nulls_bool col ~value] replaces null values with the given boolean
      value.

      Works only on boolean columns. None values are treated as nulls and
      replaced with the specified value.

      @param value The replacement value for null entries
      @raise Invalid_argument if column is not boolean type

      Time complexity: O(n) where n is the column length. *)
end

(** {1 DataFrame Creation}

    Functions for creating dataframes from various data sources. *)

val empty : t
(** [empty] creates an empty dataframe with no rows or columns.

    This is the neutral element for operations like [concat]. Useful as a
    starting point for building dataframes incrementally.

    Example:
    {[
      let df = empty in
      let df' = add_column df "first" (Col.int32 [| 1l; 2l |]) in
      assert (shape df' = (2, 1))
    ]} *)

val create : (string * Col.t) list -> t
(** [create pairs] creates a dataframe from (column_name, column) pairs.

    This is the primary constructor for dataframes. Each pair specifies a column
    name and its data.

    Invariants:
    - Column names must be unique (case-sensitive)
    - All columns must have exactly the same length
    - Numeric columns must be 1D tensors (checked automatically by [Col] module)

    @param pairs List of (column_name, column_data) pairs

    @raise Invalid_argument
      if duplicate column names exist, column lengths differ, or any column has
      invalid structure.

    Example:
    {[
      let df =
        create
          [
            ("name", Col.string [| "Alice"; "Bob" |]);
            ("age", Col.int32 [| 25l; 30l |]);
            ("score", Col.float64 [| 85.5; 92.0 |]);
          ]
      in
      assert (shape df = (2, 3))
    ]} *)

val of_tensors : ?names:string list -> ('a, 'b) Nx.t list -> t
(** [of_tensors ?names tensors] creates dataframe from 1D Nx tensors.

    All tensors must have the same shape and dtype. This is efficient for
    creating dataframes from pre-computed tensor data.

    @param names Column names (default: "col0", "col1", etc.)
    @param tensors List of 1D tensors with identical shapes and dtypes

    @raise Invalid_argument
      if tensors have inconsistent shapes, any tensor is not 1D, names are not
      unique, or wrong number of names provided.

    Example:
    {[
      let t1 = Nx.create Nx.float64 [| 3 |] [| 1.0; 2.0; 3.0 |] in
      let t2 = Nx.create Nx.float64 [| 3 |] [| 4.0; 5.0; 6.0 |] in
      let df = of_tensors [ t1; t2 ] ~names:[ "x"; "y" ] in
      assert (shape df = (3, 2))
    ]} *)

val from_nx : ?names:string list -> ('a, 'b) Nx.t -> t
(** [from_nx ?names tensor] creates dataframe from 2D tensor.

    Each column of the tensor becomes a dataframe column. This is useful for
    converting tensor data from machine learning operations back to tabular
    format.

    @param names Column names (default: "col0", "col1", etc.)
    @param tensor 2D tensor where rows are observations, columns are variables

    @raise Invalid_argument if tensor is not 2D or names are not unique.

    Example:
    {[
      let data =
        Nx.create Nx.float64 [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |]
      in
      let df = from_nx data ~names:[ "x"; "y"; "z" ] in
      (* Result: 2 rows × 3 columns dataframe *)
      assert (shape df = (2, 3))
    ]} *)

(** {1 Shape and Inspection}

    Functions for examining dataframe structure and metadata. *)

val shape : t -> int * int
(** [shape df] returns (num_rows, num_columns).

    This is the fundamental size information for the dataframe.

    Time complexity: O(1) for non-empty dataframes. *)

val num_rows : t -> int
(** [num_rows df] returns number of rows.

    Equivalent to [fst (shape df)] but more convenient when you only need row
    count.

    Time complexity: O(1) for non-empty dataframes. *)

val num_columns : t -> int
(** [num_columns df] returns number of columns.

    Equivalent to [snd (shape df)] but more convenient when you only need column
    count.

    Time complexity: O(1). *)

val column_names : t -> string list
(** [column_names df] returns column names in their current order.

    The order matches the column order for operations like [print] and [to_nx].

    Time complexity: O(k) where k is the number of columns. *)

val column_types :
  t ->
  (string
  * [ `Float32 | `Float64 | `Int32 | `Int64 | `Bool | `String | `Other ])
  list
(** [column_types df] returns column names with their detected types.

    Type detection:
    - [`Float32], [`Float64], [`Int32], [`Int64]: Numeric Nx tensor columns
    - [`Bool]: Boolean option array columns
    - [`String]: String option array columns
    - [`Other]: Any other Nx tensor types (e.g., uint8)

    Useful for programmatic dataframe inspection and type-based operations.

    Time complexity: O(k) where k is the number of columns. *)

val is_empty : t -> bool
(** [is_empty df] returns true if dataframe has no rows.

    Note that a dataframe can have columns but zero rows, which is still
    considered empty by this function.

    Time complexity: O(1). *)

(** {2 Column Selection Utilities} *)

module Cols : sig
  (** Column selection utilities for working with subsets of columns.

      These functions provide convenient ways to select columns by type, name
      patterns, or other criteria. They return lists of column names that can be
      used with functions like [select], [drop_columns], or row-wise
      aggregations. *)

  val numeric : t -> string list
  (** [numeric df] returns all numeric column names.

      Includes float32, float64, int32, and int64 columns. Useful for operations
      that work on any numeric type.

      Example:
      {[
        let nums = Cols.numeric df in
        let stats = select df nums |> describe
      ]} *)

  val float : t -> string list
  (** [float df] returns all floating-point column names.

      Includes float32 and float64 columns only, excluding integer types. *)

  val int : t -> string list
  (** [int df] returns all integer column names.

      Includes int32 and int64 columns only, excluding floating-point types. *)

  val bool : t -> string list
  (** [bool df] returns all boolean column names. *)

  val string : t -> string list
  (** [string df] returns all string column names. *)

  val matching : t -> Re.re -> string list
  (** [matching df regex] returns column names matching the regex pattern.

      Uses the Re library for pattern matching. Useful for selecting columns
      with systematic naming patterns.

      Example:
      {[
        let numeric_cols = Cols.matching df (Re.Posix.compile ".*_[0-9]+") in
        (* Selects columns like "feature_1", "score_99", etc. *)
      ]} *)

  val with_prefix : t -> string -> string list
  (** [with_prefix df prefix] returns column names starting with prefix.

      Example:
      {[
        let temp_cols = Cols.with_prefix df "temp_" in
        (* Selects "temp_morning", "temp_evening", etc. *)
      ]} *)

  val with_suffix : t -> string -> string list
  (** [with_suffix df suffix] returns column names ending with suffix.

      Example:
      {[
        let score_cols = Cols.with_suffix df "_score" in
        (* Selects "math_score", "reading_score", etc. *)
      ]} *)

  val except : t -> string list -> string list
  (** [except df exclude] returns all column names except those in exclude list.

      Useful for selecting "everything but" a few specific columns.

      Example:
      {[
        let features = Cols.except df ["id"; "target"] in
        (* All columns except "id" and "target" *)
      ]} *)

  val select_dtypes :
    t -> [ `Numeric | `Float | `Int | `Bool | `String ] list -> string list
  (** [select_dtypes df types] returns column names of the specified types.

      Type categories:
      - [`Numeric]: All numeric types (float32, float64, int32, int64)
      - [`Float]: Floating-point types (float32, float64)
      - [`Int]: Integer types (int32, int64)
      - [`Bool]: Boolean columns
      - [`String]: String columns

      Example:
      {[
        let numeric_and_bool = Cols.select_dtypes df [`Numeric; `Bool] in
        (* Includes all numeric columns plus boolean columns *)
      ]} *)
end

(** {1 Column Access and Manipulation}

    Functions for working with individual columns within dataframes. *)

val get_column : t -> string -> Col.t option
(** [get_column df name] returns column data or [None].

    Returns the packed column if it exists, [None] otherwise. Use
    [get_column_exn] if you want an exception on missing columns.

    Time complexity: O(1) - uses internal hash table lookup. *)

val get_column_exn : t -> string -> Col.t
(** [get_column_exn df name] returns packed column.

    Use this when you know the column should exist and want to fail fast if it
    doesn't.

    @raise Not_found if column doesn't exist.

    Time complexity: O(1) - uses internal hash table lookup. *)

val to_float32_array : t -> string -> float array option
(** [to_float32_array df name] extracts column as float array if it's float32.

    Returns [Some array] if the column exists and is float32 type, [None]
    otherwise. Null values in the column become NaN in the array.

    @param name Column name to extract

    Example:
    {[
      let df = create [("values", Col.float32 [|1.0; 2.0; Float.nan|])] in
      match to_float32_array df "values" with
      | Some arr -> (* arr = [|1.0; 2.0; nan|] *)
      | None -> (* column doesn't exist or wrong type *)
    ]} *)

val to_float64_array : t -> string -> float array option
(** [to_float64_array df name] extracts column as float array if it's float64.

    Returns [Some array] if the column exists and is float64 type, [None]
    otherwise. Null values become NaN in the array.

    @param name Column name to extract *)

val to_int32_array : t -> string -> int32 array option
(** [to_int32_array df name] extracts column as int32 array if it's int32.

    Returns [Some array] if the column exists and is int32 type, [None]
    otherwise. Null values become [Int32.min_int] in the array.

    @param name Column name to extract *)

val to_int64_array : t -> string -> int64 array option
(** [to_int64_array df name] extracts column as int64 array if it's int64.

    Returns [Some array] if the column exists and is int64 type, [None]
    otherwise. Null values become [Int64.min_int] in the array.

    @param name Column name to extract *)

val to_bool_array : t -> string -> bool array option
(** [to_bool_array df name] extracts column as bool array if it's bool.

    Returns [Some array] if the column exists and is bool type, [None]
    otherwise. Null values become [false] in the array.

    @param name Column name to extract *)

val to_string_array : t -> string -> string array option
(** [to_string_array df name] extracts column as string array if it's string.

    Returns [Some array] if the column exists and is string type, [None]
    otherwise. Null values become empty strings in the array.

    @param name Column name to extract *)

val to_float32_options : t -> string -> float option array option
(** [to_float32_options df name] extracts column as float option array.

    Returns [Some array] if the column exists and is float32 type, [None]
    otherwise. Null values (NaN or masked) become [None] in the array.

    @param name Column name to extract *)

val to_float64_options : t -> string -> float option array option
(** [to_float64_options df name] extracts column as float option array.

    Returns [Some array] if the column exists and is float64 type, [None]
    otherwise. Null values (NaN or masked) become [None] in the array.

    @param name Column name to extract *)

val to_int32_options : t -> string -> int32 option array option
(** [to_int32_options df name] extracts column as int32 option array.

    Returns [Some array] if the column exists and is int32 type, [None]
    otherwise. Null values (Int32.min_int or masked) become [None] in the array.

    @param name Column name to extract *)

val to_int64_options : t -> string -> int64 option array option
(** [to_int64_options df name] extracts column as int64 option array.

    Returns [Some array] if the column exists and is int64 type, [None]
    otherwise. Null values (Int64.min_int or masked) become [None] in the array.

    @param name Column name to extract *)

val to_bool_options : t -> string -> bool option array option
(** [to_bool_options df name] extracts column as bool option array.

    Returns [Some array] if the column exists and is bool type, [None]
    otherwise. Null values are represented as [None] in the array.

    @param name Column name to extract *)

val to_string_options : t -> string -> string option array option
(** [to_string_options df name] extracts column as string option array.

    Returns [Some array] if the column exists and is string type, [None]
    otherwise. Null values are represented as [None] in the array.

    @param name Column name to extract *)

val has_column : t -> string -> bool
(** [has_column df name] returns true if column exists.

    Useful for conditional logic when working with dataframes of unknown
    structure.

    Time complexity: O(1) - uses internal hash table lookup. *)

val add_column : t -> string -> Col.t -> t
(** [add_column df name col] adds or replaces a column.

    If a column with the same name already exists, it is replaced. Otherwise, a
    new column is added to the dataframe.

    @param name Column name
    @param col Column data

    @raise Invalid_argument if column length doesn't match [num_rows df].

    Example:
    {[
      let df = create [("x", Col.int32 [|1l; 2l|])] in
      let df' = add_column df "y" (Col.float64 [|3.0; 4.0|]) in
      (* df' now has both "x" and "y" columns *)
    ]} *)

val drop_column : t -> string -> t
(** [drop_column df name] removes a column.

    Returns the dataframe unchanged if the column doesn't exist (no error). This
    makes it safe to use in pipelines where column existence is uncertain.

    @param name Column name to remove

    Example:
    {[
      let df = create [("x", Col.int32 [|1l; 2l|]); ("y", Col.float64 [|3.0; 4.0|])] in
      let df' = drop_column df "y" in
      (* df' now has only "x" column *)
      let df'' = drop_column df' "nonexistent" in
      (* df'' is unchanged (no error) *)
    ]} *)

val drop_columns : t -> string list -> t
(** [drop_columns df names] removes multiple columns.

    Equivalent to applying [drop_column] for each name in the list. Non-existent
    columns are silently ignored.

    @param names List of column names to remove

    Example:
    {[
      let df = create [("a", Col.int32 [|1l|]); ("b", Col.int32 [|2l|]); ("c", Col.int32 [|3l|])] in
      let df' = drop_columns df ["a"; "c"] in
      (* df' now has only "b" column *)
    ]} *)

val rename_column : t -> old_name:string -> new_name:string -> t
(** [rename_column df ~old_name ~new_name] renames a column.

    Changes the name of an existing column. The column data remains unchanged.

    @param old_name Current column name
    @param new_name Desired column name

    @raise Not_found if [old_name] doesn't exist.
    @raise Invalid_argument if [new_name] already exists as a different column.

    Example:
    {[
      let df = create [("old_name", Col.int32 [|1l; 2l|])] in
      let df' = rename_column df ~old_name:"old_name" ~new_name:"new_name" in
      (* df' has column "new_name" instead of "old_name" *)
    ]} *)

val select : t -> string list -> t
(** [select df names] returns dataframe with only specified columns.

    The resulting dataframe has columns in the order specified by [names]. This
    allows both column filtering and reordering in one operation.

    @param names List of column names to include (in desired order)

    @raise Not_found if any column name doesn't exist.

    Example:
    {[
      let df =
        create
          [
            ("a", Col.int32 [| 1l |]);
            ("b", Col.int32 [| 2l |]);
            ("c", Col.int32 [| 3l |]);
          ]
      in
      let df' = select df [ "c"; "a" ] in
      (* df' has columns "c" and "a" in that order *)
      assert (column_names df' = [ "c"; "a" ])
    ]} *)

val select_loose : t -> string list -> t
(** [select_loose df names] returns dataframe with specified columns that exist.

    Like [select], but silently ignores column names that don't exist. Useful
    when working with dataframes that may have varying column sets.

    @param names List of column names to include if they exist

    Example:
    {[
      let df =
        create [ ("a", Col.int32 [| 1l |]); ("b", Col.int32 [| 2l |]) ]
      in
      let df' = select_loose df [ "a"; "nonexistent"; "b" ] in
      (* df' has columns "a" and "b" only *)
      assert (column_names df' = [ "a"; "b" ])
    ]} *)

val reorder_columns : t -> string list -> t
(** [reorder_columns df names] reorders columns according to the specified list.

    Columns listed in [names] appear first in that order. Any existing columns
    not mentioned in [names] are appended at the end in their original relative
    order.

    @param names List specifying the desired order for some/all columns

    @raise Not_found if any name in the list doesn't exist.

    Example:
    {[
      let df =
        create
          [
            ("a", Col.int32 [| 1l |]);
            ("b", Col.int32 [| 2l |]);
            ("c", Col.int32 [| 3l |]);
          ]
      in
      let df' = reorder_columns df [ "c"; "a" ] in
      (* df' has columns in order: "c", "a", "b" *)
      assert (column_names df' = [ "c"; "a"; "b" ])
    ]} *)

(** {1 Row-wise Operations}

    The Row module provides a functional interface for computations that operate
    across multiple columns within each row. This is the primary way to create
    derived columns and perform row-level filtering. *)

(** Row-wise computations using an applicative interface.

    The [Row] module provides a declarative way to express computations over
    dataframe rows. Rather than imperatively iterating through rows, you compose
    row-wise operations that are executed efficiently in batch.

    The applicative interface allows combining values from multiple columns with
    type safety. Operations are lazy and only executed when the dataframe is
    processed with functions like [filter_by], [map], or [with_column].

    Performance: Row computations compile to efficient loops that process all
    rows in a single pass. Use [with_columns_map] to compute multiple columns
    simultaneously for better cache locality. *)
module Row : sig
  (** {2 Applicative Interface}

      The applicative pattern allows combining independent computations. This is
      more compositional than monadic interfaces and maps naturally to columnar
      data processing. *)

  val return : 'a -> 'a row
  (** [return x] creates a constant computation returning [x] for each row.

      Example:
      {[
        let always_true = Row.return true in
        let filtered = filter_by df always_true (* no-op filter *)
      ]}

      Time complexity: O(1) construction, O(n) when executed over n rows. *)

  val apply : ('a -> 'b) row -> 'a row -> 'b row
  (** [apply f x] applies a function computation to a value computation.

      This is the fundamental applicative operation. Most users will prefer the
      [map] and [map2] convenience functions.

      Example:
      {[
        let add_one = Row.return (fun x -> x + 1) in
        let values = Row.int32 "age" in
        let incremented = Row.apply add_one values
      ]} *)

  val map : 'a row -> f:('a -> 'b) -> 'b row
  (** [map x ~f] maps a function over a computation.

      This is the most common way to transform column values. The function [f]
      is applied to each row's value from the computation [x].

      Example:
      {[
        let ages = Row.int32 "age" in
        let is_adult = Row.map ages ~f:(fun age -> age >= 18l)
      ]} *)

  val map2 : 'a row -> 'b row -> f:('a -> 'b -> 'c) -> 'c row
  (** [map2 x y ~f] combines two computations with a binary function.

      Applies [f] to corresponding values from both computations. This is
      efficient for combining columns element-wise.

      Example:
      {[
        let first_name = Row.string "first_name" in
        let last_name = Row.string "last_name" in
        let full_name = Row.map2 first_name last_name ~f:(fun f l -> f ^ " " ^ l)
      ]} *)

  val map3 : 'a row -> 'b row -> 'c row -> f:('a -> 'b -> 'c -> 'd) -> 'd row
  (** [map3 x y z ~f] combines three computations with a ternary function.

      Useful for operations involving three columns, such as computing weighted
      averages or three-way comparisons. *)

  val both : 'a row -> 'b row -> ('a * 'b) row
  (** [both x y] pairs two computations, creating tuples.

      Equivalent to [map2 x y ~f:(fun a b -> (a, b))] but more explicit about
      the intent to pair values.

      Example:
      {[
        let coords = Row.both (Row.float64 "x") (Row.float64 "y") in
        let distances = Row.map coords ~f:(fun (x, y) -> sqrt (x*.x +. y*.y))
      ]} *)

  (** {2 Column Accessors}

      These functions extract values from named columns with type safety. Each
      accessor verifies the column exists and has the expected type at runtime.
  *)

  val float32 : string -> float row
  (** [float32 name] extracts float32 values from column.

      @raise Not_found if column doesn't exist.
      @raise Invalid_argument if column is not float32 type. *)

  val float64 : string -> float row
  (** [float64 name] extracts float64 values from column.

      @raise Not_found if column doesn't exist.
      @raise Invalid_argument if column is not float64 type. *)

  val int32 : string -> int32 row
  (** [int32 name] extracts int32 values from column.

      @raise Not_found if column doesn't exist.
      @raise Invalid_argument if column is not int32 type. *)

  val int64 : string -> int64 row
  (** [int64 name] extracts int64 values from column.

      @raise Not_found if column doesn't exist.
      @raise Invalid_argument if column is not int64 type. *)

  val string : string -> string row
  (** [string name] extracts string values from column.

      Null values are converted to empty strings for compatibility with
      non-option return type.

      @raise Not_found if column doesn't exist.
      @raise Invalid_argument if column is not string type. *)

  val bool : string -> bool row
  (** [bool name] extracts boolean values from column.

      Null values are converted to [false] for compatibility with non-option
      return type.

      @raise Not_found if column doesn't exist.
      @raise Invalid_argument if column is not boolean type. *)

  val number : string -> float row
  (** [number name] extracts numeric values from column, coercing all numeric
      types (int32/int64/float32/float64) to float.

      This is convenient for generic numeric operations where the exact integer
      vs float distinction doesn't matter. Null values become NaN.

      @raise Not_found if column doesn't exist.
      @raise Invalid_argument if column is not a numeric type. *)

  val numbers : string list -> float row list
  (** [numbers names] creates a list of number accessors for the given column
      names.

      Equivalent to [List.map number names] but more concise for common use
      cases like row-wise aggregations across multiple columns.

      Example:
      {[
        let score_cols = ["math"; "science"; "english"] in
        let scores = Row.numbers score_cols in
        let total = Row.map_list scores ~f:(List.fold_left (+.) 0.)
      ]} *)

  (** {2 Row Information} *)

  val index : int row
  (** [index] returns the current row index (0-based).

      Useful for creating row numbers, conditional logic based on position, or
      debugging row-wise computations.

      Example:
      {[
        let with_row_num =
          with_column df "row_num" Nx.int32 (Row.map Row.index ~f:Int32.of_int)
      ]} *)

  val sequence : 'a row list -> 'a list row
  (** [sequence xs] transforms a list of computations into a computation of a
      list.

      Standard applicative operation for collecting values from multiple
      columns. This is the fundamental operation for working with dynamic lists
      of columns.

      Example:
      {[
        let numeric_cols = Cols.numeric df in
        let values = List.map Row.number numeric_cols in
        let all_values = Row.sequence values in
        let row_sums = Row.map all_values ~f:(List.fold_left (+.) 0.)
      ]} *)

  val all : 'a row list -> 'a list row
  (** [all xs] is an alias for [sequence xs].

      More readable name when the intent is to collect all values from a list of
      computations. *)

  val map_list : 'a row list -> f:('a list -> 'b) -> 'b row
  (** [map_list xs ~f] sequences computations then maps [f] over the resulting
      list.

      Equivalent to [map (sequence xs) ~f] but more convenient. This is the
      standard pattern for applying reductions across multiple columns.

      Example:
      {[
        let score_computations = Row.numbers ["math"; "science"; "english"] in
        let averages = Row.map_list score_computations ~f:(fun scores ->
          List.fold_left (+.) 0. scores /. float (List.length scores))
      ]} *)

  val fold_list : 'a row list -> init:'b -> f:('b -> 'a -> 'b) -> 'b row
  (** [fold_list xs ~init ~f] folds over a list of computations without creating
      an intermediate list.

      More memory-efficient than [map_list] for reductions, especially when
      processing many columns. The fold happens during row iteration rather than
      creating intermediate lists.

      Example:
      {[
        let score_computations = Row.numbers ["q1"; "q2"; "q3"; "q4"] in
        let total_scores = Row.fold_list score_computations ~init:0. ~f:(+.)
      ]} *)

  val float32s : string list -> float row list
  (** Convenience builders to avoid writing [List.map float32 names] etc.

      These functions are particularly useful when you need type-specific
      accessors for multiple columns of the same type. *)

  val float64s : string list -> float row list
  (** [float64s names] creates float64 accessors for all column names. *)

  val int32s : string list -> int32 row list
  (** [int32s names] creates int32 accessors for all column names. *)

  val int64s : string list -> int64 row list
  (** [int64s names] creates int64 accessors for all column names. *)

  val bools : string list -> bool row list
  (** [bools names] creates boolean accessors for all column names. *)

  val strings : string list -> string row list
  (** [strings names] creates string accessors for all column names. *)

  (** {2 Option-based accessors}

      These accessors return [None] for null values instead of using placeholder
      tensor values. Use these when you need to distinguish genuine values from
      missing data. *)

  val float32_opt : string -> float option row
  (** [float32_opt name] extracts float32 values as options from column.

      Returns [None] for null values (as indicated by the mask). Use this
      instead of [float32] when you need to distinguish null values from valid
      data.

      @raise Not_found if column doesn't exist.
      @raise Invalid_argument if column is not float32 type. *)

  val float64_opt : string -> float option row
  (** [float64_opt name] extracts float64 values as options from column.

      Returns [None] for null values (as indicated by the mask).

      @raise Not_found if column doesn't exist.
      @raise Invalid_argument if column is not float64 type. *)

  val int32_opt : string -> int32 option row
  (** [int32_opt name] extracts int32 values as options from column.

      Returns [None] for null values (as indicated by the mask).

      @raise Not_found if column doesn't exist.
      @raise Invalid_argument if column is not int32 type. *)

  val int64_opt : string -> int64 option row
  (** [int64_opt name] extracts int64 values as options from column.

      Returns [None] for null values (as indicated by the mask).

      @raise Not_found if column doesn't exist.
      @raise Invalid_argument if column is not int64 type. *)

  val string_opt : string -> string option row
  (** [string_opt name] extracts string values as options from column.

      Returns [None] for null values. Use this instead of [string] when you need
      to distinguish null strings from empty strings.

      @raise Not_found if column doesn't exist.
      @raise Invalid_argument if column is not string type. *)

  val bool_opt : string -> bool option row
  (** [bool_opt name] extracts boolean values as options from column.

      Returns [None] for null values. Use this instead of [bool] when you need
      to distinguish null values from false.

      @raise Not_found if column doesn't exist.
      @raise Invalid_argument if column is not boolean type. *)

  val float32s_opt : string list -> float option row list
  (** [float32s_opt names] creates float32 option accessors for all column
      names. *)

  val float64s_opt : string list -> float option row list
  (** [float64s_opt names] creates float64 option accessors for all column
      names. *)

  val int32s_opt : string list -> int32 option row list
  (** [int32s_opt names] creates int32 option accessors for all column names. *)

  val int64s_opt : string list -> int64 option row list
  (** [int64s_opt names] creates int64 option accessors for all column names. *)

  val bools_opt : string list -> bool option row list
  (** [bools_opt names] creates bool option accessors for all column names. *)

  val strings_opt : string list -> string option row list
  (** [strings_opt names] creates string option accessors for all column names.
  *)

  (** {2 Row-wise Aggregations}

      Efficient horizontal aggregations across columns within each row. These
      operations are vectorized using Nx operations for performance. *)

  module Agg : sig
    (** Row-wise aggregations using vectorized operations.

        These functions compute aggregations horizontally across columns for
        each row, similar to pandas' [axis=1] operations or Polars' horizontal
        functions. They are much more efficient than using [Row.map_list] for
        common reductions.

        Null handling:
        - Float columns: NaN values are treated as nulls
        - Integer columns: [Int32.min_int] and [Int64.min_int] are treated as
          nulls
        - String/Bool columns: [None] values are treated as nulls
        - When [skipna=true] (default): nulls are excluded from computations
        - When [skipna=false]: any null in a row produces a null result

        Performance: These functions use vectorized Nx operations internally,
        which are significantly faster than row-by-row iteration for large
        datasets. *)

    val sum : ?skipna:bool -> t -> names:string list -> Col.t
    (** [sum ?skipna df ~names] computes row-wise sum across specified columns.

        Uses vectorized Nx operations for efficiency. All specified columns must
        be numeric (float or integer types).

        @param skipna
          If true (default), skip null values. If false, any null in a row makes
          the entire row sum null.

        @raise Invalid_argument if any column is not numeric or doesn't exist.

        Example:
        {[
          let df = create [("a", Col.float64 [|1.; 2.; 3.|]);
                          ("b", Col.float64 [|4.; 5.; 6.|])] in
          let sums = Row.Agg.sum df ~names:["a"; "b"] in
          (* Result: Col.float64 [|5.; 7.; 9.|] *)
        ]} *)

    val mean : ?skipna:bool -> t -> names:string list -> Col.t
    (** [mean ?skipna df ~names] computes row-wise mean across specified
        columns.

        The result is always a float64 column regardless of input types. When
        [skipna=true], the divisor is the count of non-null values per row.

        @param skipna If true (default), exclude nulls from mean calculation.

        @raise Invalid_argument if any column is not numeric or doesn't exist.
    *)

    val min : ?skipna:bool -> t -> names:string list -> Col.t
    (** [min ?skipna df ~names] computes row-wise minimum across specified
        columns.

        The result column preserves the most precise numeric type among inputs
        (e.g., if any input is float64, result is float64).

        @param skipna If true (default), ignore nulls when finding minimum.

        @raise Invalid_argument if any column is not numeric or doesn't exist.
    *)

    val max : ?skipna:bool -> t -> names:string list -> Col.t
    (** [max ?skipna df ~names] computes row-wise maximum across specified
        columns.

        The result column preserves the most precise numeric type among inputs.

        @param skipna If true (default), ignore nulls when finding maximum.

        @raise Invalid_argument if any column is not numeric or doesn't exist.
    *)

    val dot : t -> names:string list -> weights:float array -> Col.t
    (** [dot df ~names ~weights] computes weighted sum (dot product) across
        columns.

        Computes the dot product of row values with the given weights.
        Equivalent to pandas' [df[cols].dot(weights)].

        @param weights Must have the same length as [names].

        @raise Invalid_argument
          if lengths don't match or columns aren't numeric.

        Example:
        {[
          let portfolio_weights = [|0.6; 0.3; 0.1|] in
          let weighted_returns = Row.Agg.dot df
            ~names:["stock_a"; "stock_b"; "stock_c"]
            ~weights:portfolio_weights
        ]} *)

    val all : t -> names:string list -> Col.t
    (** [all df ~names] returns true if all values in the row are true.

        All specified columns must be boolean type. Null values are treated as
        false for the purpose of the "all" operation.

        @raise Invalid_argument if any column is not boolean or doesn't exist.
    *)

    val any : t -> names:string list -> Col.t
    (** [any df ~names] returns true if any value in the row is true.

        All specified columns must be boolean type. Null values are treated as
        false for the purpose of the "any" operation.

        @raise Invalid_argument if any column is not boolean or doesn't exist.
    *)
  end
end

(** {1 Row Filtering and Transformation}

    Functions that operate on entire rows, including filtering, sampling, and
    creating new columns from row computations. *)

val head : ?n:int -> t -> t
(** [head ?n df] returns the first n rows.

    Useful for quick inspection of dataframe contents. If n is larger than the
    number of rows, returns the entire dataframe.

    @param n Number of rows to return (default: 5)

    Time complexity: O(n * k) where k is the number of columns. *)

val tail : ?n:int -> t -> t
(** [tail ?n df] returns the last n rows.

    Useful for quick inspection of dataframe contents. If n is larger than the
    number of rows, returns the entire dataframe.

    @param n Number of rows to return (default: 5)

    Time complexity: O(n * k) where k is the number of columns. *)

val slice : t -> start:int -> stop:int -> t
(** [slice df ~start ~stop] returns rows from start (inclusive) to stop
    (exclusive).

    Uses Python-style slicing semantics. Negative indices are not supported.

    @param start Starting row index (inclusive, 0-based)
    @param stop Ending row index (exclusive, 0-based)

    @raise Invalid_argument
      if [start < 0], [stop < start], or indices are out of bounds.

    Example:
    {[
      let df = create [("id", Col.int32 [|1l; 2l; 3l; 4l; 5l|])] in
      let middle = slice df ~start:1 ~stop:4 in
      (* Result: rows with ids 2, 3, 4 *)
    ]} *)

val sample : ?n:int -> ?frac:float -> ?replace:bool -> ?seed:int -> t -> t
(** [sample ?n ?frac ?replace ?seed df] returns random sample of rows.

    Samples rows randomly from the dataframe. Exactly one of [n] or [frac] must
    be specified.

    @param n Exact number of rows to sample
    @param frac Fraction of rows to sample (between 0.0 and 1.0)
    @param replace If true, sample with replacement (default: false)
    @param seed Random seed for reproducible sampling

    @raise Invalid_argument
      if both [n] and [frac] are specified, or neither, or if [frac] is outside
      [0, 1], or if [n] > rows when [replace=false].

    Example:
    {[
      let df = create [("id", Col.int32 [|1l; 2l; 3l; 4l; 5l|])] in
      let sample1 = sample df ~n:3 ~seed:42 () in (* 3 random rows *)
      let sample2 = sample df ~frac:0.6 () in     (* 60% of rows *)
    ]} *)

val filter : t -> bool array -> t
(** [filter df mask] filters rows where mask is true.

    Creates a new dataframe containing only rows where the corresponding mask
    element is true. The mask array must have exactly the same length as the
    number of dataframe rows.

    @param mask Boolean array indicating which rows to keep

    @raise Invalid_argument if mask length doesn't match [num_rows df].

    Time complexity: O(n * k) where n is rows and k is columns.

    Example:
    {[
      let df = create [("age", Col.int32 [|25l; 30l; 35l|])] in
      let mask = [|true; false; true|] in
      let filtered = filter df mask in
      (* Result contains rows 0 and 2 (age 25 and 35) *)
    ]} *)

val filter_by : t -> bool row -> t
(** [filter_by df pred] filters rows where predicate returns true. *)

val drop_nulls : ?subset:string list -> t -> t
(** [drop_nulls ?subset df] removes rows containing any null values.

    If [subset] is provided, only checks those columns for nulls. Otherwise
    checks all columns. A row is dropped if any value in the checked columns is
    null.

    Null definitions:
    - Float columns: NaN values or entries with mask bit set
    - Integer columns: Int32.min_int/Int64.min_int or entries with mask bit set
    - String/Boolean columns: [None] values

    @param subset Columns to check for nulls (default: all columns)

    Example:
    {[
      let df =
        create
          [
            ("a", Col.float64_opt [| Some 1.0; None; Some 3.0 |]);
            ("b", Col.int32 [| 10l; 20l; 30l |]);
          ]
      in
      let cleaned = drop_nulls df in
      (* Result: 2 rows (indices 0 and 2) *)

      let partial = drop_nulls df ~subset:[ "b" ] in
      (* Result: all 3 rows kept (no nulls in "b") *)
    ]} *)

val fill_missing :
  t ->
  string ->
  with_value:
    [ `Float of float
    | `Int32 of int32
    | `Int64 of int64
    | `String of string
    | `Bool of bool ] ->
  t
(** [fill_missing df col_name ~with_value] replaces null values in a column.

    Creates a new dataframe with null values in the specified column replaced by
    the given value. The value type must match the column type.

    @param col_name Column to fill
    @param with_value Replacement value (must match column type)

    @raise Invalid_argument
      if column doesn't exist or value type doesn't match column type.

    Example:
    {[
      let df = create [ ("x", Col.float64_opt [| Some 1.0; None; Some 3.0 |]) ] in
      let filled = fill_missing df "x" ~with_value:(`Float 0.0) in
      (* "x" now contains [1.0; 0.0; 3.0] *)
    ]} *)

val has_nulls : t -> string -> bool
(** [has_nulls df col_name] checks if a column contains any null values.

    @param col_name Column to check

    @raise Invalid_argument if column doesn't exist.

    Time complexity: O(n) in worst case. *)

val null_count : t -> string -> int
(** [null_count df col_name] returns the number of null values in a column.

    @param col_name Column to count nulls in

    @raise Invalid_argument if column doesn't exist.

    Time complexity: O(n). *)

val drop_duplicates : ?subset:string list -> t -> t
(** [drop_duplicates ?subset df] removes duplicate rows.

    Keeps the first occurrence of each unique row. If [subset] is provided, only
    considers those columns when determining duplicates (but keeps all columns
    in the result).

    @param subset
      Columns to consider for duplicate detection (default: all columns)

    @raise Not_found if any column in [subset] doesn't exist.

    Time complexity: O(n * k) where n is rows and k is columns in subset.

    Example:
    {[
      let df = create [("name", Col.string [|"Alice"; "Bob"; "Alice"|]);
                      ("age", Col.int32 [|25l; 30l; 25l|])] in
      let deduped = drop_duplicates df in
      (* Result has 2 rows: ("Alice", 25) and ("Bob", 30) *)

      let deduped_by_name = drop_duplicates df ~subset:["name"] in
      (* Result has 2 rows: first Alice entry and Bob entry *)
    ]} *)

val concat : axis:[ `Rows | `Columns ] -> t list -> t
(** [concat ~axis dfs] concatenates dataframes along the specified axis.

    Row concatenation ([`Rows]):
    - All dataframes must have the same columns (order doesn't matter)
    - Combines all rows into a single dataframe
    - Column types must be compatible across dataframes

    Column concatenation ([`Columns]):
    - All dataframes must have the same number of rows
    - Combines all columns into a single dataframe
    - Column names must be unique across dataframes

    @param axis Direction of concatenation
    @param dfs List of dataframes to concatenate (must be non-empty)

    @raise Invalid_argument
      if dataframes are incompatible for the chosen axis or if the list is
      empty.

    Example:
    {[
      let df1 = create [("a", Col.int32 [|1l; 2l|])] in
      let df2 = create [("a", Col.int32 [|3l; 4l|])] in
      let rows = concat ~axis:`Rows [df1; df2] in
      (* Result: 4 rows with column "a" *)

      let df3 = create [("b", Col.string [|"x"; "y"|])] in
      let cols = concat ~axis:`Columns [df1; df3] in
      (* Result: 2 rows with columns "a" and "b" *)
    ]} *)

val map : t -> ('a, 'b) Nx.dtype -> 'a row -> ('a, 'b) Nx.t
(** [map df dtype f] maps row-wise computation to create a new tensor.

    Applies the row computation [f] to each row of the dataframe and collects
    the results into a 1D tensor of the specified dtype.

    @param dtype Target Nx dtype for the result tensor
    @param f Row computation that produces values of type compatible with dtype

    Time complexity: O(n * k) where n is rows and k is complexity of computation
    f.

    Example:
    {[
      let df = create [("x", Col.float64 [|1.0; 2.0; 3.0|]);
                      ("y", Col.float64 [|4.0; 5.0; 6.0|])] in
      let sums = map df Nx.float64
        (Row.map2 (Row.float64 "x") (Row.float64 "y") ~f:(+.)) in
      (* sums = tensor [5.0; 7.0; 9.0] *)
    ]} *)

val with_column : t -> string -> ('a, 'b) Nx.dtype -> 'a row -> t
(** [with_column df name dtype f] creates new column from row-wise computation.

    Applies the row computation [f] to each row and adds the results as a new
    column with the specified name and dtype. If a column with that name already
    exists, it is replaced.

    @param name Name for the new column
    @param dtype Nx dtype for the new column
    @param f Row computation that produces values of type compatible with dtype

    Example:
    {[
      let df = create [("x", Col.float64 [|1.0; 2.0|]);
                      ("y", Col.float64 [|3.0; 4.0|])] in
      let df' = with_column df "sum" Nx.float64
        (Row.map2 (Row.float64 "x") (Row.float64 "y") ~f:(+.)) in
      (* df' now has columns "x", "y", and "sum" *)
    ]} *)

val with_columns : t -> (string * Col.t) list -> t
(** [with_columns df cols] adds or replaces multiple columns at once.

    This is an efficient way to add multiple pre-computed columns to a
    dataframe. Similar to Polars' [with_columns] or pandas' [assign]. All
    columns must have the same length as the dataframe.

    @param cols List of (column_name, column_data) pairs

    @raise Invalid_argument if any column length doesn't match the dataframe.

    Example:
    {[
      let df = create [("x", Col.float64 [|1.0; 2.0; 3.0|])] in
      let df' = with_columns df
        [
          ("y", Col.float64 [|4.0; 5.0; 6.0|]);
          ("sum", Col.float64 [|5.0; 7.0; 9.0|]);
        ] in
      (* df' now has columns "x", "y", and "sum" *)
    ]} *)

val with_columns_map : t -> (string * ('a, 'b) Nx.dtype * 'a row) list -> t
(** [with_columns_map df specs] computes multiple row-wise columns in one pass.

    This is more efficient than multiple [with_column] calls because it
    processes all computations in a single iteration over the dataframe rows.
    Similar to pandas' [assign] or Polars' [with_columns].

    Each specification is a tuple of:
    - Column name for the result
    - Nx dtype for the result column
    - Row computation that produces values of that type

    @param specs List of (name, dtype, computation) specifications

    Time complexity: O(n * k) where n is rows and k is total complexity of all
    computations.

    Example:
    {[
      let df = create [("x", Col.float64 [|1.0; 2.0|]);
                      ("y", Col.float64 [|3.0; 4.0|])] in
      let df' = with_columns_map df
        [
          ("sum", Nx.float64,
           Row.map2 (Row.float64 "x") (Row.float64 "y") ~f:(+.));
          ("ratio", Nx.float64,
           Row.map2 (Row.float64 "x") (Row.float64 "y") ~f:(/.));
        ] in
      (* df' has original columns plus "sum" and "ratio" *)
    ]} *)

val iter : t -> unit row -> unit
(** [iter df f] iterates over rows for side effects.

    Applies the row computation [f] to each row but discards the results. Useful
    for side effects like printing or accumulating external state.

    @param f Row computation that produces unit (typically for side effects)

    Example:
    {[
      let df = create [ ("name", Col.string [| "Alice"; "Bob" |]) ] in
      iter df (Row.map (Row.string "name") ~f:(Printf.printf "Hello %s\n"))
      (* Prints: Hello Alice, Hello Bob *)
    ]} *)

val fold : t -> init:'acc -> f:('acc -> 'acc) row -> 'acc
(** [fold df ~init ~f] folds over rows with an accumulator.

    The row computation [f] receives the current accumulator value and should
    return the updated accumulator. This is useful for reductions that depend on
    previous row results.

    @param init Initial accumulator value
    @param f Row computation that takes and returns accumulator type

    Example:
    {[
      let df = create [("value", Col.int32 [|1l; 2l; 3l|])] in
      let sum = fold df ~init:0l ~f:(Row.map (Row.int32 "value") ~f:(Int32.add)) in
      (* sum = 6l *)
    ]} *)

val fold_left :
  t -> init:'acc -> f:('acc -> 'a) row -> ('acc -> 'a -> 'acc) -> 'acc
(** [fold_left df ~init ~f combine] folds with explicit combine function.

    More flexible than [fold] because the row computation [f] can access the
    current accumulator and produce any type, which is then combined with the
    accumulator using the [combine] function.

    @param init Initial accumulator value
    @param f
      Row computation that takes accumulator and produces intermediate result
    @param combine Function to combine accumulator with intermediate result

    Example:
    {[
      let df = create [("x", Col.int32 [|1l; 2l; 3l|])] in
      let product = fold_left df ~init:1l
        ~f:(Row.map (Row.int32 "x") ~f:(fun x -> x))
        ~combine:Int32.mul in
      (* product = 6l *)
    ]} *)

(** {1 Sorting and Grouping}

    Functions for reordering rows and grouping data by key values. *)

val sort : t -> 'a row -> compare:('a -> 'a -> int) -> t
(** [sort df key ~compare] sorts rows by computed key values.

    The key computation is applied to each row to produce sort keys, which are
    then compared using the provided comparison function.

    @param key Row computation that produces sort keys
    @param compare Comparison function for sort keys (< 0, = 0, > 0)

    Time complexity: O(n log n * k) where k is the complexity of key
    computation.

    Example:
    {[
      let df = create [("first", Col.string [|"Bob"; "Alice"|]);
                      ("last", Col.string [|"Smith"; "Jones"|])] in
      let sorted = sort df
        (Row.map2 (Row.string "last") (Row.string "first")
         ~f:(fun l f -> l ^ ", " ^ f))
        ~compare:String.compare in
      (* Sorted by "last, first" *)
    ]} *)

val sort_values : ?ascending:bool -> t -> string -> t
(** [sort_values ?ascending df name] sorts rows by column values.

    Sorts the entire dataframe based on values in the specified column. Null
    values are always sorted to the end regardless of sort direction.

    @param ascending Sort direction (default: true for ascending)
    @param name Column to sort by

    @raise Not_found if column doesn't exist.

    Time complexity: O(n log n) where n is the number of rows.

    Example:
    {[
      let df = create [("age", Col.int32 [|30l; 25l; 35l|]);
                      ("name", Col.string [|"Bob"; "Alice"; "Charlie"|])] in
      let sorted = sort_values df "age" in
      (* Result: Alice (25), Bob (30), Charlie (35) *)

      let desc_sorted = sort_values df "age" ~ascending:false in
      (* Result: Charlie (35), Bob (30), Alice (25) *)
    ]} *)

val group_by : t -> 'key row -> ('key * t) list
(** [group_by df key] groups rows by key values.

    Applies the key computation to each row and groups rows with the same key
    value together. Returns a list of (key_value, sub_dataframe) pairs.

    The order of groups is not guaranteed. Rows within each group maintain their
    original relative order.

    @param key Row computation that produces grouping keys

    Time complexity: O(n * k) where n is rows and k is key computation
    complexity.

    Example:
    {[
      let df = create [("age", Col.int32 [|25l; 30l; 25l; 35l|]);
                      ("name", Col.string [|"Alice"; "Bob"; "Charlie"; "Dave"|])] in
      let age_groups = group_by df (Row.int32 "age") in
      (* Result: [(25l, df_with_alice_charlie); (30l, df_with_bob); (35l, df_with_dave)] *)

      let adult_groups = group_by df
        (Row.map (Row.int32 "age") ~f:(fun age -> age >= 30l)) in
      (* Result: [(false, young_people_df); (true, adults_df)] *)
    ]} *)

val group_by_column : t -> string -> (Col.t * t) list
(** [group_by_column df name] groups rows by values in the specified column.

    This is a convenience function equivalent to [group_by] with appropriate
    column accessor. Returns (group_key_column, sub_dataframe) pairs where the
    key column contains the single unique value for that group.

    @param name Column to group by

    @raise Not_found if column doesn't exist.

    Example:
    {[
      let df = create [("category", Col.string [|"A"; "B"; "A"; "C"|]);
                      ("value", Col.int32 [|1l; 2l; 3l; 4l|])] in
      let groups = group_by_column df "category" in
      (* Result: groups for "A" (rows 0,2), "B" (row 1), "C" (row 3) *)
    ]} *)

(** {1 Aggregations and Column Transformations}

    The Agg module provides efficient column-wise aggregations and
    transformations. Operations are organized by data type for type safety and
    performance. *)

module Agg : sig
  (** Column-wise aggregation and transformation operations.

      This module provides efficient aggregations that operate on entire
      columns, producing scalar results or new columns. All operations preserve
      type safety through dedicated submodules for different data types.

      Performance: Operations leverage vectorized Nx computations where
      possible, providing significant speedups over manual iteration. Use
      [cumsum], [diff], and similar functions for efficient column
      transformations. *)

  (** {2 Type-specific aggregations}

      Each submodule ensures operations are only applied to compatible column
      types, preventing runtime type errors and providing clear semantics. *)

  (** Float aggregations - work on any numeric column (int or float types).

      Values are coerced to float for computation. All functions in this module
      will accept int32, int64, float32, or float64 columns and return float
      results.

      @raise Invalid_argument if column is not numeric. *)
  module Float : sig
    val sum : t -> string -> float
    (** [sum df name] returns the sum as float.

        Works on any numeric column type (int32, int64, float32, float64). Null
        values are excluded from the sum calculation.

        Time complexity: O(n) where n is the number of rows.

        @raise Invalid_argument if column is not numeric or doesn't exist. *)

    val mean : t -> string -> float
    (** [mean df name] returns the arithmetic mean.

        Computes sum divided by count of non-null values. Returns NaN if all
        values are null or the column is empty.

        Time complexity: O(n) where n is the number of rows. *)

    val std : t -> string -> float
    (** [std df name] returns the population standard deviation.

        Computes standard deviation over non-null values, dividing by n. Returns
        NaN if no non-null values exist.

        Time complexity: O(n) - requires two passes over the data. *)

    val var : t -> string -> float
    (** [var df name] returns the population variance.

        Computes variance over non-null values, dividing by n. The standard
        deviation is the square root of this value.

        Time complexity: O(n) - requires two passes over the data. *)

    val min : t -> string -> float option
    (** [min df name] returns minimum value, [None] if empty or all nulls.

        Null values are ignored during comparison.

        Time complexity: O(n) where n is the number of rows. *)

    val max : t -> string -> float option
    (** [max df name] returns maximum value, [None] if empty or all nulls.

        Null values are ignored during comparison.

        Time complexity: O(n) where n is the number of rows. *)

    val median : t -> string -> float
    (** [median df name] returns the median (50th percentile).

        For even-length arrays, returns the average of the two middle values.
        Null values are excluded before sorting.

        Time complexity: O(n log n) due to sorting requirement. *)

    val quantile : t -> string -> q:float -> float
    (** [quantile df name ~q] returns the q-th quantile where 0 <= q <= 1.

        Uses linear interpolation between data points. q=0.5 gives the median,
        q=0.25 gives the first quartile, etc.

        @param q Quantile level between 0.0 and 1.0 inclusive.

        @raise Invalid_argument if q is outside [0, 1].

        Time complexity: O(n log n) due to sorting requirement. *)
  end

  (** Integer aggregations - work on any integer column type.

      Operations in this module accept int32 and int64 columns only. Results
      preserve integer semantics where possible (sum returns int64, mean returns
      float since averages are often fractional).

      @raise Invalid_argument if column is not an integer type. *)
  module Int : sig
    val sum : t -> string -> int64
    (** [sum df name] returns sum as int64.

        Works on int32 and int64 columns. Uses int64 to avoid overflow issues
        that could occur with int32 sums. Null values (as indicated by the
        column mask) are excluded from the sum.

        Time complexity: O(n) where n is the number of rows. *)

    val min : t -> string -> int64 option
    (** [min df name] returns minimum value as int64, [None] if empty or all
        nulls.

        Sentinel null values (Int32.min_int, Int64.min_int) are excluded from
        comparison. Result is converted to int64 for consistency.

        Time complexity: O(n) where n is the number of rows. *)

    val max : t -> string -> int64 option
    (** [max df name] returns maximum value as int64, [None] if empty or all
        nulls.

        Sentinel null values are excluded from comparison.

        Time complexity: O(n) where n is the number of rows. *)

    val mean : t -> string -> float
    (** [mean df name] returns mean as float.

        Since the mean of integers is often fractional, the result is always a
        float regardless of input type. Null values are excluded from
        calculation.

        Time complexity: O(n) where n is the number of rows. *)
  end

  (** String aggregations - work on string columns only.

      Operations in this module work exclusively with string columns. Null
      values ([None] in string option arrays) are handled consistently across
      all functions.

      @raise Invalid_argument if column is not string type. *)
  module String : sig
    val min : t -> string -> string option
    (** [min df name] returns lexicographically smallest string, [None] if
        empty.

        Uses OCaml's string comparison (which compares byte values). Null values
        are excluded from comparison.

        Time complexity: O(n * m) where n is rows and m is average string
        length. *)

    val max : t -> string -> string option
    (** [max df name] returns lexicographically largest string, [None] if empty.

        Uses OCaml's string comparison. Null values are excluded from
        comparison.

        Time complexity: O(n * m) where n is rows and m is average string
        length. *)

    val concat : t -> string -> ?sep:string -> unit -> string
    (** [concat df name ?sep ()] concatenates all non-null strings with
        separator.

        @param sep Separator between strings (default is empty string).

        Null values are skipped during concatenation. If all values are null,
        returns empty string.

        Time complexity: O(n * m) where n is rows and m is average string
        length. *)

    val unique : t -> string -> string array
    (** [unique df name] returns array of unique non-null values.

        The order of unique values is not guaranteed. Null values are excluded
        from the result.

        Time complexity: O(n * m) where n is rows and m is average string
        length. *)

    val nunique : t -> string -> int
    (** [nunique df name] returns count of unique non-null values.

        Null values are not counted towards the unique count.

        Time complexity: O(n * m) where n is rows and m is average string
        length. *)

    val mode : t -> string -> string option
    (** [mode df name] returns most frequent non-null value, [None] if empty.

        If multiple values are tied for most frequent, returns one of them (the
        choice is implementation-dependent). Null values are excluded.

        Time complexity: O(n * m) where n is rows and m is average string
        length. *)
  end

  (** Boolean aggregations - work on boolean columns only.

      Operations in this module work exclusively with boolean columns. Null
      values ([None] in bool option arrays) are handled consistently.

      @raise Invalid_argument if column is not boolean type. *)
  module Bool : sig
    val all : t -> string -> bool
    (** [all df name] returns true if all non-null values are true.

        Returns true for empty columns or columns with only null values. This
        follows the mathematical convention that universal quantification over
        an empty set is true.

        Time complexity: O(n) where n is the number of rows. *)

    val any : t -> string -> bool
    (** [any df name] returns true if any non-null value is true.

        Returns false for empty columns or columns with only null/false values.

        Time complexity: O(n) in worst case, but often faster due to
        short-circuiting. *)

    val sum : t -> string -> int
    (** [sum df name] returns count of true values.

        Treats true as 1 and false as 0, then sums. Null values are excluded
        from the count.

        Time complexity: O(n) where n is the number of rows. *)

    val mean : t -> string -> float
    (** [mean df name] returns proportion of true values among non-null values.

        Equivalent to (count of true values) / (count of non-null values).
        Returns NaN if all values are null.

        Time complexity: O(n) where n is the number of rows. *)
  end

  (** {2 Generic aggregations}

      These functions work on any column type and provide information about the
      data structure rather than mathematical operations. *)

  val count : t -> string -> int
  (** [count df name] returns number of non-null values.

      Null definition varies by column type:
      - Float columns: NaN values
      - Integer columns: Int32.min_int, Int64.min_int sentinel values
      - String/Bool columns: [None] values

      This is the complement of [null_count] from the [Col] module.

      Time complexity: O(n) where n is the number of rows. *)

  val nunique : t -> string -> int
  (** [nunique df name] returns count of unique non-null values for any column
      type.

      Works with any column type. Null values are excluded from the unique
      count. For large datasets, this operation may use significant memory to
      track unique values.

      Time complexity: O(n) for simple types, O(n * m) for strings where m is
      average length. *)

  val value_counts : t -> string -> Col.t * int array
  (** [value_counts df name] returns unique non-null values and their counts.

      Returns a tuple of (unique_values_column, counts_array) where the arrays
      have the same length and corresponding indices match. Useful for frequency
      analysis and building histograms.

      The order of values is not guaranteed.

      Time complexity: O(n) for simple types, O(n * m) for strings. *)

  val is_null : t -> string -> bool array
  (** [is_null df name] returns boolean array where true indicates null values.

      Null definition varies by column type:
      - Float columns: NaN values
      - Integer columns: Int32.min_int, Int64.min_int sentinel values
      - String/Bool columns: [None] values

      Useful for conditional operations and null-aware filtering.

      Time complexity: O(n) where n is the number of rows. *)

  (** {2 Column transformations}

      These operations return new columns, preserving the input column's dtype
      where possible. They are efficient alternatives to row-wise computations
      for common column transformations. *)

  val cumsum : t -> string -> Col.t
  (** [cumsum df name] returns cumulative sum preserving column dtype.

      Computes running total from first row to current row. Null values are
      treated as 0 for the cumulative operation but preserved in the output
      (i.e., null + value = null in the result).

      The result column has the same dtype as the input column.

      @raise Invalid_argument if column is not numeric.

      Time complexity: O(n) where n is the number of rows. *)

  val cumprod : t -> string -> Col.t
  (** [cumprod df name] returns cumulative product preserving column dtype.

      Computes running product from first row to current row. Null values
      propagate through the computation (null * value = null).

      @raise Invalid_argument if column is not numeric.

      Time complexity: O(n) where n is the number of rows. *)

  val diff : t -> string -> ?periods:int -> unit -> Col.t
  (** [diff df name ?periods ()] returns difference between elements.

      Computes [value[i] - value[i-periods]] for each element. The first
      [periods] elements will be null since there are no previous values.

      @param periods Number of periods to shift for difference (default 1).

      @raise Invalid_argument if column is not numeric.

      Time complexity: O(n) where n is the number of rows. *)

  val pct_change : t -> string -> ?periods:int -> unit -> Col.t
  (** [pct_change df name ?periods ()] returns percentage change between
      elements.

      Computes [(value[i] - value[i-periods]) / value[i-periods]] for each
      element. The first [periods] elements will be null. Division by zero
      produces null.

      @param periods Number of periods to shift for comparison (default 1).

      @raise Invalid_argument if column is not numeric.

      Time complexity: O(n) where n is the number of rows. *)

  val shift : t -> string -> periods:int -> Col.t
  (** [shift df name ~periods] shifts values by periods.

      Positive periods shift forward (values move down, nulls fill the top).
      Negative periods shift backward (values move up, nulls fill the bottom).

      @param periods
        Number of positions to shift. Positive values shift forward, negative
        values shift backward.

      Time complexity: O(n) where n is the number of rows. *)

  val fillna : t -> string -> value:Col.t -> Col.t
  (** [fillna df name ~value] fills null/missing values with provided value.

      The [value] column must either:
      - Have exactly one element (broadcast to all null positions)
      - Have the same length as the target column (element-wise replacement)

      The value column must have the same type as the target column.

      @param value Column containing replacement values for nulls.

      @raise Invalid_argument
        if value type doesn't match column type or if value column has
        incompatible length.

      Time complexity: O(n) where n is the number of rows. *)
end

(** {1 Joins and Merges}

    Join operations combine dataframes based on shared key values. Talon
    provides SQL-style joins with explicit null handling semantics. *)

val join :
  t ->
  t ->
  on:string ->
  how:[ `Inner | `Left | `Right | `Outer ] ->
  ?suffixes:string * string ->
  unit ->
  t
(** [join df1 df2 ~on ~how ?suffixes ()] joins two dataframes on a common
    column.

    Join types:
    - [`Inner]: Returns only rows where key exists in both dataframes
    - [`Left]: Returns all rows from df1, null-filled for missing df2 rows
    - [`Right]: Returns all rows from df2, null-filled for missing df1 rows
    - [`Outer]: Returns all rows from both dataframes, null-filled where missing

    Null semantics:
    - Null keys never match other null keys (null != null in join logic)
    - Inner joins exclude rows with null keys entirely
    - Outer joins preserve null key rows but don't match them to anything
    - Unmatched rows get null values for all columns from the other dataframe

    Column naming:
    - Common key column appears once in result
    - Duplicate column names get suffixes (default: "_x" for df1, "_y" for df2)
    - Use [suffixes] parameter to customize the suffixes

    @param on Column name that must exist in both dataframes
    @param how Type of join to perform
    @param suffixes Tuple of (left_suffix, right_suffix) for duplicate columns

    @raise Not_found if the [on] column doesn't exist in either dataframe.
    @raise Invalid_argument if the [on] columns have incompatible types.

    Example:
    {[
      let customers = create [("id", Col.int32 [|1l; 2l; 3l|]);
                             ("name", Col.string [|"Alice"; "Bob"; "Charlie"|])] in
      let orders = create [("id", Col.int32 [|1l; 1l; 2l|]);
                          ("amount", Col.float64 [|100.; 200.; 150.|])] in
      let result = join customers orders ~on:"id" ~how:`Inner () in
      (* Result has customers with their orders, Alice appears twice *)
    ]} *)

val merge :
  t ->
  t ->
  left_on:string ->
  right_on:string ->
  how:[ `Inner | `Left | `Right | `Outer ] ->
  ?suffixes:string * string ->
  unit ->
  t
(** [merge df1 df2 ~left_on ~right_on ~how ?suffixes ()] merges dataframes on
    different column names.

    This is identical to [join] except it allows using different column names
    from each dataframe as the join keys. The columns must have compatible types
    for comparison.

    The result contains both key columns (with suffixes if they have the same
    name).

    @param left_on Column name from the left dataframe (df1)
    @param right_on Column name from the right dataframe (df2)
    @param how Type of join to perform (same semantics as [join])
    @param suffixes Tuple of (left_suffix, right_suffix) for duplicate columns

    @raise Not_found if either column doesn't exist in its respective dataframe.
    @raise Invalid_argument if the key columns have incompatible types.

    Example:
    {[
      let products = create [("product_id", Col.int32 [|1l; 2l; 3l|]);
                            ("name", Col.string [|"Widget"; "Gadget"; "Tool"|])] in
      let sales = create [("item_id", Col.int32 [|1l; 1l; 2l|]);
                         ("quantity", Col.int32 [|10l; 5l; 3l|])] in
      let result = merge products sales
                     ~left_on:"product_id" ~right_on:"item_id"
                     ~how:`Inner () in
      (* Result links products to sales via the id mapping *)
    ]} *)

(** {1 Pivot and Reshape}

    Reshape operations transform dataframe structure between wide and long
    formats. *)

val pivot :
  t ->
  index:string ->
  columns:string ->
  values:string ->
  ?agg_func:[ `Sum | `Mean | `Count | `Min | `Max ] ->
  unit ->
  t
(** [pivot df ~index ~columns ~values ?agg_func ()] creates a pivot table.

    Transforms data from long format to wide format by: 1. Grouping by the
    [index] column (becomes row identifiers) 2. Using unique values from
    [columns] as new column names 3. Filling the table with [values], aggregated
    by [agg_func] if needed

    @param index Column to use as row identifiers in the pivot table
    @param columns Column whose unique values become new column names
    @param values Column containing the data to fill the pivot table
    @param agg_func
      Aggregation function for handling duplicate combinations (default: [`Sum]
      for numeric, [`Count] for others)

    @raise Not_found if any specified column doesn't exist.
    @raise Invalid_argument if [values] column is incompatible with [agg_func].

    Example:
    {[
      let sales = create [("date", Col.string [|"2023-01"; "2023-01"; "2023-02"|]);
                         ("product", Col.string [|"A"; "B"; "A"|]);
                         ("amount", Col.float64 [|100.; 200.; 150.|])] in
      let pivot_table = pivot sales ~index:"date" ~columns:"product"
                             ~values:"amount" ~agg_func:`Sum () in
      (* Result: dates as rows, products as columns, amounts as values *)
    ]} *)

val melt :
  t ->
  ?id_vars:string list ->
  ?value_vars:string list ->
  ?var_name:string ->
  ?value_name:string ->
  unit ->
  t
(** [melt df ?id_vars ?value_vars ?var_name ?value_name ()] unpivots dataframe.

    Transforms data from wide format to long format by: 1. Keeping [id_vars]
    columns as identifiers (repeated for each melted row) 2. Converting
    [value_vars] column names into a single "variable" column 3. Converting
    [value_vars] values into a single "value" column

    @param id_vars
      Columns to keep as identifiers (default: all non-[value_vars])
    @param value_vars Columns to melt (default: all non-[id_vars])
    @param var_name Name for the new "variable" column (default: "variable")
    @param value_name Name for the new "value" column (default: "value")

    @raise Not_found if any specified column doesn't exist.
    @raise Invalid_argument
      if [id_vars] and [value_vars] overlap or cover all columns.

    Example:
    {[
      let wide = create [("id", Col.int32 [|1l; 2l|]);
                        ("A", Col.float64 [|1.0; 3.0|]);
                        ("B", Col.float64 [|2.0; 4.0|])] in
      let long = melt wide ~id_vars:["id"] ~value_vars:["A"; "B"] () in
      (* Result: 4 rows with id, variable ("A" or "B"), and value columns *)
    ]} *)

(** {1 Conversion}

    Functions for converting dataframes to and from other data structures. *)

val to_nx : t -> (float, Bigarray.float32_elt) Nx.t
(** [to_nx df] converts all numeric columns to a 2D float32 tensor.

    Creates a tensor where:
    - Rows correspond to dataframe rows
    - Columns correspond to numeric dataframe columns (in order)
    - All numeric types are cast to float32
    - Null values become NaN in the result tensor

    Only numeric columns (int32, int64, float32, float64) are included. String
    and boolean columns are ignored.

    @raise Invalid_argument if the dataframe contains no numeric columns.

    Example:
    {[
      let df =
        create
          [
            ("a", Col.int32 [| 1l; 2l |]);
            ("b", Col.float64 [| 3.0; 4.0 |]);
            ("c", Col.string [| "x"; "y" |]);
          ]
      in
      let tensor = to_nx df in
      (* Result: 2x2 float32 tensor with values [[1.0, 3.0], [2.0, 4.0]] *)
      assert (Nx.shape tensor = [| 2; 2 |])
    ]} *)

(** {1 Display and Inspection}

    Functions for examining and debugging dataframe contents. *)

val print : ?max_rows:int -> ?max_cols:int -> t -> unit
(** [print ?max_rows ?max_cols df] pretty-prints dataframe in tabular format.

    Displays a formatted table showing column names and values. Large dataframes
    are truncated for readability.

    @param max_rows Maximum number of rows to display (default: 10)
    @param max_cols Maximum number of columns to display (default: 10)

    Truncated output shows "..." to indicate hidden rows/columns.

    Example output:
    {v
       name      age    score
    0  Alice     25     85.5
    1  Bob       30     92.0
    2  Charlie   35     78.5
    v} *)

val describe : t -> t
(** [describe df] returns summary statistics for numeric columns.

    Creates a new dataframe with statistical summaries as rows:
    - count: number of non-null values
    - mean: arithmetic mean
    - std: standard deviation
    - min: minimum value
    - 25%: first quartile
    - 50%: median
    - 75%: third quartile
    - max: maximum value

    Only numeric columns are included in the result. String and boolean columns
    are ignored.

    Time complexity: O(n * k * log n) where n is rows and k is numeric columns
    (due to quantile calculations). *)

val cast_column : t -> string -> ('a, 'b) Nx.dtype -> t
(** [cast_column df name dtype] converts column to specified numeric dtype.

    Creates a new dataframe with the specified column converted to the target
    numeric type. Only works for numeric columns and numeric target types.

    Type conversions:
    - int32 ↔ int64: direct conversion
    - int32/int64 → float32/float64: exact for small integers
    - float32/float64 → int32/int64: truncation (may lose precision)
    - float32 ↔ float64: precision change

    Null values are preserved through the conversion.

    @param name Column to convert
    @param dtype Target Nx dtype (must be numeric)

    @raise Not_found if column doesn't exist.
    @raise Invalid_argument if source column is not numeric or conversion fails.

    Example:
    {[
      let df = create [("values", Col.int32 [|1l; 2l; 3l|])] in
      let df' = cast_column df "values" Nx.float64 in
      (* "values" column is now float64 type *)
    ]} *)

val info : t -> unit
(** [info df] prints detailed dataframe information to stdout.

    Displays:
    - Dataframe shape (rows × columns)
    - Column names and types
    - Null value counts per column
    - Memory usage estimates

    Useful for debugging and understanding dataframe structure.

    Example output:
    {v
    Dataframe Info:
    Shape: (1000, 3)
    Columns:
      name (string): 0 nulls
      age (int32): 5 nulls  
      score (float64): 2 nulls
    Memory usage: ~24KB
    v} *)
