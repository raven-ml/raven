(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Dataframe library for tabular data manipulation.

    Dataframes are immutable collections of named, typed columns with equal
    length. Columns can hold numeric tensors (via {!Nx}), strings, or booleans,
    each with explicit null semantics. *)

type t
(** The type for dataframes.

    Dataframes are immutable tabular data structures with named, typed columns.
    All columns in a dataframe have the same length. *)

type 'a row
(** The type for row-wise computations producing values of type ['a].

    Row computations form an applicative functor, allowing composition of
    independent computations from multiple columns. *)

(** {1:columns Columns} *)

module Col : sig
  (** Column creation and manipulation.

      Columns are the building blocks of dataframes, each storing a homogeneous
      sequence of values with consistent null handling. *)

  type t
  (** The type for columns.

      Columns store homogeneous data with consistent null handling:
      - Numeric data backed by 1D {!Nx} tensors with an optional null mask.
      - String data as [string option array].
      - Boolean data as [bool option array]. *)

  (** {2:generic_constructors Generic constructors} *)

  val numeric : ('a, 'b) Nx.dtype -> 'a array -> t
  (** [numeric dtype arr] is a numeric column from [arr] with dtype [dtype]. *)

  val numeric_opt : ('a, 'b) Nx.dtype -> 'a option array -> t
  (** [numeric_opt dtype arr] is a nullable numeric column from [arr] with dtype
      [dtype]. [None] values are recorded in the null mask. *)

  (** {2:non_nullable From arrays (non-nullable)}

      Create columns from arrays without introducing null masks. Values are
      taken literally; to represent missing data, use the [_opt] constructors
      instead. *)

  val float32 : float array -> t
  (** [float32 arr] is a non-nullable float32 column from [arr].

      The resulting column has no null mask. All values, including [nan], are
      treated as regular data. *)

  val float64 : float array -> t
  (** [float64 arr] is a non-nullable float64 column from [arr].

      The resulting column has no null mask. All values, including [nan], are
      treated as regular data. *)

  val int32 : int32 array -> t
  (** [int32 arr] is a non-nullable int32 column from [arr].

      The resulting column has no null mask. *)

  val int64 : int64 array -> t
  (** [int64 arr] is a non-nullable int64 column from [arr].

      The resulting column has no null mask. *)

  val bool : bool array -> t
  (** [bool arr] is a non-nullable boolean column from [arr].

      All values are wrapped as [Some value], creating a column with no nulls.
  *)

  val string : string array -> t
  (** [string arr] is a non-nullable string column from [arr].

      All values are wrapped as [Some value], creating a column with no nulls.
  *)

  (** {2:nullable From option arrays (nullable)}

      Create columns from option arrays with explicit null representation.
      Numeric types attach a null mask (while storing placeholder values in the
      tensor), whereas string and boolean types preserve the option structure.
  *)

  val float32_opt : float option array -> t
  (** [float32_opt arr] is a nullable float32 column from [arr].

      [None] values are recorded in the null mask. Placeholder [nan] values are
      stored in the tensor; callers must rely on the mask (via option accessors
      or {!module:Agg} helpers) to detect nulls. *)

  val float64_opt : float option array -> t
  (** [float64_opt arr] is a nullable float64 column from [arr].

      [None] values are recorded in the null mask. Placeholder [nan] values are
      stored in the tensor; callers must rely on the mask to detect nulls. *)

  val int32_opt : int32 option array -> t
  (** [int32_opt arr] is a nullable int32 column from [arr].

      [None] values are recorded in the null mask. The tensor stores
      [Int32.min_int] as placeholder, but the mask is authoritative when
      checking for nulls. *)

  val int64_opt : int64 option array -> t
  (** [int64_opt arr] is a nullable int64 column from [arr].

      [None] values are recorded in the null mask. The tensor stores
      [Int64.min_int] as placeholder, but the mask is authoritative when
      checking for nulls. *)

  val bool_opt : bool option array -> t
  (** [bool_opt arr] is a nullable boolean column from [arr].

      The option array is used directly without conversion. O(1). *)

  val string_opt : string option array -> t
  (** [string_opt arr] is a nullable string column from [arr].

      The option array is used directly without conversion. O(1). *)

  (** {2:properties Properties} *)

  val length : t -> int
  (** [length col] is the number of elements in [col]. *)

  val null_mask : t -> bool array option
  (** [null_mask col] is the null mask of [col], if any.

      Returns [Some mask] when an explicit mask was attached via a nullable
      constructor, [None] otherwise. *)

  val dtype :
    t -> [ `Float32 | `Float64 | `Int32 | `Int64 | `Bool | `String | `Other ]
  (** [dtype col] is the column's data type as a poly-variant tag. *)

  val is_null_at : t -> int -> bool
  (** [is_null_at col i] is [true] iff the value at index [i] is null.

      Checks the null mask for numeric columns, or tests for [None] in
      string/boolean columns. *)

  (** {2:of_tensor From tensors} *)

  val of_tensor : ('a, 'b) Nx.t -> t
  (** [of_tensor t] is a non-nullable column from the 1D tensor [t].

      The tensor's dtype is preserved. Existing payload values (including NaNs
      or extremal integers) remain regular data.

      Raises [Invalid_argument] if [t] is not 1D. O(1) — the tensor is used
      directly without copying. *)

  (** {2:nulls Null handling} *)

  val has_nulls : t -> bool
  (** [has_nulls col] is [true] iff [col] contains at least one null value.

      Checks the null mask for numeric columns, or scans for [None] in
      string/boolean columns. *)

  val null_count : t -> int
  (** [null_count col] is the number of null values in [col]. *)

  val drop_nulls : t -> t
  (** [drop_nulls col] is [col] with all null values removed.

      The column type is preserved. *)

  val fill_nulls : t -> value:t -> t
  (** [fill_nulls col ~value] is [col] with null values replaced by the first
      element of [value].

      [value] must be a single-element column of the same type as [col]. Raises
      [Invalid_argument] if column types don't match.

      See also {!Talon.fill_null} for a more convenient scalar-based API at the
      dataframe level. *)

  (** {2:col_transforms Column transforms} *)

  val cumsum : t -> t
  (** [cumsum col] is the cumulative sum of [col], preserving the dtype. *)

  val cumprod : t -> t
  (** [cumprod col] is the cumulative product of [col], preserving the dtype. *)

  val diff : ?periods:int -> t -> t
  (** [diff ?periods col] is the element-wise difference between consecutive
      values. [periods] defaults to [1]. *)

  val pct_change : ?periods:int -> t -> t
  (** [pct_change ?periods col] is the fractional change between consecutive
      values. [nan] where the previous value is zero. [periods] defaults to [1].
      Result is always float64. *)

  val shift : periods:int -> t -> t
  (** [shift ~periods col] is [col] with values shifted by [periods] positions.
      Positive shifts move values down (inserting nulls at the top), negative
      shifts move values up. *)

  (** {2:extraction Extraction} *)

  val to_tensor : ('a, 'b) Nx.dtype -> t -> ('a, 'b) Nx.t option
  (** [to_tensor dtype col] is the underlying tensor if [col] is numeric and its
      dtype matches [dtype]. *)

  val to_string_array : t -> string option array option
  (** [to_string_array col] is the underlying string option array if [col] is a
      string column. *)

  val to_bool_array : t -> bool option array option
  (** [to_bool_array col] is the underlying bool option array if [col] is a
      boolean column. *)

  val to_string_fn : ?null:string -> t -> int -> string
  (** [to_string_fn ?null col] is a function that formats the value at index [i]
      as a string.

      The underlying array is extracted once so repeated calls are O(1). [null]
      defaults to ["<null>"]. *)

  val pp : Format.formatter -> t -> unit
  (** [pp] formats a column for inspection. Shows the dtype, length, and up to 5
      values. *)
end

(** {1:creation DataFrame creation} *)

val empty : t
(** [empty] is an empty dataframe with no rows or columns.

    Neutral element for {!concat}. *)

val create : (string * Col.t) list -> t
(** [create pairs] is a dataframe from [(name, column)] pairs.

    Column names must be unique (case-sensitive) and all columns must have the
    same length.

    Raises [Invalid_argument] if duplicate column names exist or column lengths
    differ. *)

val of_tensors : ?names:string list -> ('a, 'b) Nx.t list -> t
(** [of_tensors ?names tensors] is a dataframe from 1D tensors.

    All tensors must have the same shape and dtype. [names] defaults to
    ["col0"], ["col1"], etc.

    Raises [Invalid_argument] if tensors have inconsistent shapes, any tensor is
    not 1D, names are not unique, or the wrong number of names is provided. *)

val of_nx : ?names:string list -> ('a, 'b) Nx.t -> t
(** [of_nx ?names tensor] is a dataframe from a 2D tensor.

    Each column of the tensor becomes a dataframe column. [names] defaults to
    ["col0"], ["col1"], etc.

    Raises [Invalid_argument] if [tensor] is not 2D or names are not unique. *)

(** {1:inspection Shape and inspection} *)

val shape : t -> int * int
(** [shape df] is [(rows, columns)]. *)

val num_rows : t -> int
(** [num_rows df] is the number of rows in [df]. *)

val num_columns : t -> int
(** [num_columns df] is the number of columns in [df]. *)

val column_names : t -> string list
(** [column_names df] is the column names of [df] in order. *)

val column_types :
  t ->
  (string
  * [ `Float32 | `Float64 | `Int32 | `Int64 | `Bool | `String | `Other ])
  list
(** [column_types df] is the column names paired with their detected types. *)

val select_columns :
  t -> [ `Numeric | `Float | `Int | `Bool | `String ] -> string list
(** [select_columns df category] is the column names matching [category].

    Categories:
    - [`Numeric]: all numeric types (float32, float64, int32, int64)
    - [`Float]: floating-point types only (float32, float64)
    - [`Int]: integer types only (int32, int64)
    - [`Bool]: boolean columns
    - [`String]: string columns *)

val is_empty : t -> bool
(** [is_empty df] is [true] iff [df] has no rows.

    {b Note.} A dataframe can have columns but zero rows and still be considered
    empty. *)

(** {1:col_access Column access and manipulation} *)

val get_column : t -> string -> Col.t option
(** [get_column df name] is the column named [name] in [df], if any. *)

val get_column_exn : t -> string -> Col.t
(** [get_column_exn df name] is the column named [name] in [df].

    Raises [Not_found] if the column does not exist. *)

val to_array : ('a, 'b) Nx.dtype -> t -> string -> 'a array option
(** [to_array dtype df name] is the numeric column [name] as a typed array if
    the column exists and matches [dtype].

    Null values retain their placeholder representation (NaN for floats,
    sentinel values for integers). See {!to_opt_array} to distinguish nulls from
    data. *)

val to_opt_array : ('a, 'b) Nx.dtype -> t -> string -> 'a option array option
(** [to_opt_array dtype df name] is the numeric column [name] as an option array
    if the column exists and matches [dtype]. [None] for null elements, [Some v]
    for present values. *)

val to_bool_array : t -> string -> bool option array option
(** [to_bool_array df name] is the bool option array for column [name] if it
    exists and is bool type. *)

val to_string_array : t -> string -> string option array option
(** [to_string_array df name] is the string option array for column [name] if it
    exists and is string type. *)

val has_column : t -> string -> bool
(** [has_column df name] is [true] iff [df] has a column named [name]. *)

val add_column : t -> string -> Col.t -> t
(** [add_column df name col] is [df] with column [name] added or replaced.

    Raises [Invalid_argument] if [Col.length col] differs from [num_rows df]. *)

val drop_column : t -> string -> t
(** [drop_column df name] is [df] without column [name].

    {b Note.} Returns [df] unchanged if the column does not exist. *)

val drop_columns : t -> string list -> t
(** [drop_columns df names] is [df] without the named columns.

    Non-existent columns are silently ignored. *)

val rename_column : t -> old_name:string -> new_name:string -> t
(** [rename_column df ~old_name ~new_name] is [df] with column [old_name]
    renamed to [new_name].

    Raises [Not_found] if [old_name] does not exist. Raises [Invalid_argument]
    if [new_name] already exists as a different column. *)

val select : ?strict:bool -> t -> string list -> t
(** [select ?strict df names] is the sub-dataframe with only the named columns,
    in the order given by [names].

    [strict] defaults to [true]: raises [Not_found] if any name is missing. When
    [false], missing columns are silently skipped. *)

val reorder_columns : t -> string list -> t
(** [reorder_columns df names] is [df] with columns reordered so that [names]
    appear first (in that order), followed by any remaining columns in their
    original relative order.

    Raises [Not_found] if any name in the list does not exist. *)

(** {1:row_ops Row-wise operations}

    The {!Row} module provides a declarative way to express computations over
    dataframe rows. *)

module Row : sig
  (** Row-wise computations using an applicative interface. *)

  (** {2:applicative Applicative interface} *)

  val return : 'a -> 'a row
  (** [return x] is a computation that produces [x] for every row. *)

  val apply : ('a -> 'b) row -> 'a row -> 'b row
  (** [apply f x] is the computation that applies [f] to [x] for each row. *)

  val map : 'a row -> f:('a -> 'b) -> 'b row
  (** [map x ~f] is the computation that applies [f] to each row's value from
      [x]. *)

  val map2 : 'a row -> 'b row -> f:('a -> 'b -> 'c) -> 'c row
  (** [map2 x y ~f] is the computation that applies [f] to corresponding values
      from [x] and [y]. *)

  val map3 : 'a row -> 'b row -> 'c row -> f:('a -> 'b -> 'c -> 'd) -> 'd row
  (** [map3 x y z ~f] combines three computations with [f]. *)

  val both : 'a row -> 'b row -> ('a * 'b) row
  (** [both x y] is the computation that pairs values from [x] and [y]. *)

  (** {2:accessors Column accessors} *)

  val float32 : string -> float row
  (** [float32 name] extracts float32 values from column [name].

      Raises [Not_found] if the column does not exist. Raises [Invalid_argument]
      if the column is not float32 type. *)

  val float64 : string -> float row
  (** [float64 name] extracts float64 values from column [name].

      Raises [Not_found] if the column does not exist. Raises [Invalid_argument]
      if the column is not float64 type. *)

  val int32 : string -> int32 row
  (** [int32 name] extracts int32 values from column [name].

      Raises [Not_found] if the column does not exist. Raises [Invalid_argument]
      if the column is not int32 type. *)

  val int64 : string -> int64 row
  (** [int64 name] extracts int64 values from column [name].

      Raises [Not_found] if the column does not exist. Raises [Invalid_argument]
      if the column is not int64 type. *)

  val string : string -> string row
  (** [string name] extracts string values from column [name].

      {b Note.} Null values are converted to empty strings.

      Raises [Not_found] if the column does not exist. Raises [Invalid_argument]
      if the column is not string type. *)

  val bool : string -> bool row
  (** [bool name] extracts boolean values from column [name].

      {b Note.} Null values are converted to [false].

      Raises [Not_found] if the column does not exist. Raises [Invalid_argument]
      if the column is not boolean type. *)

  val number : string -> float row
  (** [number name] extracts numeric values from column [name], coercing all
      numeric types to float.

      {b Note.} Null values become [nan].

      Raises [Not_found] if the column does not exist. Raises [Invalid_argument]
      if the column is not a numeric type. *)

  (** {2:row_info Row information} *)

  val index : int row
  (** [index] is the current row index (0-based). *)

  val sequence : 'a row list -> 'a list row
  (** [sequence xs] is the computation that collects values from all
      computations in [xs] into a list. *)

  val fold_list : 'a row list -> init:'b -> f:('b -> 'a -> 'b) -> 'b row
  (** [fold_list xs ~init ~f] folds [f] over the computations in [xs] without
      creating an intermediate list. *)

  (** {2:opt_accessors Option-based accessors}

      These accessors return [None] for null values instead of using placeholder
      values. Use these when you need to distinguish genuine values from missing
      data. *)

  val float32_opt : string -> float option row
  (** [float32_opt name] extracts float32 values as options from column [name].
      [None] for null values.

      Raises [Not_found] if the column does not exist. Raises [Invalid_argument]
      if the column is not float32 type. *)

  val float64_opt : string -> float option row
  (** [float64_opt name] extracts float64 values as options from column [name].
      [None] for null values.

      Raises [Not_found] if the column does not exist. Raises [Invalid_argument]
      if the column is not float64 type. *)

  val int32_opt : string -> int32 option row
  (** [int32_opt name] extracts int32 values as options from column [name].
      [None] for null values.

      Raises [Not_found] if the column does not exist. Raises [Invalid_argument]
      if the column is not int32 type. *)

  val int64_opt : string -> int64 option row
  (** [int64_opt name] extracts int64 values as options from column [name].
      [None] for null values.

      Raises [Not_found] if the column does not exist. Raises [Invalid_argument]
      if the column is not int64 type. *)

  val string_opt : string -> string option row
  (** [string_opt name] extracts string values as options from column [name].
      [None] for null values.

      Raises [Not_found] if the column does not exist. Raises [Invalid_argument]
      if the column is not string type. *)

  val bool_opt : string -> bool option row
  (** [bool_opt name] extracts boolean values as options from column [name].
      [None] for null values.

      Raises [Not_found] if the column does not exist. Raises [Invalid_argument]
      if the column is not boolean type. *)
end

(** {1:filtering Row filtering and transformation} *)

val head : ?n:int -> t -> t
(** [head ?n df] is the first [n] rows of [df]. [n] defaults to [5].

    If [n] exceeds the number of rows, returns the entire dataframe. *)

val tail : ?n:int -> t -> t
(** [tail ?n df] is the last [n] rows of [df]. [n] defaults to [5].

    If [n] exceeds the number of rows, returns the entire dataframe. *)

val slice : t -> start:int -> stop:int -> t
(** [slice df ~start ~stop] is the rows from [start] (inclusive) to [stop]
    (exclusive).

    Raises [Invalid_argument] if [start < 0], [stop < start], or indices are out
    of bounds. *)

val sample : ?n:int -> ?frac:float -> ?replace:bool -> ?seed:int -> t -> t
(** [sample ?n ?frac ?replace ?seed df] is a random sample of rows from [df].

    Exactly one of [n] or [frac] must be specified:
    - [n]: exact number of rows to sample.
    - [frac]: fraction of rows to sample (in \[[0];[1]\]).
    - [replace] defaults to [false].
    - [seed]: random seed for reproducible sampling.

    Raises [Invalid_argument] if both [n] and [frac] are specified, neither is
    specified, [frac] is outside \[[0];[1]\], or [n > num_rows df] when
    [replace] is [false]. *)

val filter : t -> bool array -> t
(** [filter df mask] is the rows of [df] where [mask] is [true].

    Raises [Invalid_argument] if [Array.length mask] differs from [num_rows df].
*)

val filter_by : t -> bool row -> t
(** [filter_by df pred] is the rows of [df] where [pred] is [true].

    Raises the same exceptions as the column accessors used in [pred]. *)

val drop_nulls : ?subset:string list -> t -> t
(** [drop_nulls ?subset df] is [df] with rows containing null values removed.

    When [subset] is provided, only those columns are checked for nulls.
    Otherwise all columns are checked. A row is dropped if any checked column is
    null at that position. *)

val fill_null :
  t ->
  string ->
  with_value:
    [ `Float of float
    | `Int32 of int32
    | `Int64 of int64
    | `String of string
    | `Bool of bool ] ->
  t
(** [fill_null df col_name ~with_value] is [df] with null values in column
    [col_name] replaced by [with_value].

    The value type must match the column type. Raises [Invalid_argument] if the
    column does not exist or the types do not match. *)

val drop_duplicates : ?subset:string list -> t -> t
(** [drop_duplicates ?subset df] is [df] with duplicate rows removed, keeping
    the first occurrence.

    When [subset] is provided, only those columns are considered for equality.
    Raises [Not_found] if any column in [subset] does not exist. *)

val concat : axis:[ `Rows | `Columns ] -> t list -> t
(** [concat ~axis dfs] is the concatenation of [dfs].

    - [`Rows]: all dataframes must have the same columns; rows are stacked.
    - [`Columns]: all dataframes must have the same number of rows; columns are
      combined. Column names must be unique across dataframes.

    Raises [Invalid_argument] if [dfs] is empty or the dataframes are
    incompatible for the chosen axis. *)

val map : t -> ('a, 'b) Nx.dtype -> 'a row -> ('a, 'b) Nx.t
(** [map df dtype f] is a 1D tensor of the given [dtype] obtained by applying
    [f] to each row of [df]. *)

val with_column : t -> string -> ('a, 'b) Nx.dtype -> 'a row -> t
(** [with_column df name dtype f] is [df] with a column [name] whose values are
    produced by applying [f] to each row. If a column named [name] already
    exists, it is replaced. *)

val with_string_column : t -> string -> string row -> t
(** [with_string_column df name f] is [df] with a string column [name] whose
    values are produced by [f]. *)

val with_bool_column : t -> string -> bool row -> t
(** [with_bool_column df name f] is [df] with a boolean column [name] whose
    values are produced by [f]. *)

val with_columns : t -> (string * Col.t) list -> t
(** [with_columns df cols] is [df] with the given columns added or replaced.

    Raises [Invalid_argument] if any column length differs from [num_rows df].
*)

val iter : t -> unit row -> unit
(** [iter df f] applies [f] to each row of [df] for side effects. *)

val fold : t -> init:'acc -> f:('acc -> 'acc) row -> 'acc
(** [fold df ~init ~f] folds [f] over the rows of [df] with accumulator [init].
*)

(** {1:sorting Sorting and grouping} *)

val sort : t -> 'a row -> compare:('a -> 'a -> int) -> t
(** [sort df key ~compare] is [df] with rows sorted by the values produced by
    [key], ordered according to [compare].

    O(n log n) in the number of rows. *)

val sort_values : ?ascending:bool -> t -> string -> t
(** [sort_values ?ascending df name] is [df] with rows sorted by column [name].

    [ascending] defaults to [true]. Null values are always sorted to the end
    regardless of direction.

    Raises [Not_found] if the column does not exist. O(n log n). *)

val group_by : t -> 'key row -> ('key * t) list
(** [group_by df key] is the list of [(k, sub_df)] pairs obtained by grouping
    the rows of [df] by the values produced by [key].

    The order of groups is not guaranteed. Rows within each group maintain their
    original relative order. *)

(** {1:transforms Column transforms} *)

val cumsum : t -> string -> t
(** [cumsum df name] is [df] with column [name] replaced by its cumulative sum,
    preserving the column's dtype.

    Raises [Not_found] if the column does not exist. *)

val cumprod : t -> string -> t
(** [cumprod df name] is [df] with column [name] replaced by its cumulative
    product, preserving the column's dtype.

    Raises [Not_found] if the column does not exist. *)

val diff : t -> string -> ?periods:int -> unit -> t
(** [diff df name ?periods ()] is [df] with column [name] replaced by the
    element-wise difference between consecutive values. [periods] defaults to
    [1].

    Raises [Not_found] if the column does not exist. *)

val pct_change : t -> string -> ?periods:int -> unit -> t
(** [pct_change df name ?periods ()] is [df] with column [name] replaced by the
    fractional change between consecutive values. [nan] where the previous value
    is zero. [periods] defaults to [1]. Result column is always float64.

    Raises [Not_found] if the column does not exist. *)

val shift : t -> string -> periods:int -> t
(** [shift df name ~periods] is [df] with column [name] shifted by [periods]
    positions. Positive shifts move values down (inserting nulls at the top),
    negative shifts move values up.

    Raises [Not_found] if the column does not exist. *)

(** {1:col_inspect Column inspection} *)

val is_null : t -> string -> Col.t
(** [is_null df name] is a boolean column where [true] indicates a null value at
    that position. *)

val value_counts : t -> string -> t
(** [value_counts df name] is a two-column dataframe with columns ["value"] and
    ["count"], containing the unique non-null values and their frequencies,
    sorted by count descending. *)

(** {1:aggregations Aggregations} *)

module Agg : sig
  (** Column-wise aggregation operations.

      All numeric aggregations coerce any numeric column to float, eliminating
      the need for type-specific sub-modules. *)

  (** {2:scalar Scalar aggregations}

      These reduce a column to a single scalar value. *)

  val sum : t -> string -> float
  (** [sum df name] is the sum of non-null values in column [name] as float. *)

  val mean : t -> string -> float
  (** [mean df name] is the arithmetic mean of non-null values in column [name].
  *)

  val std : t -> string -> float
  (** [std df name] is the population standard deviation of column [name]
      (divides by [n], not [n-1]). *)

  val var : t -> string -> float
  (** [var df name] is the population variance of column [name] (divides by [n],
      not [n-1]). *)

  val min : t -> string -> float option
  (** [min df name] is the minimum non-null value, or [None] if the column is
      empty or all null. *)

  val max : t -> string -> float option
  (** [max df name] is the maximum non-null value, or [None] if the column is
      empty or all null. *)

  val median : t -> string -> float
  (** [median df name] is the median (50th percentile) of column [name]. *)

  val quantile : t -> string -> q:float -> float
  (** [quantile df name ~q] is the [q]-th quantile of column [name] ([q] in
      \[[0];[1]\]). *)

  (** {2:generic Generic aggregations} *)

  val count : t -> string -> int
  (** [count df name] is the number of non-null values in column [name]. *)

  val nunique : t -> string -> int
  (** [nunique df name] is the number of unique non-null values in column
      [name]. *)

  (** {2:row_agg Row-wise (horizontal) aggregations}

      These compute aggregations across columns for each row. *)

  val row_sum : ?skipna:bool -> t -> names:string list -> Col.t
  (** [row_sum ?skipna df ~names] is the row-wise sum across the named columns.
      [skipna] defaults to [true]: skip null values. When [false], any null in a
      row makes the entire row result null. *)

  val row_mean : ?skipna:bool -> t -> names:string list -> Col.t
  (** [row_mean ?skipna df ~names] is the row-wise mean across the named
      columns. [skipna] defaults to [true]. *)

  val row_min : ?skipna:bool -> t -> names:string list -> Col.t
  (** [row_min ?skipna df ~names] is the row-wise minimum across the named
      columns. [skipna] defaults to [true]. *)

  val row_max : ?skipna:bool -> t -> names:string list -> Col.t
  (** [row_max ?skipna df ~names] is the row-wise maximum across the named
      columns. [skipna] defaults to [true]. *)

  val dot : t -> names:string list -> weights:float array -> Col.t
  (** [dot df ~names ~weights] is the weighted sum (dot product) across the
      named columns for each row.

      [weights] must have the same length as [names]. Raises [Invalid_argument]
      if lengths differ or columns are not numeric. *)

  val row_all : t -> names:string list -> Col.t
  (** [row_all df ~names] is the row-wise logical AND across the named boolean
      columns.

      Each row is [true] only if all values are [Some true]. [None] and
      [Some false] both count as false. Raises [Invalid_argument] if any column
      is not boolean or does not exist. *)

  val row_any : t -> names:string list -> Col.t
  (** [row_any df ~names] is the row-wise logical OR across the named boolean
      columns.

      Each row is [true] if any value is [Some true]. Raises [Invalid_argument]
      if any column is not boolean or does not exist. *)

  (** {2:string_agg String aggregations} *)

  module String : sig
    val min : t -> string -> string option
    (** [min df name] is the lexicographically smallest non-null string, or
        [None] if the column is empty or all null. *)

    val max : t -> string -> string option
    (** [max df name] is the lexicographically largest non-null string, or
        [None] if the column is empty or all null. *)

    val concat : t -> string -> ?sep:string -> unit -> string
    (** [concat df name ?sep ()] is the concatenation of all non-null strings.
        [sep] defaults to [""]. Empty string if all values are null. *)

    val unique : t -> string -> string array
    (** [unique df name] is the array of unique non-null values.

        Order is not guaranteed. *)

    val nunique : t -> string -> int
    (** [nunique df name] is the number of unique non-null values. *)

    val mode : t -> string -> string option
    (** [mode df name] is the most frequent non-null value, or [None] if the
        column is empty or all null. *)
  end

  (** {2:bool_agg Boolean aggregations} *)

  module Bool : sig
    val all : t -> string -> bool
    (** [all df name] is [true] iff all non-null values are [true].

        Returns [true] for columns with only null values (vacuous truth). *)

    val any : t -> string -> bool
    (** [any df name] is [true] iff any non-null value is [true].

        Returns [false] for columns with only null values. *)

    val sum : t -> string -> int
    (** [sum df name] is the number of [true] values. Nulls are excluded. *)

    val mean : t -> string -> float
    (** [mean df name] is the proportion of [true] values among non-null. [nan]
        if all values are null. *)
  end
end

(** {1:joins Joins and merges} *)

val join :
  t ->
  t ->
  on:string ->
  ?right_on:string ->
  how:[ `Inner | `Left | `Right | `Outer ] ->
  ?suffixes:string * string ->
  unit ->
  t
(** [join df1 df2 ~on ?right_on ~how ?suffixes ()] joins two dataframes on key
    columns.

    [on] names the key column in [df1]. [right_on] names the key column in
    [df2]; defaults to [on] when both dataframes share the same column name.

    Join types:
    - [`Inner]: rows where key exists in both dataframes.
    - [`Left]: all rows from [df1], null-filled for missing [df2] rows.
    - [`Right]: all rows from [df2], null-filled for missing [df1] rows.
    - [`Outer]: all rows from both, null-filled where missing.

    Null keys never match (null != null). Duplicate column names receive
    [suffixes] (default: ["_x"], ["_y"]).

    Raises [Not_found] if a key column is missing. Raises [Invalid_argument] if
    key columns have incompatible types. *)

(** {1:reshape Pivot and reshape} *)

val pivot :
  t ->
  index:string ->
  columns:string ->
  values:string ->
  ?agg_func:[ `Sum | `Mean | `Count | `Min | `Max ] ->
  unit ->
  t
(** [pivot df ~index ~columns ~values ?agg_func ()] is a pivot table from [df].

    - [index]: column whose values become row identifiers.
    - [columns]: column whose unique values become new column names.
    - [values]: column containing the data to fill the table.
    - [agg_func] defaults to [`Sum] for numeric, [`Count] for others.

    Raises [Not_found] if any specified column does not exist. Raises
    [Invalid_argument] if [values] is incompatible with [agg_func]. *)

val melt :
  t ->
  ?id_vars:string list ->
  ?value_vars:string list ->
  ?var_name:string ->
  ?value_name:string ->
  unit ->
  t
(** [melt df ?id_vars ?value_vars ?var_name ?value_name ()] unpivots [df] from
    wide to long format.

    - [id_vars]: columns to keep as identifiers (default: all non-[value_vars]).
    - [value_vars]: columns to melt (default: all non-[id_vars]).
    - [var_name] defaults to ["variable"].
    - [value_name] defaults to ["value"].

    Raises [Not_found] if any specified column does not exist. Raises
    [Invalid_argument] if [id_vars] and [value_vars] overlap. *)

(** {1:converting Converting} *)

val to_nx : t -> (float, Bigarray.float32_elt) Nx.t
(** [to_nx df] is a 2D float32 tensor from the numeric columns of [df].

    Rows correspond to dataframe rows, columns to numeric dataframe columns (in
    order). All numeric types are cast to float32. Null values become [nan].
    String and boolean columns are ignored.

    Raises [Invalid_argument] if [df] contains no numeric columns. *)

(** {1:fmt Formatting and inspecting} *)

val pp : ?max_rows:int -> ?max_cols:int -> Format.formatter -> t -> unit
(** [pp ?max_rows ?max_cols ppf df] formats [df] as a table on [ppf].

    [max_rows] defaults to [10]. [max_cols] defaults to [10]. *)

val to_string : ?max_rows:int -> ?max_cols:int -> t -> string
(** [to_string ?max_rows ?max_cols df] is [df] formatted as a table string. *)

val print : ?max_rows:int -> ?max_cols:int -> t -> unit
(** [print ?max_rows ?max_cols df] is
    [pp ?max_rows ?max_cols Format.std_formatter df]. *)

val describe : t -> t
(** [describe df] is a dataframe of summary statistics for the numeric columns
    of [df].

    Rows are: count, mean, std, min, 25%, 50%, 75%, max. String and boolean
    columns are ignored. *)

val cast_column : t -> string -> ('a, 'b) Nx.dtype -> t
(** [cast_column df name dtype] is [df] with column [name] converted to the
    numeric [dtype].

    Null values are preserved through the conversion.

    Raises [Not_found] if the column does not exist. Raises [Invalid_argument]
    if the source column is not numeric. *)

val pp_info : Format.formatter -> t -> unit
(** [pp_info ppf df] formats detailed information about [df] on [ppf]: shape,
    column names and types, null counts, and memory usage. *)

val info : t -> unit
(** [info df] is [pp_info Format.std_formatter df]. *)
