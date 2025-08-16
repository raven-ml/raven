(** Talon - A dataframe library for OCaml

    Talon provides efficient tabular data manipulation with heterogeneous column
    types, inspired by pandas and polars.

    {2 Overview}

    Dataframes are immutable collections of named columns with equal length.
    Each column can contain different types of data:
    - Numeric tensors (via Nx)
    - String arrays
    - Other custom types

    {2 Example}

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
(** Type of dataframe. *)

(** {1 Columns} *)

module Col : sig
  (** Column types and creation utilities. *)

  type t =
    | P : ('a, 'b) Nx.dtype * ('a, 'b) Nx.t -> t
    | S : string option array -> t
    | B : bool option array -> t
        (** Column representation for heterogeneous types with nullable support.
            - P: Numeric tensors (uses NaN for float nulls, min_int for int
              nulls)
            - S: String option arrays (None represents null)
            - B: Boolean option arrays (None represents null)

            Null semantics:
            - Numeric columns: NaN for floats, designated sentinel for ints
            - String/Bool columns: None represents null values
            - Operations propagate nulls (null + x = null)
            - Aggregations skip nulls by default

            Type promotion rules:
            - Mixed int/float operations promote to float
            - int32 + int64 promotes to int64
            - float32 + float64 promotes to float64 *)

  (** {3 From arrays (non-nullable)} *)

  val float32 : float array -> t
  (** [float32 arr] creates a float32 column from array. NaN values are treated
      as nulls. *)

  val float64 : float array -> t
  (** [float64 arr] creates a float64 column from array. NaN values are treated
      as nulls. *)

  val int32 : int32 array -> t
  (** [int32 arr] creates an int32 column from array. *)

  val int64 : int64 array -> t
  (** [int64 arr] creates an int64 column from array. *)

  val bool : bool array -> t
  (** [bool arr] creates a non-nullable boolean column from array. *)

  val string : string array -> t
  (** [string arr] creates a non-nullable string column from array. *)

  (** {3 From option arrays (nullable)} *)

  val float32_opt : float option array -> t
  (** [float32_opt arr] creates a nullable float32 column. None represents null.
  *)

  val float64_opt : float option array -> t
  (** [float64_opt arr] creates a nullable float64 column. None represents null.
  *)

  val int32_opt : int32 option array -> t
  (** [int32_opt arr] creates a nullable int32 column. None represents null. *)

  val int64_opt : int64 option array -> t
  (** [int64_opt arr] creates a nullable int64 column. None represents null. *)

  val bool_opt : bool option array -> t
  (** [bool_opt arr] creates a nullable boolean column. None represents null. *)

  val string_opt : string option array -> t
  (** [string_opt arr] creates a nullable string column. None represents null.
  *)

  (** {3 From lists} *)

  val float32_list : float list -> t
  (** [float32_list lst] creates a float32 column from list. *)

  val float64_list : float list -> t
  (** [float64_list lst] creates a float64 column from list. *)

  val int32_list : int32 list -> t
  (** [int32_list lst] creates an int32 column from list. *)

  val int64_list : int64 list -> t
  (** [int64_list lst] creates an int64 column from list. *)

  val bool_list : bool list -> t
  (** [bool_list lst] creates a boolean column from list. *)

  val string_list : string list -> t
  (** [string_list lst] creates a string column from list. *)

  (** {3 From tensors} *)

  val of_tensor : ('a, 'b) Nx.t -> t
  (** [of_tensor t] creates a column from a 1D tensor.
      @raise Invalid_argument if tensor is not 1D. *)

  (** {3 Null handling} *)

  val has_nulls : t -> bool
  (** [has_nulls col] returns true if column contains any null values. *)

  val null_count : t -> int
  (** [null_count col] returns the number of null values in column. *)

  val drop_nulls : t -> t
  (** [drop_nulls col] returns a new column with null values removed. *)

  val fill_nulls : t -> value:'a -> t
  (** [fill_nulls col ~value] replaces null values with the given value. *)
end

(** {1 Creation} *)

val empty : t
(** [empty] creates an empty dataframe with no rows or columns. *)

val create : (string * Col.t) list -> t
(** [create pairs] creates a dataframe from (column_name, column) pairs.

    Invariants:
    - Column names must be unique
    - All columns must have the same length
    - Numeric columns must be 1D tensors

    @raise Invalid_argument
      if:
      - Duplicate column names exist
      - Column lengths differ
      - Invalid tensor shapes *)

val of_tensors : ?names:string list -> ('a, 'b) Nx.t list -> t
(** [of_tensors ?names tensors] creates from 1D Nx tensors (all same dtype).
    Default names "col0", "col1", etc.

    @raise Invalid_argument
      if:
      - Tensors have inconsistent shapes
      - Names are not unique
      - Wrong number of names provided *)

val from_nx : ?names:string list -> ('a, 'b) Nx.t -> t
(** [from_nx ?names tensor] creates from 2D tensor, treating each column as a
    separate series. Default names "col0", "col1", etc.

    @raise Invalid_argument if tensor is not 2D or names are not unique. *)

(** {1 Shape and Info} *)

val shape : t -> int * int
(** [shape df] returns (num_rows, num_columns). *)

val num_rows : t -> int
(** [num_rows df] returns number of rows. *)

val num_columns : t -> int
(** [num_columns df] returns number of columns. *)

val column_names : t -> string list
(** [column_names df] returns column names in order. *)

val column_types :
  t ->
  (string
  * [ `Float32 | `Float64 | `Int32 | `Int64 | `Bool | `String | `Other ])
  list
(** [column_types df] returns column names with their types. *)

val is_empty : t -> bool
(** [is_empty df] returns true if dataframe has no rows. *)

val numeric_column_names : t -> string list
(** [numeric_column_names df] returns all columns of type float32/float64/int32/int64. *)

(** {1 Column Operations} *)

val get_column : t -> string -> Col.t option
(** [get_column df name] returns column or None. *)

val get_column_exn : t -> string -> Col.t
(** [get_column_exn df name] returns packed column.

    @raise Not_found if column doesn't exist. *)

val to_float32_array : t -> string -> float array option
(** [to_float32_array df name] extracts column as float array if it's float32.
*)

val to_float64_array : t -> string -> float array option
(** [to_float64_array df name] extracts column as float array if it's float64.
*)

val to_int32_array : t -> string -> int32 array option
(** [to_int32_array df name] extracts column as int32 array if it's int32. *)

val to_int64_array : t -> string -> int64 array option
(** [to_int64_array df name] extracts column as int64 array if it's int64. *)

val to_bool_array : t -> string -> bool array option
(** [to_bool_array df name] extracts column as bool array if it's bool. *)

val to_string_array : t -> string -> string array option
(** [to_string_array df name] extracts column as string array if it's string. *)

val has_column : t -> string -> bool
(** [has_column df name] returns true if column exists. *)

val add_column : t -> string -> Col.t -> t
(** [add_column df name col] adds or replaces column.

    @raise Invalid_argument if column length doesn't match dataframe rows. *)

val drop_column : t -> string -> t
(** [drop_column df name] removes column. Returns unchanged if column doesn't
    exist. *)

val drop_columns : t -> string list -> t
(** [drop_columns df names] removes multiple columns. *)

val rename_column : t -> old_name:string -> new_name:string -> t
(** [rename_column df ~old_name ~new_name] renames a column.

    @raise Not_found if old_name doesn't exist.
    @raise Invalid_argument if new_name already exists. *)

val select : t -> string list -> t
(** [select df names] returns dataframe with only specified columns, in the
    given order.

    @raise Not_found if any column name doesn't exist. *)

val select_loose : t -> string list -> t
(** [select_loose df names] returns dataframe with specified columns that exist.
    Silently ignores missing column names. Order is preserved. *)

val reorder_columns : t -> string list -> t
(** [reorder_columns df names] reorders columns. Missing names are appended at
    the end. *)

(** {1 Row-wise Operations} *)

(** Row applicative for functional row operations. *)
module Row : sig
  type 'a t
  (** Applicative for row-wise computations. *)

  (** {2 Applicative Interface} *)

  val return : 'a -> 'a t
  (** [return x] creates a constant computation returning x for each row. *)

  val apply : ('a -> 'b) t -> 'a t -> 'b t
  (** [apply f x] applies a function computation to a value computation. *)

  val map : 'a t -> f:('a -> 'b) -> 'b t
  (** [map x ~f] maps a function over a computation. *)

  val map2 : 'a t -> 'b t -> f:('a -> 'b -> 'c) -> 'c t
  (** [map2 x y ~f] combines two computations with a binary function. *)

  val map3 : 'a t -> 'b t -> 'c t -> f:('a -> 'b -> 'c -> 'd) -> 'd t
  (** [map3 x y z ~f] combines three computations with a ternary function. *)

  val both : 'a t -> 'b t -> ('a * 'b) t
  (** [both x y] pairs two computations. *)

  (** {2 Column Accessors} *)

  val float32 : string -> float t
  (** [float32 name] extracts float32 values from column. *)

  val float64 : string -> float t
  (** [float64 name] extracts float64 values from column. *)

  val int32 : string -> int32 t
  (** [int32 name] extracts int32 values from column. *)

  val int64 : string -> int64 t
  (** [int64 name] extracts int64 values from column. *)

  val string : string -> string t
  (** [string name] extracts string values from column. *)

  val bool : string -> bool t
  (** [bool name] extracts boolean values from column. *)

  (** {2 Row Information} *)

  val index : int t
  (** [index] returns the current row index. *)

  val sequence : 'a t list -> 'a list t
  (** [sequence xs] transforms a list of computations into a computation of a list.
      Standard applicative operation for collecting multiple column values. *)

  val all : 'a t list -> 'a list t
  (** [all xs] is an alias for [sequence xs]. Collects all values from the list of computations. *)

  val map_list : 'a t list -> f:('a list -> 'b) -> 'b t
  (** [map_list xs ~f] sequences computations then maps f over the resulting list.
      Equivalent to [map (sequence xs) ~f]. *)

  (** Convenience builders to avoid writing [List.map int32 names] etc. *)
  val float32s : string list -> float t list
  val float64s : string list -> float t list
  val int32s   : string list -> int32 t list
  val int64s   : string list -> int64 t list
  val bools    : string list -> bool t list
  val strings  : string list -> string t list
end

(** {1 Row Operations} *)

val head : ?n:int -> t -> t
(** [head ?n df] returns first n rows (default 5). *)

val tail : ?n:int -> t -> t
(** [tail ?n df] returns last n rows (default 5). *)

val slice : t -> start:int -> stop:int -> t
(** [slice df ~start ~stop] returns rows from start (inclusive) to stop
    (exclusive). *)

val sample : ?n:int -> ?frac:float -> ?replace:bool -> ?seed:int -> t -> t
(** [sample ?n ?frac ?replace ?seed df] returns random sample of rows. Either n
    (number of rows) or frac (fraction) must be specified. *)

val filter : t -> bool array -> t
(** [filter df mask] filters rows where mask is true. Mask must match num_rows.
*)

val filter_by : t -> bool Row.t -> t
(** [filter_by df pred] filters rows where predicate returns true. *)

val drop_duplicates : ?subset:string list -> t -> t
(** [drop_duplicates ?subset df] removes duplicate rows. If subset is provided,
    only considers those columns for duplicates. *)

val concat : axis:[ `Rows | `Columns ] -> t list -> t
(** [concat ~axis dfs] concatenates dataframes along rows or columns. *)

val map : t -> ('a, 'b) Nx.dtype -> 'a Row.t -> ('a, 'b) Nx.t
(** [map df dtype f] maps row-wise computation to new numeric column. *)

val map_column : t -> string -> ('a, 'b) Nx.dtype -> 'a Row.t -> t
(** [map_column df name dtype f] creates new column from row-wise computation.
*)

val iter : t -> unit Row.t -> unit
(** [iter df f] iterates over rows. *)

val fold : t -> init:'acc -> f:('acc -> 'acc) Row.t -> 'acc
(** [fold df ~init ~f] folds over rows. *)

val fold_left :
  t -> init:'acc -> f:('acc -> 'a) Row.t -> ('acc -> 'a -> 'acc) -> 'acc
(** [fold_left df ~init ~f combine] folds with explicit combine function. *)

(** {1 Sorting and Grouping} *)

val sort : t -> 'a Row.t -> compare:('a -> 'a -> int) -> t
(** [sort df key ~compare] sorts rows by key. *)

val sort_by_column : ?ascending:bool -> t -> string -> t
(** [sort_by_column ?ascending df name] sorts by column values (default
    ascending=true). *)

val group_by : t -> 'key Row.t -> ('key * t) list
(** [group_by df key] groups rows by key, returning list of (key,
    sub-dataframe). *)

val group_by_column : t -> string -> (Col.t * t) list
(** [group_by_column df name] groups by column values. *)

(** {1 Aggregations and Column Operations} *)

module Agg : sig
  (** Aggregation and column transformation operations. *)

  (** {2 Type-specific aggregations} *)

  (** Float aggregations - work on any numeric column (int or float types).
      Values are coerced to float. Raises exception if column is not numeric. *)
  module Float : sig
    val sum : t -> string -> float
    (** [sum df name] returns the sum as float. Works on any numeric column. *)

    val mean : t -> string -> float
    (** [mean df name] returns the mean. *)

    val std : t -> string -> float
    (** [std df name] returns the standard deviation. *)

    val var : t -> string -> float
    (** [var df name] returns the variance. *)

    val min : t -> string -> float option
    (** [min df name] returns minimum value, None if empty. *)

    val max : t -> string -> float option
    (** [max df name] returns maximum value, None if empty. *)

    val median : t -> string -> float
    (** [median df name] returns the median. *)

    val quantile : t -> string -> q:float -> float
    (** [quantile df name ~q] returns the q-th quantile (0 <= q <= 1). *)
  end

  (** Int aggregations - work on any integer column type. Raises exception if
      column is not integer type. *)
  module Int : sig
    val sum : t -> string -> int64
    (** [sum df name] returns sum as int64. Works on any int type (int32, int64,
        etc). *)

    val min : t -> string -> int64 option
    (** [min df name] returns minimum value as int64, None if empty. *)

    val max : t -> string -> int64 option
    (** [max df name] returns maximum value as int64, None if empty. *)

    val mean : t -> string -> float
    (** [mean df name] returns mean as float (since mean of ints is often
        fractional). *)
  end

  (** String aggregations - work on string columns only. Raises exception if
      column is not string type. *)
  module String : sig
    val min : t -> string -> string option
    (** [min df name] returns lexicographically smallest string, None if empty.
    *)

    val max : t -> string -> string option
    (** [max df name] returns lexicographically largest string, None if empty.
    *)

    val concat : t -> string -> ?sep:string -> unit -> string
    (** [concat df name ?sep ()] concatenates all strings with separator
        (default ""). *)

    val unique : t -> string -> string array
    (** [unique df name] returns array of unique values. *)

    val nunique : t -> string -> int
    (** [nunique df name] returns count of unique values. *)

    val mode : t -> string -> string option
    (** [mode df name] returns most frequent value, None if empty. *)
  end

  (** Bool aggregations - work on boolean columns only. Raises exception if
      column is not bool type. *)
  module Bool : sig
    val all : t -> string -> bool
    (** [all df name] returns true if all values are true. *)

    val any : t -> string -> bool
    (** [any df name] returns true if any value is true. *)

    val sum : t -> string -> int
    (** [sum df name] returns count of true values. *)

    val mean : t -> string -> float
    (** [mean df name] returns proportion of true values. *)
  end

  (** {2 Generic aggregations - work on any column type} *)

  val count : t -> string -> int
  (** [count df name] returns number of non-null values. Nulls are: NaN for
      floats, None for options, sentinel values for ints. *)

  val nunique : t -> string -> int
  (** [nunique df name] returns count of unique values for any column type. *)

  val value_counts : t -> string -> Col.t * int array
  (** [value_counts df name] returns unique values and their counts. *)

  val is_null : t -> string -> bool array
  (** [is_null df name] returns boolean array where true indicates null values.
      For numeric columns: NaN for floats, sentinel values for ints. For
      string/bool columns: None values. *)

  (** {2 Column operations - return new columns preserving dtype} *)

  val cumsum : t -> string -> Col.t
  (** [cumsum df name] returns cumulative sum preserving column dtype. Raises
      exception if column is not numeric. *)

  val cumprod : t -> string -> Col.t
  (** [cumprod df name] returns cumulative product preserving column dtype.
      Raises exception if column is not numeric. *)

  val diff : t -> string -> ?periods:int -> unit -> Col.t
  (** [diff df name ?periods ()] returns difference between elements (default
      periods=1). Raises exception if column is not numeric. *)

  val pct_change : t -> string -> ?periods:int -> unit -> Col.t
  (** [pct_change df name ?periods ()] returns percentage change between
      elements. Raises exception if column is not numeric. *)

  val shift : t -> string -> periods:int -> Col.t
  (** [shift df name ~periods] shifts values by periods (positive = forward,
      negative = backward). *)

  val fillna : t -> string -> value:Col.t -> Col.t
  (** [fillna df name ~value] fills null/missing values with provided value.
      Value column must have a single element or match the column length.
      @raise Invalid_argument if value type doesn't match column type. *)
end

(** {1 Joins and Merges} *)

val join :
  t ->
  t ->
  on:string ->
  how:[ `Inner | `Left | `Right | `Outer ] ->
  ?suffixes:string * string ->
  unit ->
  t
(** [join df1 df2 ~on ~how ?suffixes ()] joins two dataframes on column.
    Suffixes default to ("_x", "_y") for duplicate column names.

    Null semantics:
    - Inner join: nulls don't match (null != null)
    - Left/Right join: unmatched rows have null values in joined columns
    - Outer join: combines both behaviors *)

val merge :
  t ->
  t ->
  left_on:string ->
  right_on:string ->
  how:[ `Inner | `Left | `Right | `Outer ] ->
  ?suffixes:string * string ->
  unit ->
  t
(** [merge df1 df2 ~left_on ~right_on ~how ?suffixes ()] merges on different
    column names. *)

(** {1 Pivot and Reshape} *)

val pivot :
  t ->
  index:string ->
  columns:string ->
  values:string ->
  ?agg_func:[ `Sum | `Mean | `Count | `Min | `Max ] ->
  unit ->
  t
(** [pivot df ~index ~columns ~values ?agg_func ()] creates pivot table. *)

val melt :
  t ->
  ?id_vars:string list ->
  ?value_vars:string list ->
  ?var_name:string ->
  ?value_name:string ->
  unit ->
  t
(** [melt df ?id_vars ?value_vars ?var_name ?value_name ()] unpivots dataframe.
*)

(** {1 I/O} *)

val to_nx : t -> (float, Bigarray.float32_elt) Nx.t
(** [to_nx df] converts all numeric columns to 2D tensor.

    @raise Invalid_argument
      if columns have different types or non-numeric columns exist. *)

(* CSV and JSON I/O are now available in separate sublibraries: - Talon_csv
   module provides CSV reading/writing functionality - Talon_json module
   provides JSON serialization/deserialization *)

(** {1 Display} *)

val print : ?max_rows:int -> ?max_cols:int -> t -> unit
(** [print ?max_rows ?max_cols df] pretty-prints dataframe. Default max_rows=10,
    max_cols=10. *)

val describe : t -> t
(** [describe df] returns summary statistics for numeric columns. *)

val cast_column : t -> string -> ('a, 'b) Nx.dtype -> t
(** [cast_column df name dtype] converts column to specified dtype. Useful
    before operations that need specific types.
    @raise Invalid_argument if conversion is not possible. *)

val info : t -> unit
(** [info df] prints dataframe information (shape, column types, memory usage).
*)
