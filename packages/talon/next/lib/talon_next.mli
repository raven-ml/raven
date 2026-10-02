(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tables: named, typed columns.

    {!Type}s say what columns store and {!Kind}s what their cells read as in
    OCaml. {!Binary}, {!Decimal}, {!Time} and {!Record} are the OCaml values
    that cells read as, {!Schema}s name and type a table's columns, and {!Tz}
    reads the time zone database that zoned operations take.

    A {!Query} is the centre: a plan over a table or a {!Source}, transformed by
    verbs, whose schema is known before any data is read. Its verbs take
    {!Expr}essions, read through {!Col} handles and {!Ext} declarations; {!Sel}
    chooses columns, {!Order} sorts, {!Window} cuts the rows that
    {!Expr.rolling} reduces, and {!Join} conditions pair rows. Plan problems
    raise [Invalid_argument]; failures in data are {!Error} values. *)

type t
(** The type for tables: named, typed columns of equal length. *)

type table := t

module Binary = Binary
module Decimal = Decimal
module Time = Time

module Kind : sig
  (** Kinds: the OCaml types that cells read as.

      A column's {!Type.t} is what it stores, and its kind is the OCaml type its
      cells read as. Several types share a kind: [int8] through [uint64] all
      read as [int]. Handles name a kind, never a type, so storage width never
      appears in user code.

      {b Binding.} A handle of kind [k] binds a column of type [t] iff
      [provably_equal k (Type.kind t)] is [Some Equal]. An extension column's
      kind is the extension kind, which no kind is provably equal to, so a
      handle never binds one: only an extension's declaration reads its cells.
  *)

  type 'a t
  (** The type for kinds whose cells read as OCaml values of type ['a]. *)

  (** {1:kinds Kinds}

      Record columns read through {!Record.kind}. *)

  val bool : bool t
  (** [bool] reads [bool] columns. *)

  val int : int t
  (** [int] reads the integer columns, [int8] through [int64] and [uint8]
      through [uint64]. A value outside OCaml's [int] range fails when it is
      read. *)

  val float : float t
  (** [float] reads [float16], [float32] and [float64] columns. *)

  val string : string t
  (** [string] reads [string] and [categorical] columns. *)

  val binary : Binary.t t
  (** [binary] reads [binary] columns. *)

  val decimal : Decimal.t t
  (** [decimal] reads [decimal] columns of every precision and scale. *)

  val date : Time.date t
  (** [date] reads [date] columns. *)

  val instant : Time.instant t
  (** [instant] reads [datetime] columns of every unit and zone. A value outside
      the range of {!Time.instant} fails when it is read. *)

  val span : Time.span t
  (** [span] reads [duration] and [clock] columns of every unit. A time of day
      reads as its span from midnight. A value outside the range of {!Time.span}
      fails when it is read. *)

  val list : 'a t -> 'a array t
  (** [list k] reads list columns whose elements [k] reads. *)

  val tensor : ('a, 'b) Nx.dtype -> ('a, 'b) Nx.t t
  (** [tensor dt] reads tensor columns of dtype [dt] and any cell shape. *)

  (** {1:comparing Comparing and formatting} *)

  val provably_equal : 'a t -> 'b t -> ('a, 'b) Stdlib.Type.eq option
  (** [provably_equal k0 k1] is [Some Equal] iff [k0] and [k1] are the same
      kind, neither being nor containing the extension kind: lists are equal
      when their elements are, and tensors when their dtypes are. It is [None]
      otherwise, in particular for two extension kinds. *)

  val pp : Format.formatter -> 'a t -> unit
  (** [pp ppf k] formats [k] by the name of the value that builds it: [bool],
      [int], [float], [string], [binary], [decimal], [date], [instant] and
      [span]. The record kind formats as [record] and the extension kind as
      [ext]. A list kind formats its element kind in brackets, as in
      [list[float]], and a tensor kind its dtype, as in [tensor[float32]]. *)
end

module Record : sig
  (** Record cells.

      A record cell is the value of one row of a record column ({!Type.record}):
      named fields in order, each holding a value or null. Record columns read
      as {!t} through {!kind}. *)

  type t
  (** The type for record cells. A record cell's field names are distinct. *)

  val kind : t Kind.t
  (** [kind] reads record columns of every field list. *)

  val empty : t
  (** [empty] is the record cell with no fields. *)

  val add : 'a Kind.t -> string -> 'a option -> t -> t
  (** [add k name v r] is [r] with a last field [name] of kind [k] holding [v],
      or null if [v] is [None].

      Raises [Invalid_argument] if [r] has a field [name], if [name] is not
      valid UTF-8, or if [k] is or contains the extension kind. *)

  val field : 'a Kind.t -> string -> t -> 'a option
  (** [field k name r] is the field [name] of [r] read as [k]: [Some v] if it
      holds [v], and [None] if it is null.

      Raises [Invalid_argument] if [r] has no field [name], or if [k] does not
      read it: a field whose type is or contains an extension, or a field of
      another kind. *)

  val names : t -> string list
  (** [names r] is the names of [r]'s fields, in order. *)
end

module Type : sig
  (** Column types.

      A column's type is what it stores: [float32], [datetime[ms, UTC]],
      [list[string]]. A type is indexed by its {e kind}, the OCaml type its
      cells read as ({!Kind}): {!float32} is a [float t]. Whatever fixes what a
      column stores therefore also fixes what OCaml reads from it. Collections
      of types of several kinds, such as a {!Schema.t}, hold them as {!any}.

      The set of types is closed. A new type is an extension: a named type
      stored as one of these, identified by its name, its metadata and its
      storage type together, as in Arrow. Computing on an extension column takes
      its declaration, a value of the [Ext] module.

      The constructors of {!t} are exposed for matching and private: types are
      built with the {{!constructors}constructors} below, which check their
      arguments.

      Under [open Talon_next], this module shadows [Stdlib.Type]. *)

  (** {1:types Types} *)

  (** The type for the units of temporal ticks. *)
  type unit_ =
    | S  (** Seconds. *)
    | Ms  (** Milliseconds. *)
    | Us  (** Microseconds. *)
    | Ns  (** Nanoseconds. *)

  type ext
  (** The type that extension cells read as. It has no values: an extension
      column is read only through the extension's declaration, and its kind
      binds no handle. *)

  (** The type for column types whose cells read as ['a]. *)
  type 'a t = private
    | Bool : bool t  (** Booleans. *)
    | Int8 : int t  (** Signed 8-bit integers. *)
    | Int16 : int t  (** Signed 16-bit integers. *)
    | Int32 : int t  (** Signed 32-bit integers. *)
    | Int64 : int t
        (** Signed 64-bit integers. A value outside OCaml's [int] range fails
            when it is read as [int]. *)
    | Uint8 : int t  (** Unsigned 8-bit integers. *)
    | Uint16 : int t  (** Unsigned 16-bit integers. *)
    | Uint32 : int t
        (** Unsigned 32-bit integers. A value outside OCaml's [int] range fails
            when it is read as [int]. *)
    | Uint64 : int t
        (** Unsigned 64-bit integers. A value outside OCaml's [int] range fails
            when it is read as [int]. *)
    | Float16 : float t  (** IEEE 754 binary16 floats. *)
    | Float32 : float t  (** IEEE 754 binary32 floats. *)
    | Float64 : float t  (** IEEE 754 binary64 floats. *)
    | Decimal : { precision : int; scale : int } -> Decimal.t t
        (** Decimals of at most [precision] digits, [scale] of them after the
            point, stored as int64 unscaled values. Here [1 <= precision <= 18]
            and [0 <= scale <= precision]. *)
    | String : string t
        (** UTF-8 text. Text is validated when it enters talon, and invalid
            bytes belong in {!Binary}. *)
    | Binary : Binary.t t  (** Byte strings. *)
    | Categorical : string iarray -> string t
        (** Strings from a dictionary, stored as int32 positions in it. The
            dictionary holds distinct UTF-8 strings, and its order is the order
            of the values. *)
    | Date : Time.date t  (** Dates, stored as int32 days since 1970-01-01. *)
    | Clock : unit_ -> Time.span t
        (** Times of day, stored as int64 ticks of the unit since midnight. A
            time of day reads as its span from midnight. *)
    | Duration : unit_ -> Time.span t
        (** Signed durations, stored as int64 ticks of the unit. *)
    | Datetime : { unit_ : unit_; zone : string option } -> Time.instant t
        (** Instants, stored as int64 ticks of [unit_] since 1970-01-01
            00:00:00. With [zone = Some z], the ticks count UTC and [z] names
            the zone the data belongs to, as in Arrow's timestamp with a time
            zone; temporal operations still take their zone explicitly. With
            [zone = None], the ticks count a wall clock in no particular zone,
            as in Arrow's timestamp without one. A tick outside the range of
            {!Time.instant} fails when it is read. *)
    | List : 'a t -> 'a array t
        (** Lists of values of the element type, read as arrays. *)
    | Record : (string * any) list -> Record.t t
        (** Records with the given fields in order, Arrow's struct. Field names
            are distinct UTF-8 strings. *)
    | Tensor : ('a, 'b) Nx.dtype * int iarray -> ('a, 'b) Nx.t t
        (** Tensors of one dtype and one shape, one per cell, stored as Arrow's
            canonical [arrow.fixed_shape_tensor]. The shape has at least one
            dimension, and no dimension is negative. *)
    | Ext : { name : string; metadata : string; storage : 's t } -> ext t
        (** Extension types: values stored as [storage] and identified by
            [name], [metadata] and [storage] together. The name is non-empty
            UTF-8, and [storage] is not an extension type. *)

  and any = Any : 'a t -> any  (** The type for types of any kind. *)

  (** {1:constructors Constructors} *)

  val bool : bool t
  (** [bool] is {!Bool}. *)

  val int8 : int t
  (** [int8] is {!Int8}. *)

  val int16 : int t
  (** [int16] is {!Int16}. *)

  val int32 : int t
  (** [int32] is {!Int32}. *)

  val int64 : int t
  (** [int64] is {!Int64}. *)

  val uint8 : int t
  (** [uint8] is {!Uint8}. *)

  val uint16 : int t
  (** [uint16] is {!Uint16}. *)

  val uint32 : int t
  (** [uint32] is {!Uint32}. *)

  val uint64 : int t
  (** [uint64] is {!Uint64}. *)

  val float16 : float t
  (** [float16] is {!Float16}. *)

  val float32 : float t
  (** [float32] is {!Float32}. *)

  val float64 : float t
  (** [float64] is {!Float64}. *)

  val decimal : precision:int -> scale:int -> Decimal.t t
  (** [decimal ~precision ~scale] is {!Decimal} with [precision] and [scale].

      Raises [Invalid_argument] if [precision] is not in \[[1];[18]\] or [scale]
      is not in \[[0];[precision]\]. *)

  val string : string t
  (** [string] is {!String}. *)

  val binary : Binary.t t
  (** [binary] is {!Binary}. *)

  val categorical : string array -> string t
  (** [categorical d] is {!Categorical} with a copy of the dictionary [d], which
      may be empty.

      Raises [Invalid_argument] if [d] holds a string twice or a string that is
      not valid UTF-8, or holds more than 2{^ 31} - 1 strings. *)

  val date : Time.date t
  (** [date] is {!Date}. *)

  val clock : unit_ -> Time.span t
  (** [clock u] is {!Clock} in the unit [u]. *)

  val duration : unit_ -> Time.span t
  (** [duration u] is {!Duration} in the unit [u]. *)

  val datetime : ?zone:string -> unit_ -> Time.instant t
  (** [datetime ?zone u] is {!Datetime} in the unit [u], with the zone [zone],
      or without a zone when [zone] is absent. The zone is not looked up: a
      zoned operation resolves it in the {!Tz} database it is given.

      Raises [Invalid_argument] if [zone] is empty or not valid UTF-8. *)

  val list : 'a t -> 'a array t
  (** [list t] is {!List} of [t]. *)

  val record : (string * any) list -> Record.t t
  (** [record fields] is {!Record} with [fields], in order. A record type may
      have no fields, and a field's name may be empty.

      Raises [Invalid_argument] if two fields have the same name, or a name is
      not valid UTF-8. *)

  val tensor : ('a, 'b) Nx.dtype -> int array -> ('a, 'b) Nx.t t
  (** [tensor dt shape] is {!Tensor} of [dt] with a copy of [shape].

      Raises [Invalid_argument] if [shape] is empty or holds a negative
      dimension. *)

  val ext : name:string -> ?metadata:string -> 'a t -> ext t
  (** [ext ~name ?metadata storage] is {!Ext} named [name], with [metadata]
      (defaults to [""]) and stored as [storage]. Format readers build the types
      of extension columns with it.

      Raises [Invalid_argument] if [name] is empty or not valid UTF-8, or if
      [storage] is an extension type. *)

  (** {1:kinds Kinds} *)

  val kind : 'a t -> 'a Kind.t
  (** [kind t] is the kind that [t]'s cells read as: {!Kind.bool}, {!Kind.int}
      for the integer types, {!Kind.float} for the float types, {!Kind.decimal},
      {!Kind.string} for [String] and [Categorical], {!Kind.binary},
      {!Kind.date}, {!Kind.span} for [Clock] and [Duration], {!Kind.instant} for
      [Datetime], [Kind.list (kind e)] for [List e], {!Record.kind} for every
      [Record], and [Kind.tensor dt] for [Tensor (dt, _)]. For [Ext] it is the
      extension kind, which reads nothing (see {!Kind.provably_equal}). *)

  (** {1:values Values} *)

  val holds : 'a t -> 'a -> bool
  (** [holds t v] is [true] iff [t] holds [v], that is iff [v] can be stored as
      [t]:
      - an integer is in the type's range;
      - a float is NaN, infinite, or rounds to a finite value at the type's
        precision;
      - a decimal is exact with the type's scale, in at most its precision
        digits;
      - a string is valid UTF-8, and for [Categorical] it is in the dictionary;
      - an instant or a span is a whole number of the type's unit, and for
        [Clock] it is at least zero and less than one day;
      - each element of a list is held by the element type;
      - a record has the type's field names in order, and each of its non-null
        fields has the kind of the field's type and is held by it, an extension
        field by its storage value;
      - a tensor has the type's shape.

      Booleans, byte strings and dates are always held. [Ext] has no values.
      This is the test a literal passes when it takes the type of the operand it
      meets, and the test values pass when they become a column.

      [holds t] does its work on [t] once, such as indexing a categorical's
      dictionary: apply it to [t] once and use the result for many values. *)

  val compare_value : 'a t -> 'a -> 'a -> int
  (** [compare_value t v0 v1] orders [v0] and [v1] by talon's total order on the
      values of [t], the order that sorting, comparisons and grouping use:
      - floats order [neg_infinity] < … < [infinity] < [nan]. [-0.] equals [0.],
        and every NaN equals every other;
      - integers, decimals, dates, spans and instants order by value;
      - [false] comes before [true];
      - strings and byte strings order by their bytes, which for UTF-8 text is
        code point order;
      - [Categorical] values order by their position in the dictionary;
      - lists order lexicographically, a list coming before every list it is a
        proper prefix of;
      - records order lexicographically by their fields, in the type's order;
      - tensors order lexicographically by their elements in row-major order:
        floats as above, complex numbers by their real then their imaginary
        part, each as a float, and other elements by value, unsigned integers as
        unsigned.

      A null record field comes after every value and equals another null field;
      talon's order puts nulls last at the top level and inside lists too, where
      this function does not see them. An extension field orders by its storage
      value. That order serves key identity; sort keys and ordering reductions
      over a type that is or contains an extension not declared ordered are
      refused when the verb is applied.

      {b Key identity.} Two non-null values are the same key iff
      [compare_value t v0 v1 = 0]. Null is one more key, the same as itself
      only.

      [compare_value t] does its work on [t] once, such as indexing a
      categorical's dictionary: apply it to [t] once and use the result for many
      comparisons.

      Raises [Invalid_argument] if a [Categorical] value is not in the
      dictionary, a record does not have the type's field names in order or
      holds a non-null field of another kind, or a tensor does not have the
      type's shape. *)

  (** {1:operands Operands} *)

  val common : 'a t list -> 'a t option
  (** [common ts] is the type at which operands of the types [ts] meet in an
      operation, and [None] if they do not meet or [ts] is empty. The result
      does not depend on the order of [ts]. Scalar types meet at the one of them
      that contains all the others: a type contains another when it can store
      every value the other can store, with the same meaning. Lists meet at the
      list of their elements' common type, and records with the same field names
      in the same order at the record of their fields' common types. One type
      contains another when:
      - they are equal;
      - [int8] in [int16] in [int32] in [int64], [uint8] in [uint16] in [uint32]
        in [uint64], and [uint8] in [int16], [uint16] in [int32], [uint32] in
        [int64];
      - [float16] in [float32] in [float64];
      - [decimal[p0, s0]] in [decimal[p1, s1]] when [s0 <= s1] and
        [p0 - s0 <= p1 - s1];
      - a [Categorical] in [String], and in a categorical whose dictionary
        begins with its own;
      - a [Clock] in a clock of a finer unit.

      So [int8] and [uint8] do not meet, while [int8], [uint8] and [int16] meet
      at [int16]. [uint64] does not meet [int64], although every [uint64] that
      OCaml reads is an [int64]. [Clock] and [Duration] never meet, and neither
      do datetimes or durations that differ in unit, since a coarser unit stores
      ticks a finer one cannot, or in zone: cast first. *)

  (** {1:predicates Equality and formatting} *)

  val equal : 'a t -> 'b t -> bool
  (** [equal t0 t1] is [true] iff [t0] and [t1] are the same type: the same
      constructor with equal arguments. Dictionaries are equal when they hold
      the same strings in the same order, zones when they are the same string,
      and record types when they have the same field names with equal types in
      the same order. *)

  val pp : Format.formatter -> 'a t -> unit
  (** [pp ppf t] formats [t] as schemas and plans show it:
      - [bool], [int8] to [uint64], [float16] to [float64], [string], [binary]
        and [date];
      - [decimal[10, 2]] for a precision and a scale;
      - [categorical["AA", "B6"]], with at most the first eight strings of the
        dictionary, followed by an ellipsis and the dictionary's size when it
        holds more: [categorical["a", "b", "c", "d", "e", "f", "g", "h", … 26]];
      - [clock[ns]], [duration[ms]], [datetime[us]] and [datetime[us, UTC]],
        with the units [s], [ms], [us] and [ns];
      - [list[float64]], [record[carrier string, delay float64]] and
        [tensor[float32, 3×4]];
      - [ext[ymir.epoch, float64]], and [ext[units.mass "kg", float64]] when the
        metadata is not empty.

      A field name, an extension name or a zone formats as is when it is
      non-empty and holds no ASCII space, comma, bracket, double quote,
      backslash or ASCII control byte. Otherwise it is quoted, as categories and
      metadata always are: between double quotes, with double quotes and
      backslashes preceded by a backslash, and control bytes and bytes that are
      not part of valid UTF-8 written as [\x] and two hexadecimal digits. *)

  val pp_name : Format.formatter -> string -> unit
  (** [pp_name ppf n] formats the name [n] of a field or a column as {!pp}
      formats a field name: as is, or quoted when it must be. Formats print
      their columns' names with it. *)

  val pp_quoted : Format.formatter -> string -> unit
  (** [pp_quoted ppf s] formats [s] quoted, as {!pp} quotes a category. Messages
      quote names and texts with it. *)
end

module Schema : sig
  (** Schemas: the names and types of a table's columns.

      A schema is an ordered list of columns, each a name and a {!Type.t}, with
      distinct names. Tables, queries and format descriptions each have one, and
      a query's is known before any data is read. *)

  type t
  (** The type for schemas. *)

  val v : (string * Type.any) list -> t
  (** [v columns] is the schema of [columns], in order. A schema may have no
      columns, and a column's name may be empty.

      Raises [Invalid_argument] if two columns have the same name, or a name is
      not valid UTF-8. *)

  val columns : t -> (string * Type.any) list
  (** [columns s] is the columns of [s], in order. [columns (v cs)] is [cs]. *)

  val find : t -> string -> Type.any option
  (** [find s name] is the type of the column [name] of [s], or [None] if [s]
      has no such column. It costs O(log n) for a schema of n columns. *)

  (** {1:comparing Comparing} *)

  val equal : t -> t -> bool
  (** [equal s0 s1] is [true] iff [s0] and [s1] have the same names with equal
      types ({!Type.equal}) in the same order. *)

  (** The type for the differences between two schemas [s0] and [s1]. *)
  type change =
    | Added of string * Type.any
        (** [Added (name, t)]: [s1] has the column [name] of type [t], and [s0]
            has none. *)
    | Removed of string * Type.any
        (** [Removed (name, t)]: [s0] has the column [name] of type [t], and
            [s1] has none. *)
    | Retyped of string * Type.any * Type.any
        (** [Retyped (name, t0, t1)]: the column [name] has the type [t0] in
            [s0] and a different type [t1] in [s1]. *)

  val diff : t -> t -> change list
  (** [diff s0 s1] is the changes from [s0] to [s1]: first the {!Removed} and
      {!Retyped} columns in the order of [s0], then the {!Added} ones in the
      order of [s1]. It ignores order, so it is empty iff [s0] and [s1] have the
      same names with equal types, in any order; then [equal s0 s1] holds iff
      the order is also the same. *)

  (** {1:fmt Formatting} *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf s] formats the columns of [s] in order, separated by commas, each
      as its name and its type: [carrier string, delay float64]. Names are
      quoted as {!Type.pp} quotes field names, and types format with {!Type.pp}.
      The empty schema formats as nothing. *)
end

module Column : sig
  (** Columns: one typed array of values, some of them null.

      A column is an Arrow array over nx buffers. Its {e validity} is a bitmap
      ({!Nx_bits.t}) with the bit of each row that holds a value set; it is
      absent when no row is null. Its values are laid out by its type:
      - one element per row of a primitive nx array: [bool] (one byte per
        value), the integer and float types, [int64] unscaled values for
        decimals, [int32] positions in the dictionary for categoricals, [int32]
        days for dates and [int64] ticks for clocks, durations and datetimes. A
        tensor column is one [(rows, …shape)] array;
      - offsets into a child for byte strings, text and lists: text is a list of
        bytes;
      - one child per field for records.

      An extension column is laid out as its storage. The values under a null
      are unspecified; talon writes zeros, and empty rows, under the nulls it
      makes. Columns are immutable, and share their buffers with the tensors and
      layouts that read them. *)

  type t
  (** The type for columns. *)

  val type_ : t -> Type.any
  (** [type_ c] is the type of [c]'s values. *)

  val length : t -> int
  (** [length c] is the number of rows of [c]. *)

  val null_count : t -> int
  (** [null_count c] is the number of null rows of [c]. It costs O(1). *)

  (** {1:ocaml OCaml values} *)

  val v : 'a Type.t -> 'a array -> t
  (** [v ty vs] is the column of type [ty] holding [vs], without nulls. A
      [float32] or [float16] value is stored rounded to the nearest value of the
      type, ties to even.

      Raises [Invalid_argument] naming the row if [ty] does not hold a value of
      [vs] ({!Type.holds}): text that is not UTF-8, a string outside a
      categorical's dictionary, an integer outside the type's range, a span that
      is not a whole number of the unit, a record of other fields. An extension
      type has no values, so a column of it holds only nulls. *)

  val of_options : 'a Type.t -> 'a option array -> t
  (** [of_options ty vs] is like {!v}, with a null for each [None]. *)

  val values : 'a Kind.t -> t -> 'a array
  (** [values k c] is [c]'s values read as [k].

      Raises [Invalid_argument] if [k] does not read [c]'s type (see
      {!Kind.provably_equal}: no kind reads an extension column), if a row of
      [c] is null, or, naming the row, if a value is outside what [k] reads: an
      integer outside OCaml's [int], an instant or a span outside {!Time}'s
      range, a list with a null element. *)

  val options : 'a Kind.t -> t -> 'a option array
  (** [options k c] is like {!values}, with [None] for each null. *)

  (** {1:tensors Tensors and bytes} *)

  val of_tensor : ?validity:Nx_bits.t -> ('a, 'b) Nx.t -> t
  (** [of_tensor ?validity x] is the column of [x]'s rows, without a copy:
      - for a 1-D [x], of the type of [x]'s dtype: [bool], [int8] to [uint64],
        [float16] to [float64];
      - for [x] of shape [(n, …shape)], a tensor column of [x]'s dtype and cell
        shape [shape].

      [validity] marks the rows that hold a value; it defaults to every row.

      Raises [Invalid_argument] if [x] is a scalar, if [x] is 1-D of a dtype
      that no scalar type stores ([bfloat16], the float8 and int4 dtypes,
      complex), or if [validity]'s length is not [x]'s rows. *)

  val to_tensor : ('a, 'b) Nx.dtype -> t -> ('a, 'b) Nx.t
  (** [to_tensor dt c] is [c]'s values as stored, in O(1): numbers and booleans,
      the days or ticks of temporal values, the codes of a categorical (its
      dictionary is in its type), the unscaled values of decimals, and
      [(rows, …shape)] for a tensor column. It shares [c]'s buffer, which must
      not be written.

      Raises [Invalid_argument] if [dt] is not [c]'s storage dtype ({!Nx.cast}
      converts the result), if [c] is not stored one element per row, or if [c]
      has a null. *)

  val validity : t -> Nx_bits.t option
  (** [validity c] is [c]'s validity, [None] iff [c] has no null. *)

  val ragged : t -> (int, Nx.uint8_elt) Nx_ragged.t
  (** [ragged c] is the bytes of [c], one row per row of [c], in O(1). [c] is
      stored as bytes: its type is [string], [binary] or an extension of either.

      Raises [Invalid_argument] if [c] is not stored as bytes, or has a null. *)

  (** {1:layout Layouts}

      A layout is a column's Arrow buffers, as formats read and write them. *)

  (** The type for layouts. *)
  type layout =
    | Fixed of { validity : Nx_bits.t option; values : Nx.packed }
        (** One element per row, or one cell for a tensor column. *)
    | Varsize of {
        validity : Nx_bits.t option;
        offsets : Nx.int64_t;
        child : t;
      }
        (** Row [r] is the child's rows [offsets.{r}] to [offsets.{r + 1} - 1]:
            the elements of a list, or, for [string] and [binary], the bytes, a
            [uint8] child without nulls. *)
    | Children of {
        validity : Nx_bits.t option;
        length : int;
        fields : (string * t) list;
      }
        (** [length] rows, with one child per field of a record, in the record
            type's order. *)

  val layout : t -> layout
  (** [layout c] is [c]'s layout, in O(1). *)

  val of_layout : Type.any -> layout -> (t, int * string) result
  (** [of_layout ty l] is the column of type [ty] laid out as [l], without a
      copy, or [Error (row, reason)] for the first row whose value [ty] does not
      hold: text that is not UTF-8, a code outside a categorical's dictionary, a
      decimal of more digits than its precision, a time of day outside the day.
      A row is checked only where it is not null. [reason] is a phrase, as in
      [invalid UTF-8 at byte 3]. A child holds its own values, so only [l]'s own
      values are checked.

      Raises [Invalid_argument] if [l] does not lay out [ty]: values of another
      dtype or cell shape than [ty]'s storage, a validity of another length,
      offsets that are not 1-D, start below [0], decrease or reach past the
      child, a child of another type (for text, a [uint8] column with a null),
      or fields of other names, types or lengths than [ty]'s. *)

  val parse : Type.any -> t -> (t, int * string) result
  (** [parse ty c] is the column of type [ty] whose rows are the values that the
      rows of the [string] or [binary] column [c] write, null where [c] is null,
      or [Error (row, reason)] at the first non-null row that is not [ty]'s text
      or holds a value [ty] does not, [reason] a phrase such as [not a number].
      Formats that read text call it, mapping [row] to their own location. The
      text of each type is:
      - [bool]: [true] or [false];
      - [int8] to [uint64]: a decimal integer with an optional sign, in the
        type's range;
      - [float16] to [float64]: a decimal number with an optional sign, digits
        on at least one side of an optional point and an optional exponent, or
        [inf], [infinity] or [nan] in any case with an optional sign, rounded to
        the nearest value of the type, ties to even, beyond its range to an
        infinity;
      - [decimal[p, s]]: a decimal number with an optional sign, exact at the
        scale [s] and of at most [p] digits;
      - [string]: the bytes, valid UTF-8; [binary]: the bytes;
      - a categorical: one of the dictionary's strings;
      - [date]: [YYYY-MM-DD], a year outside [0000] to [9999] signed and of at
        least four digits, as {!Time.Date.pp} writes it: [-0044-03-15];
      - [datetime[u]] and [datetime[u, z]]: a date, [T] or a space, [hh:mm:ss]
        and an optional fraction of one to nine digits, then, with a zone only,
        [Z] or [±hh:mm]; a whole number of [u] in [u]'s range.

      Raises [Invalid_argument] if [c] is neither [string] nor [binary], or [ty]
      is another type. *)

  val print : t -> t
  (** [print c] is the column of the canonical texts of [c]'s rows, null where
      [c] is null, which {!parse} reads back: [parse (type_ c) (print c)] is
      [c]. Formats that write text call it. It is [c] itself for [string] and
      [binary], and a [string] column otherwise. Each value is written in the
      text {!parse} reads:
      - a float in the fewest significant digits that read back to it at its
        type's width, without an exponent from [1e-7] up to [1e21] ([150],
        [0.0015], [-0], [1e+21]), or [nan], [inf] or [-inf];
      - a decimal with its type's scale of digits after the point;
      - a datetime with the fewest fraction digits that are exact, and [Z] when
        its type has a zone;
      - an integer in full, past OCaml's [int] included.

      A datetime whose day is outside {!Time.Date}'s range is written, and
      {!parse} refuses it as out of range.

      Raises [Invalid_argument] if [c]'s type is not one {!parse} reads. *)
end

(** {1:tables Tables} *)

val v : ?rows:int -> (string * Column.t) list -> t
(** [v ?rows cs] is the table of the columns [cs], in order, as one batch of
    [rows] rows. [rows] defaults to the length of the columns, and gives the
    rows of a table without columns, such as the batch a source yields for a
    request that reads none.

    Raises [Invalid_argument] if [cs] is empty and [rows] is not given, if
    [rows] is negative or is not the columns' length, if two columns have the
    same name, if a name is not valid UTF-8, or if the columns have different
    lengths. *)

val of_batches : t list -> t
(** [of_batches ts] is the rows of [ts] one after the other, without a copy, in
    O(number of batches). Each table's batches become the result's.

    Raises [Invalid_argument] if [ts] is empty or the tables' schemas differ
    ({!Schema.equal}). *)

val batches : t -> t list
(** [batches t] is [t]'s batches, each a table of one batch, in order. A table
    without rows has none. *)

val schema : t -> Schema.t
(** [schema t] is the names and types of [t]'s columns. *)

val rows : t -> int
(** [rows t] is the number of rows of [t]. *)

val column : t -> string -> Column.t
(** [column t name] is the column [name] of [t]: its own when [t] is one batch
    of a column whose buffers hold exactly its rows, else one copy that holds
    exactly them.

    Raises [Invalid_argument] if [t] has no column [name]. *)

val take : Nx.int64_t -> t -> t
(** [take indices t] is the rows of [t] at [indices], in order, as one batch:
    [take (Nx.Rng.permutation key (rows t)) t] shuffles every column.

    Raises [Invalid_argument] if [indices] is not 1-D or holds an index outside
    \[[0];[rows t - 1]\]. *)

val to_tensor : ('a, 'b) Nx.dtype -> string list -> t -> ('a, 'b) Nx.t
(** [to_tensor dt names t] is the [(rows t, List.length names)] matrix whose
    column [j] is the column [List.nth names j] of [t], each value converted to
    [dt] as {!Nx.cast} converts it. It is the one copy a columnar layout forces.
    The columns are numeric or boolean, and have no null.

    Raises [Invalid_argument] if a name is not a column of [t], if a column is
    neither numeric nor boolean, or has a null. *)

val equal : t -> t -> bool
(** [equal t0 t1] is [true] iff [t0] and [t1] have equal schemas and the same
    keys row by row, by key identity ({!Type.compare_value}, null being one more
    key), whatever their batches. *)

(** {1:display Display} *)

type limits = {
  head : int;  (** The rows shown from the start. *)
  tail : int;  (** The rows shown from the end. *)
  columns : int;  (** The columns shown, from the first. *)
  width : int;  (** The widest cell, in Unicode scalar values. *)
}
(** The type for display limits. A table of at most [head + tail] rows shows all
    of them. *)

val limits : limits
(** [limits] is [{ head = 5; tail = 5; columns = 12; width = 32 }], the limits
    of {!pp}. *)

val pp : Format.formatter -> t -> unit
(** [pp] is [pp_with limits]. *)

val pp_with : limits -> Format.formatter -> t -> unit
(** [pp_with l ppf t] formats [t] for people, reading only the rows it shows.
    With [{ limits with head = 2; tail = 1 }]:
    {v
    table 16 rows × 4 columns
     carrier  mean_delay  flights  name
     string   float64     int64    string
     OO          58.0000        9  ∅
     F9          53.4214      280  Frontier Airlines Inc.
     ⋮
     HA          29.0000        1  Hawaiian Airlines Inc.
     13 rows not shown
    v}
    - a header, then the names and types of the shown columns;
    - all rows, or the first [l.head], a [⋮] line, the last [l.tail], and the
      number of rows not shown;
    - with more than [l.columns] columns, a last line naming those not shown.

    A null is [∅]. Text shows as it reads, its control characters and bytes that
    are not UTF-8 escaped as {!Type.pp_quoted} escapes them, cut with […] past
    [l.width] scalar values. Numbers are right-aligned; the floats of a column
    show with one number of decimals, the fewest, up to six, that give each
    shown value six significant digits, or in scientific notation when a shown
    value needs it. Other values show in the text that {!Column.parse} reads,
    such as [2024-03-15T09:30:00.5Z] for a zoned datetime and [12.50] for a
    decimal of scale 2; durations and clocks as {!Time.Span.pp} formats them,
    lists as OCaml lists, and records and tensors as [<record>] and [<tensor>].

    Raises [Invalid_argument] if a limit is negative or [l.width] is [0]. *)

module Error : sig
  (** Failures found in data and in the environment.

      Reading a file that is missing or malformed, or data that breaks a query's
      contract, is not a programming error, so talon returns it as [Error e],
      where [e] says what failed and where it was found: in which file, at which
      line and column of its text, in which row group, at which bytes, and on
      which raw text. Formats and sources build errors with {!v}; programs print
      them with {!pp}. *)

  type t = Error.t
  (** The type for errors. An error is a message and the places it was found at,
      each of which may be unknown. *)

  val v :
    ?file:string ->
    ?line:int ->
    ?column:int ->
    ?row_group:int ->
    ?bytes:int * int ->
    ?text:string ->
    string ->
    t
  (** [v ?file ?line ?column ?row_group ?bytes ?text msg] is the error [msg],
      found:
      - in [file], a path as the program named it;
      - at [line] of [file]'s text and at [column] of that line, both counted
        from [1], the column in bytes;
      - in the row group [row_group] of a Parquet file, counted from [0];
      - at the bytes [bytes] of [file], [(first, last)], the zero-based
        positions of the first and the last byte of the range, both included;
      - on [text], the raw bytes that failed to read, which need not be valid
        UTF-8.

      [msg] says what failed and, when there is one, how to fix it, in one or
      more sentences: [cannot read as float64. Declare the null token (~nulls).]

      Raises [Invalid_argument] if [line < 1], [column < 1], [column] is given
      without [line], [row_group < 0], or [bytes] is given and [first < 0] or
      [last < first]. *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf e] formats [e] for people: the places [e] was found at, coarsest
      first, then its message, separated by [": "]:
      - the file, followed by [:line] and [:line:column] as compilers write
        them, [flights.csv:48213:12]; without a file, [line 48213] or
        [line 48213, column 12];
      - the row group, [row group 3];
      - the bytes, [bytes 106-113], or [byte 106] for a range of one byte;
      - the text, between double quotes, with double quotes and backslashes
        preceded by a backslash. Each byte of a control character (U+0000 to
        U+001F, U+007F to U+009F), of a bidirectional formatting control (U+202A
        to U+202E, U+2066 to U+2069), which could reorder the text around it,
        and each byte that is not part of valid UTF-8, is written as [\x] and
        two hexadecimal digits. A text longer than 64 bytes is cut before the
        first byte past its 64th, or before a valid UTF-8 sequence that this
        byte would split, and an ellipsis follows the closing quote: ["aaaa"…].

      As in [flights.csv:48213:12: "NA": cannot read as float64.] and
      [zoneinfo/Europe/Paris: bytes 106-113: transition 1 is not after the
       previous one].

      A failure that a run finds in the data starts with the plan step that
      found it, as [Query.pp] prints it, and the row of the step's input it was
      found at, counted from [0]; a text that fails to read follows them:
      [filter (cast int32 x > 0): row 48212: cannot cast 3.5 to int32.] and
      [derive ["n" := Str.parse int32 s]: row 3: "x1": not an integer.] *)

  val get_ok : ('a, t) result -> 'a
  (** [get_ok r] is [v] if [r] is [Ok v].

      Raises [Failure] with [e] formatted by {!pp} if [r] is [Error e]. *)
end

module Tz = Tz

module Sel : sig
  (** Column selectors.

      A selector chooses a set of columns by name, type or kind, without naming
      a schema. A verb resolves it against its input schema when it is applied,
      to an ordered list of distinct names. [Expr.keep], [Expr.across],
      [Expr.each], the pivot's [~cols] and [Kit.drop] take selectors; keys stay
      [string list].

      Selectors combine with {!( + )}, {!( - )} and {!inter}, written inside
      [Sel.( … )]: [Sel.(prefix "wk" - names [ "wk76" ])]. *)

  type t
  (** The type for selectors. *)

  (** {1:constructors Constructors} *)

  val all : t
  (** [all] selects every column, in schema order. *)

  val names : string list -> t
  (** [names ns] selects the columns [ns], in the order of [ns]. A name that
      appears twice is selected once, at its first position. A name the schema
      lacks is a problem. *)

  val prefix : string -> t
  (** [prefix p] selects the columns whose name starts with [p], in schema
      order. *)

  val suffix : string -> t
  (** [suffix s] selects the columns whose name ends with [s], in schema order.
  *)

  val of_kind : 'a Kind.t -> t
  (** [of_kind k] selects the columns that a handle of kind [k] binds (see
      {!Kind.provably_equal}), in schema order. It selects no extension column.
  *)

  val where : (string -> Type.any -> bool) -> t
  (** [where p] selects the columns [(n, t)] for which [p n t] is [true], in
      schema order. [p] must be pure; it runs when a verb is applied. *)

  val ( + ) : t -> t -> t
  (** [s0 + s1] selects [s0]'s columns, then [s1]'s columns not in [s0]. *)

  val ( - ) : t -> t -> t
  (** [s0 - s1] selects [s0]'s columns that are not in [s1], in [s0]'s order. *)

  val inter : t -> t -> t
  (** [inter s0 s1] selects [s0]'s columns that are in [s1], in [s0]'s order. *)
end

module Order : sig
  (** Sort keys.

      A sort key names a column, a direction and where its nulls go.
      [Query.sort] and [Expr.over]'s [~order] take a list of keys, compared in
      turn; a source's [~sorted] claim is one.

      Each key orders by talon's total order (see {!Type.compare_value}):
      ascending puts NaN after every number, and descending reverses it, putting
      NaN first. Nulls go last in both directions unless {!nulls_first}. *)

  type t
  (** The type for sort keys. *)

  val asc : string -> t
  (** [asc name] orders the column [name] ascending, nulls last. *)

  val desc : string -> t
  (** [desc name] orders the column [name] descending, nulls last. *)

  val nulls_first : t -> t
  (** [nulls_first k] is [k] with its nulls before every value. *)
end

module Window : sig
  (** Windows: the rows around a row.

      A window cuts, for each row i of a frame, the rows that [Expr.rolling]
      reduces: a range of positions around i, or a range of times around the
      time of i. Positions and times are those of the enclosing frame, in its
      order. *)

  type t
  (** The type for windows. *)

  val rows : before:int -> after:int -> t
  (** [rows ~before ~after] holds, for row i, the rows j of its frame with
      [i - before <= j <= i + after]: [rows ~before:6 ~after:0] is the last
      seven rows, the row included. A negative bound excludes rows on its side:
      [rows ~before:3 ~after:(-1)] is the three rows before, without the row
      itself. Rows past the frame's edges are absent, so a window near an edge
      holds fewer rows.

      Raises [Invalid_argument] if [before + after < 0], a window that is empty
      for every row. *)

  val time : ?after:Time.span -> before:Time.span -> string -> t
  (** [time ?after ~before on] holds the rows whose time in the column [on] is
      later than [before] before the row's own time and at most [after] past it:
      [time ~before:(Time.Span.days 7) "ts"] is the last seven days, the row's
      own time included. [after] defaults to the zero span. The column [on] is a
      datetime, date, clock or duration column, a date counting 86,400 seconds a
      day, and its values must ascend within each frame: a violation is a data
      error that suggests sorting (see [Expr.over]'s [~order]).

      Raises [Invalid_argument] if [before + after] is not positive, a window
      that is empty for every row. *)
end

module Expr : sig
  (** Expressions: typed computations over the columns of a frame.

      An expression [('a, 's) t] computes values that read as ['a] from the
      columns of a {e frame}, the ordered rows a verb or an enclosing expression
      gives it. Its {e shape} ['s] is {!row}, one value per row of the frame, or
      {!agg}, one value per frame. Handles ([Col]) are [row], reductions take
      [row] to [agg], and literals and elementwise operations keep their
      operands' shape, so [sum (w *. x) /. sum w] is [agg]. ['s] is a covariant
      phantom, so [let cutoff = Expr.float 15.] generalizes, and no call site
      writes a shape.

      Expressions are data. Building one reads nothing; a verb binds it to its
      input schema when the verb is applied, checks it and infers its type,
      reporting every problem at once. Write them inside [Expr.( … )], where the
      operators below shadow OCaml's.

      {b Types.} Whatever a value meets fixes its type:
      - A handle of kind [k] binds a column whose type [k] reads (see
        {!Kind.provably_equal}).
      - Operands meet at the one of their types that contains the others
        ({!Type.common}). Types that do not meet are a problem: [cast] first.
      - A literal, or an expression of literals such as [int 2 * int 50], takes
        the type of the operand it meets, which must hold each literal and the
        value of each integer operation of literals ({!Type.holds}). Where it
        meets none, it takes its kind's default type: [int64], [float64],
        [bool], [string], [date], [datetime[ns, UTC]] or [duration[ns]].
      - {!null}, a {!const} and a {!( $ )} result take the type of the operand
        they meet. Where they meet none, {!store} gives them one. Without it, a
        [const] or [$] result stands only as an argument of {!( $ )} or as what
        [Query.values] decodes, so [if_ c (const a) (const b)] is a problem, and
        a [null] stands nowhere.
      - Result types follow from operand types alone: each function below states
        its result's type.

      {b Frames.} A context supplies a frame, and frames nest:
      - [select], [derive] and [filter]: the input's rows;
      - [aggregate ~by]: one group's rows, in input order;
      - {!over}[ ~by ~order e]: the enclosing frame, partitioned by [by], each
        partition in [order]; results return to their rows;
      - {!rolling}[ w e]: each row's window, cut from the enclosing frame.

      A frame's order is its input order, and no expression reorders values
      without returning them to their rows.

      {b Nulls.} Elementwise operations are null where an operand is null,
      except where stated. Comparisons are Kleene: a comparison with null is
      null. Reductions skip nulls except {!rows}, {!count}, {!n_unique} and
      {!collect}. *)

  (** {1:types Expressions and outputs} *)

  type row
  (** The shape of expressions with one value per row of their frame. *)

  type agg
  (** The shape of expressions with one value per frame. *)

  type ('a, +'s) t
  (** The type for expressions of shape ['s] whose values read as ['a]. *)

  type +'s out
  (** The type for outputs of shape ['s]: named expressions, as [select],
      [derive], [aggregate] and {!record} take them. *)

  (** {1:outputs Outputs} *)

  val ( := ) : string -> ('a, 's) t -> 's out
  (** [name := e] outputs [e] as the column [name]. It binds more loosely than
      every operator, so ["late" := delay > float 15.] needs no parentheses. [e]
      needs a column type: a {!const} or {!( $ )} result needs {!store}, and an
      {!option} result is never one.

      Raises [Invalid_argument] if [name] is not valid UTF-8. *)

  val keep : Sel.t -> row out
  (** [keep sel] outputs the columns that [sel] selects, unchanged and under
      their names. It keeps a column of any type, extensions included. *)

  val across : 'a Kind.t -> Sel.t -> (string -> ('a, row) t -> 's out) -> 's out
  (** [across k sel f] is the outputs [f n (Col.v k n)] for each name [n] that
      [sel] selects, in order:
      [across Kind.float Sel.(prefix "wk") (fun n x -> n := over (rank x))]. A
      selected column that [k] does not bind is a problem: narrow [sel] with
      {!Sel.of_kind}. [f] runs when the verb is applied and must be pure. *)

  type 's column = { column : 'a. string -> ('a, row) t -> 's out }
  (** The type for functions of one column of any type. *)

  val each : Sel.t -> 's column -> 's out
  (** [each sel { column }] is the outputs [column n x] for each name [n] that
      [sel] selects, in order, where [x] reads the column [n] at its own type,
      extensions included:
      [each Sel.all { column = (fun n x -> n := rows - count x) }]. [column] is
      polymorphic, so it applies only operations that take every type; each is
      checked against the column's type when the verb is applied: {!sum} of a
      string column is a problem. An extension column is read without its
      declaration, so operations that order its values ({!min}, {!( < )},
      {!rank}, …) or compute with them ({!sum}, {!mean}, …) are problems, and
      those that count, move, select or compare them for equality apply.
      [column] runs when the verb is applied and must be pure. *)

  (** {1:literals Literals} *)

  val int : int -> (int, 's) t
  (** [int n] is the integer [n]. *)

  val float : float -> (float, 's) t
  (** [float x] is the float [x]. It may round to the type it takes. *)

  val bool : bool -> (bool, 's) t
  (** [bool b] is the boolean [b]. *)

  val string : string -> (string, 's) t
  (** [string s] is the text [s].

      Raises [Invalid_argument] if [s] is not valid UTF-8. *)

  val instant : Time.instant -> (Time.instant, 's) t
  (** [instant t] is the instant [t]. *)

  val span : Time.span -> (Time.span, 's) t
  (** [span d] is the span [d]. *)

  val date : Time.date -> (Time.date, 's) t
  (** [date d] is the date [d]. *)

  val null : ('a, 's) t
  (** [null] is null, typed by what it meets. *)

  (** {1:elementwise Elementwise operations} *)

  val ( + ) : (int, 's) t -> (int, 's) t -> (int, 's) t
  (** [a + b] is the sum of [a] and [b], wrapping on overflow as nx's integers
      do. Its type is the operands' common type, as for {!( - )}, {!( * )},
      {!( / )} and {!( mod )}. *)

  val ( - ) : (int, 's) t -> (int, 's) t -> (int, 's) t
  (** [a - b] is the difference of [a] and [b]. *)

  val ( * ) : (int, 's) t -> (int, 's) t -> (int, 's) t
  (** [a * b] is the product of [a] and [b]. *)

  val ( / ) : (int, 's) t -> (int, 's) t -> (int, 's) t
  (** [a / b] is the quotient of [a] by [b], truncated toward zero, and null
      where [b] is zero. *)

  val ( mod ) : (int, 's) t -> (int, 's) t -> (int, 's) t
  (** [a mod b] is the remainder of [a] by [b], of [a]'s sign, and null where
      [b] is zero. *)

  val ( +. ) : (float, 's) t -> (float, 's) t -> (float, 's) t
  (** [a +. b] is the IEEE 754 sum of [a] and [b]. Its type is the operands'
      common type, as for {!( -. )}, {!( *. )}, {!( /. )} and {!( ** )}. *)

  val ( -. ) : (float, 's) t -> (float, 's) t -> (float, 's) t
  (** [a -. b] is the difference of [a] and [b]. *)

  val ( *. ) : (float, 's) t -> (float, 's) t -> (float, 's) t
  (** [a *. b] is the product of [a] and [b]. *)

  val ( /. ) : (float, 's) t -> (float, 's) t -> (float, 's) t
  (** [a /. b] is the quotient of [a] by [b]. *)

  val ( ** ) : (float, 's) t -> (float, 's) t -> (float, 's) t
  (** [a ** b] is [a] to the power [b]. *)

  val ( = ) : ('a, 's) t -> ('a, 's) t -> (bool, 's) t
  (** [a = b] is [true] iff [a] and [b] are equal in talon's total order
      ({!Type.compare_value}), so [nan = nan] and [-0. = 0.], and null if either
      is null. The operands meet at their common type. *)

  val ( <> ) : ('a, 's) t -> ('a, 's) t -> (bool, 's) t
  (** [a <> b] is [not (a = b)]. *)

  val ( < ) : ('a, 's) t -> ('a, 's) t -> (bool, 's) t
  (** [a < b] is [true] iff [a] comes before [b] in talon's total order, so
      [x > float 15.] holds for NaN, and null if either is null. Operands of an
      extension type need a declaration made with [~ordered:true]. *)

  val ( > ) : ('a, 's) t -> ('a, 's) t -> (bool, 's) t
  (** [a > b] is [b < a]. *)

  val ( <= ) : ('a, 's) t -> ('a, 's) t -> (bool, 's) t
  (** [a <= b] is [a < b || a = b]. *)

  val ( >= ) : ('a, 's) t -> ('a, 's) t -> (bool, 's) t
  (** [a >= b] is [b <= a]. *)

  val ( && ) : (bool, 's) t -> (bool, 's) t -> (bool, 's) t
  (** [a && b] is Kleene's conjunction: [false] if either is [false], else null
      if either is null. Both operands are computed. *)

  val ( || ) : (bool, 's) t -> (bool, 's) t -> (bool, 's) t
  (** [a || b] is Kleene's disjunction: [true] if either is [true], else null if
      either is null. *)

  val not : (bool, 's) t -> (bool, 's) t
  (** [not a] is the negation of [a], null where [a] is. *)

  val if_ : (bool, 's) t -> ('a, 's) t -> ('a, 's) t -> ('a, 's) t
  (** [if_ c a b] is [a] where [c] is [true] and [b] where [c] is [false] or
      null. [a] and [b] meet at their common type. *)

  val is_null : ('a, 's) t -> (bool, 's) t
  (** [is_null a] is [true] where [a] is null and [false] elsewhere. It is never
      null. *)

  val coalesce : ('a, 's) t list -> ('a, 's) t
  (** [coalesce es] is the first of [es] that is not null, or null. [es] meet at
      their common type. *)

  val is_in : 'a list -> ('a, 's) t -> (bool, 's) t
  (** [is_in vs a] is [true] iff [a] is the same key as one of [vs] (see
      {!Type.compare_value}), so it is [false], never null, where [a] is null:
      [not (is_in vs a)] is [true] there, whereas [not (a = v0 || a = v1)] is
      null. [a]'s type must hold each of [vs]; an extension's values are encoded
      with its declaration. *)

  val cut : 'a array -> ('a, 's) t -> (int, 's) t
  (** [cut edges a] is the number of [edges] at or below [a] in talon's total
      order, as [int64]: bins are half-open, a value below every edge is [0] and
      one at or above every edge is the number of distinct edges. [edges] need
      not be sorted, and a repeated edge counts once. [a]'s type must hold each
      edge, and an extension type needs an ordered declaration. [edges] is
      copied. *)

  val cast : 'b Type.t -> ('a, 's) t -> ('b, 's) t
  (** [cast ty a] converts [a]'s values to [ty]:
      - between [bool], integer, float and decimal types: integers take exact
        values only, so a fractional, infinite or NaN float, or a value out of
        range, is a data error; floats round to nearest; decimals round to
        nearest at their scale, ties away from zero; [bool] takes [0] and [1]
        only, and gives [0] and [1];
      - between [string] and categorical types: the text is kept, and a
        categorical takes only the strings of its dictionary;
      - between datetimes that both have a zone, or both have none, between
        durations, and between clocks: values are kept, and a value that is not
        a whole number of the new unit is a data error;
      - lists element by element, records with the same field names field by
        field, and tensors of the same shape element by element, as [Nx.cast]
        does.

      Any other pair is a problem that names the function that converts it, if
      one does: {!Str.parse} and {!Temporal.parse} for text to values,
      {!Temporal.format} for dates, clocks and datetimes to text,
      {!Temporal.localize} for zones, [Ext.storage] and [Ext.wrap] for
      extensions. [a] needs a column type. *)

  type fn = { f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t }
  (** The type for elementwise nx functions of one argument that preserve its
      dtype, such as [Nx.exp]. *)

  val nx : fn -> ('a, 's) t -> ('a, 's) t
  (** [nx { f } a] applies [f] to [a]'s values with nx's semantics, IEEE's for
      floats: [nx { f = Nx.exp } x]. [a] has an nx dtype: an integer, float or
      [bool] type. When the verb is applied, talon calls [f] once on a traced
      value of that dtype and records the operations [f] performs; an [f] that
      moves, reduces or reshapes its argument is a problem, and so is one that
      ignores it, which would lose its nulls. *)

  type fn2 = { f2 : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t }
  (** The type for elementwise nx functions of two arguments of one dtype that
      preserve it, such as [Nx.atan2]. *)

  val nx2 : fn2 -> ('a, 's) t -> ('a, 's) t -> ('a, 's) t
  (** [nx2 { f2 } a b] is like {!nx} for two arguments, which meet at their
      common type: [nx2 { f2 = Nx.atan2 } y x]. *)

  (** {1:reductions Reductions}

      A reduction takes a [row] expression to one value per frame. Over no
      values, {!sum}, {!count} and {!n_unique} are [0], {!collect} is the empty
      list, and every other reduction is null. *)

  val rows : (int, agg) t
  (** [rows] is the number of rows of the frame, as [int64]. *)

  val count : ('a, row) t -> (int, agg) t
  (** [count a] is the number of non-null values of [a], as [int64]. *)

  val sum : ('a, row) t -> ('a, agg) t
  (** [sum a] is the sum of [a]'s values: [int64] over integers, [a]'s type over
      floats and durations, and [decimal[18, s]] over decimals of scale [s].
      Integer sums wrap as nx's integers do; a duration or decimal sum that
      overflows is a data error. Other types are a problem. *)

  val min : ('a, row) t -> ('a, agg) t
  (** [min a] is the least of [a]'s values in talon's total order, of [a]'s
      type. An extension type needs an ordered declaration. *)

  val max : ('a, row) t -> ('a, agg) t
  (** [max a] is the greatest of [a]'s values, like {!min}. *)

  val first : ('a, row) t -> ('a, agg) t
  (** [first a] is [a]'s first non-null value in frame order. *)

  val last : ('a, row) t -> ('a, agg) t
  (** [last a] is [a]'s last non-null value in frame order. *)

  val only : ('a, row) t -> ('a, agg) t
  (** [only a] is [a]'s one distinct non-null value, and a data error naming the
      frame when [a] has several. *)

  val mean : ('a, row) t -> (float, agg) t
  (** [mean a] is the arithmetic mean of [a]'s values, as [float64]. [a] is an
      integer or float expression, as for {!std}, {!var}, {!median} and
      {!quantile}; other types are a problem. *)

  val std : ('a, row) t -> (float, agg) t
  (** [std a] is the sample standard deviation of [a]'s values, dividing by n -
      1, and null for fewer than two values. *)

  val var : ('a, row) t -> (float, agg) t
  (** [var a] is the sample variance of [a]'s values, like {!std}. *)

  val median : ('a, row) t -> (float, agg) t
  (** [median a] is [quantile 0.5 a]. *)

  val quantile : float -> ('a, row) t -> (float, agg) t
  (** [quantile p a] is the [p]-quantile of [a]'s values, interpolating linearly
      between the two nearest ranks.

      Raises [Invalid_argument] if [p] is not in \[[0];[1]\]. *)

  val ewm : alpha:float -> ('a, row) t -> (float, agg) t
  (** [ewm ~alpha a] is the exponentially weighted mean of [a]'s values in frame
      order, as [float64]: y is the first value, then (1 − [alpha])·y +
      [alpha]·x at each next value x, and [ewm ~alpha a] is the last y. A NaN
      propagates to every later y. [Kit.cumulative (ewm ~alpha a)] is the
      smoothed series.

      Raises [Invalid_argument] unless [0. < alpha && alpha <= 1.]. *)

  val n_unique : ('a, row) t -> (int, agg) t
  (** [n_unique a] is the number of distinct keys of [a], null being one key, as
      [int64]. *)

  val arg_min : ('a, row) t -> (int, agg) t
  (** [arg_min a] is the zero-based position in the frame of [a]'s first least
      value, as [int64]. An extension type needs an ordered declaration. *)

  val arg_max : ('a, row) t -> (int, agg) t
  (** [arg_max a] is the position of [a]'s first greatest value, like
      {!arg_min}. *)

  val collect : ('a, row) t -> ('a array, agg) t
  (** [collect a] is the list of [a]'s values in frame order, nulls included, of
      type [list[t]] for [a]'s type [t]. *)

  (** {1:frames Frames} *)

  val over : ?by:string list -> ?order:Order.t list -> ('a, 's) t -> ('a, row) t
  (** [over ~by ~order e] evaluates [e] in the enclosing frame, partitioned by
      the columns [by] (key identity, null being one key) and each partition
      ordered by [order]. A reduction is broadcast over its partition's rows,
      and a row expression such as {!shift} or {!rank} runs within its partition
      and returns its values to their rows. [by] defaults to no columns, one
      partition; [order] to none, the frame's order. At the top of [derive] or
      [filter], [over (mean x)] is the mean of the whole input, and it blocks
      the pipeline. Its type is [e]'s. A column named twice in [by] or in
      [order] is a problem, and so is an [order] key on a column whose type is
      or holds an extension type: order by its storage, derived first. *)

  val rolling : Window.t -> ('a, agg) t -> ('a, row) t
  (** [rolling w e] is, for each row, [e] over the row's window [w], as if the
      window were [e]'s frame:
      [rolling (Window.rows ~before:6 ~after:0) (mean x)]. A window with too few
      values is a comparison away:
      [if_ (rolling w (count x) >= int 7) (rolling w (mean x)) null]. Its type
      is [e]'s. *)

  val shift : int -> ('a, row) t -> ('a, row) t
  (** [shift n a] is [a]'s value [n] rows earlier in the frame, or [-n] rows
      later if [n] is negative, and null past the frame's edges. *)

  val rank : ('a, row) t -> (int, row) t
  (** [rank a] is the 1-based rank of [a]'s value among the frame's non-null
      values in talon's total order, ties taking the lowest rank (SQL's [RANK]),
      as [int64], and null where [a] is null. An extension type needs an ordered
      declaration. *)

  (** {1:ocaml OCaml values}

      These functions compute in OCaml, once per row, at native speed. They must
      be pure; an exception they raise propagates from the run. *)

  val const : 'a -> ('a, 's) t
  (** [const v] is [v] on every row. It is typed by what it meets, an
      extension's value being encoded with its declaration; a type that does not
      hold [v] is a problem. *)

  val ( $ ) : ('a -> 'b, 's) t -> ('a, 's) t -> ('b, 's) t
  (** [f $ a] applies [f]'s function to [a]'s value, once per row where no
      argument is null, and is null elsewhere: [const mk $ carrier $ delay]. An
      argument is decoded with its column type, so it needs one, unless it is
      itself an OCaml value: a [const], a [$] result or an {!option}. The result
      is typed by what it meets, or by {!store}. *)

  val option : ('a, 's) t -> ('a option, 's) t
  (** [option a] is [Some v] where [a] is [v] and [None] where [a] is null. It
      is never null, and is an argument of {!( $ )} or what [Query.values]
      decodes. [a] needs a column type. *)

  val of_option : ('a option, 's) t -> ('a, 's) t
  (** [of_option a] is [v] where [a] is [Some v] and null where it is [None]. It
      is typed by what it meets, or by {!store}. *)

  val store : 'b Type.t -> ('b, 's) t -> ('b, 's) t
  (** [store ty a] is [a] typed as [ty], as if [a] met an operand of type [ty]:
      a literal, [null], [const], [$] or {!of_option} result takes [ty], and an
      [a] of a type that [ty] contains is widened to it. A literal or {!const}
      value that [ty] does not hold is a problem, and a [$] or {!of_option}
      result that it does not hold a data error. *)

  val batch :
    (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
    (('a, 'b) Nx.t, row) t ->
    (('c, 'd) Nx.t, row) t
  (** [batch f x] applies [f] to batches of the tensor expression [x], its cells
      stacked as [(n, …shape)], for a model's forward pass. [f] must be
      row-separable: [f (a ++ b)] is [f a ++ f b]. It sees zeros under nulls,
      and its result is null where [x] is. When the verb is applied, talon calls
      [f] on an empty batch to learn the result's dtype and cell shape, which
      give its type. *)

  (** {1:nested Nested values}

      No expression reads a list's elements. A computation on them is
      [Query.unnest], then expressions, then [Query.aggregate ~by] the row's id,
      where {!collect} rebuilds a list. *)

  val record : 's out list -> (Record.t, 's) t
  (** [record os] is the record whose fields are the outputs [os], in order. An
      output name that appears twice is a problem. *)

  val field : 'a Kind.t -> string -> (Record.t, 's) t -> ('a, 's) t
  (** [field k name r] is the field [name] of the record [r], which [k] must
      bind, null where [r] is null. *)

  val unpack : (Record.t, row) t -> row out
  (** [unpack r] outputs each field of the record [r] as a column named after
      it, in order. *)

  (** {1:text Text} *)

  (** Text.

      Text counts and slices Unicode scalar values and maps case with the full
      Unicode mappings. Categorical values are text. *)
  module Str : sig
    type pattern
    (** The type for patterns: what text is matched against. *)

    val literal : string -> pattern
    (** [literal s] matches [s] anywhere in the text.

        Raises [Invalid_argument] if [s] is empty or not valid UTF-8, as
        {!prefix} and {!suffix} do. *)

    val prefix : string -> pattern
    (** [prefix s] matches [s] at the start of the text. *)

    val suffix : string -> pattern
    (** [suffix s] matches [s] at the end of the text. *)

    val pieces : string list -> pattern
    (** [pieces ss] matches the strings [ss] in order without overlap:
        [matches (pieces [ "special"; "requests" ])] is SQL's
        [LIKE '%special%requests%'].

        Raises [Invalid_argument] if [ss] is empty or holds an empty or
        non-UTF-8 string. *)

    val length : (string, 's) t -> (int, 's) t
    (** [length a] is the number of Unicode scalar values of [a], as [int64]. *)

    val slice : offset:int -> length:int -> (string, 's) t -> (string, 's) t
    (** [slice ~offset ~length a] is the at most [length] scalar values of [a]
        from [offset], counted from the end when [offset] is negative, as
        [Query.slice] counts rows.

        Raises [Invalid_argument] if [length < 0]. *)

    val lower : (string, 's) t -> (string, 's) t
    (** [lower a] is [a] mapped to lowercase with the full Unicode mapping. *)

    val upper : (string, 's) t -> (string, 's) t
    (** [upper a] is [a] mapped to uppercase with the full Unicode mapping:
        ["ß"] gives ["SS"]. *)

    val matches : pattern -> (string, 's) t -> (bool, 's) t
    (** [matches p a] is [true] iff [p] matches [a]. *)

    val parse : 'a Type.t -> (string, 's) t -> ('a, 's) t
    (** [parse ty a] is the value of type [ty] that the text [a] writes, in the
        forms that [talon.csv] reads: [true] and [false]; decimal integers with
        an optional sign; decimal or scientific floats, [inf], [-inf] and [nan];
        decimals; ISO 8601 dates; and ISO 8601 datetimes, with an offset or [Z]
        exactly when [ty] has a zone. Any other text, and a value that [ty] does
        not hold, is a data error. [ty] is a [bool], integer, float, decimal,
        [date] or [datetime] type, or a categorical, which takes only the
        strings of its dictionary. *)
  end

  (** {1:time Time} *)

  (** Temporal values.

      Dates, clocks and datetimes without a zone are wall-clock values: their
      calendar is read as it is, and they take no zone. A datetime with a zone
      holds instants, and reading its calendar takes the [~zone] whose wall
      clock reads them. Where a zone's wall clock reads a time twice or never,
      {!floor} and {!offset} keep the instant's offset when it applies and
      otherwise move past the gap; {!localize} takes the policy explicitly. *)
  module Temporal : sig
    val add : ('a, 's) t -> (Time.span, 's) t -> ('a, 's) t
    (** [add a d] is [a] advanced by [d], of [a]'s type: a datetime, a duration
        or a clock, [d] being a duration of [a]'s unit or a coarser one, or a
        date, [d] being whole days. A literal [d] that is not whole days is a
        problem, and another a data error. A result out of range, or a clock
        outside its day, is a data error. *)

    val diff : ('a, 's) t -> ('a, 's) t -> (Time.span, 's) t
    (** [diff a b] is the span from [b] to [a]: a duration of their common unit
        for datetimes, durations and clocks, which meet as {!Type.common} says,
        and [duration[s]] for dates. *)

    type field =
      [ `Year
      | `Month
      | `Day
      | `Hour
      | `Minute
      | `Second
      | `Nanosecond
      | `Weekday
      | `Yearday ]
    (** The type for calendar fields:
        - [`Year], astronomical: year [0] is 1 BC;
        - [`Month], [1] to [12], and [`Day], the day of the month, [1] to [31];
        - [`Hour], [0] to [23], [`Minute] and [`Second], [0] to [59];
        - [`Nanosecond], the nanosecond within the second;
        - [`Weekday], the ISO day of the week, [1] for Monday to [7];
        - [`Yearday], the day of the year, [1] to [366]. *)

    val field : field -> ?zone:Tz.zone -> ('a, 's) t -> (int, 's) t
    (** [field f ?zone a] is the field [f] of [a], as [int64]:
        [field `Year orderdate], [field `Hour ~zone:paris ts]. [a] is a date, a
        clock or a datetime. A datetime with a zone holds instants, read on
        [zone]'s wall clock, which it requires; dates, clocks and datetimes
        without a zone are wall-clock values and take no [zone]. A date has no
        time-of-day field and a clock no calendar field: asking for one is a
        problem. *)

    val floor : ?zone:Tz.zone -> Time.step -> ('a, 's) t -> ('a, 's) t
    (** [floor ?zone step a] is the first instant of the period of [step] that
        holds [a], a date or a datetime, on [zone]'s wall clock when [a] has a
        zone, which then requires [zone]. Periods are counted from 1970-01-01
        00:00, and weeks from Monday 1970-01-05. Where the period's first
        wall-clock time is skipped, it is the first instant after the gap. A
        date floors by calendar steps only, and a datetime by exact steps of
        whole ticks of its unit. Its type is [a]'s.

        Raises [Invalid_argument] if [step] is not positive. *)

    val offset : ?zone:Tz.zone -> Time.step -> ('a, 's) t -> ('a, 's) t
    (** [offset ?zone step a] is [a], a date or a datetime, moved by [step]: an
        exact step moves the instant, and a calendar step moves the wall clock,
        on [zone]'s when [a] has a zone, as for {!floor}; a day of the month
        past the month's end becomes its last day. A date moves by calendar
        steps only, and a datetime by exact steps of whole ticks of its unit.
        Its type is [a]'s. *)

    type policy = [ `Earlier | `Later | `Null | `Fail ]
    (** The type for resolutions of a wall-clock time that a zone reads twice or
        never. A time read twice has two instants; a skipped time has the two
        instants that the offsets before and after the skip give it, the later
        being the time moved past the gap. [`Earlier] and [`Later] pick one,
        [`Null] gives null, and [`Fail] is a data error. *)

    val localize :
      Tz.zone ->
      ambiguous:policy ->
      gap:policy ->
      (Time.instant, 's) t ->
      (Time.instant, 's) t
    (** [localize zone ~ambiguous ~gap a] is the instant at which [zone]'s wall
        clock reads [a], resolved by [ambiguous] where it reads [a] twice and by
        [gap] where it never does. [a] is a datetime without a zone, and the
        result has [a]'s unit and [zone]'s name. *)

    val windows :
      ?zone:Tz.zone ->
      every:Time.step ->
      period:Time.step ->
      ('a, 's) t ->
      ('a array, 's) t
    (** [windows ?zone ~every ~period a] is the starts of the windows that hold
        [a], a date or a datetime, in ascending order: windows start every
        [every], as {!floor} places them, and last [period], as {!offset} moves.
        Its type is [list[t]] for [a]'s type [t]. [unnest], then [aggregate],
        gives hopping windows.

        Raises [Invalid_argument] if [every] or [period] is not positive. *)

    val parse : string -> 'a Type.t -> (string, 's) t -> ('a, 's) t
    (** [parse fmt ty a] reads the text [a] in the format [fmt] as a value of
        [ty], a [date], [datetime] or [clock] type. [fmt] holds the directives
        [%Y] (the year), [%m], [%d], [%H], [%M], [%S] (two digits each), [%f]
        (one to nine digits of a fraction of a second), [%z] ([Z] or a [±hh:mm]
        offset) and [%%]; other characters match themselves. A year is four
        digits, or a sign and at least four digits, as {!Time.Date.pp} writes
        it. A field that [fmt] leaves out is that of 1970-01-01 00:00:00; a date
        reads the day alone, and a clock the time of day. A datetime with [%z]
        needs a zone in [ty], and one without needs none. Text that does not
        match, and a value that [ty] does not hold, are data errors.

        Raises [Invalid_argument] if [fmt] holds another directive. *)

    val format : string -> ('a, 's) t -> (string, 's) t
    (** [format fmt a] writes [a], a date, datetime or clock, in the format
        [fmt], as {!parse} reads it: a datetime in UTC, a clock on 1970-01-01;
        [%f] writes nine digits and [%z] writes [Z]. [%z] needs a datetime with
        a zone.

        Raises [Invalid_argument] as {!parse} does. *)
  end

  (** {1:fmt Formatting} *)

  val pp : Format.formatter -> ('a, 's) t -> unit
  (** [pp ppf e] formats [e] as it is written inside [Expr.( … )], with fewest
      parentheses: [(amount -. over (mean amount)) /. over (std amount)].
      Besides:
      - a handle formats as its column name, quoted as an OCaml string unless it
        is an OCaml lowercase identifier that is neither a keyword nor a value
        of this module;
      - a literal formats as its value: [15], [15.], [nan], ["text"], [true],
        [2024-03-15], [2024-03-15T09:30:00], [15m];
      - an OCaml value of {!const} formats as [<const>], and the functions of
        {!nx}, {!nx2} and {!batch} as [<fn>];
      - an {!is_in} list and {!cut} edges format with their operand's type once
        bound, and as [[…]] before. *)
end

module Col : sig
  (** Column handles.

      A handle names a column and the kind its values read as. It is bound to no
      table: a verb binds it to its input when it is applied, so a module of
      handles serves as a schema. A handle of kind [k] binds a column whose type
      [k] reads ({!Kind.provably_equal}); a missing column, or one of another
      kind, is a problem the verb reports. No handle binds an extension column:
      [Ext.col] does. *)

  val v : 'a Kind.t -> string -> ('a, Expr.row) Expr.t
  (** [v k name] is the column [name] read as [k]:
      [v (Kind.list Kind.int) "tokens"]. *)

  val bool : string -> (bool, Expr.row) Expr.t
  (** [bool name] is [v Kind.bool name]. *)

  val int : string -> (int, Expr.row) Expr.t
  (** [int name] is [v Kind.int name], which binds every integer type. *)

  val float : string -> (float, Expr.row) Expr.t
  (** [float name] is [v Kind.float name], which binds every float type. *)

  val string : string -> (string, Expr.row) Expr.t
  (** [string name] is [v Kind.string name], which binds [string] and
      categorical types. *)

  val binary : string -> (Binary.t, Expr.row) Expr.t
  (** [binary name] is [v Kind.binary name]. *)

  val decimal : string -> (Decimal.t, Expr.row) Expr.t
  (** [decimal name] is [v Kind.decimal name]. *)

  val date : string -> (Time.date, Expr.row) Expr.t
  (** [date name] is [v Kind.date name]. *)

  val instant : string -> (Time.instant, Expr.row) Expr.t
  (** [instant name] is [v Kind.instant name], which binds every datetime type.
  *)

  val span : string -> (Time.span, Expr.row) Expr.t
  (** [span name] is [v Kind.span name], which binds every duration and clock
      type. *)
end

module Ext : sig
  (** Extension declarations.

      An extension type is a named type stored as another ({!Type.ext}). A
      declaration gives it OCaml values ['e], converted to and from its storage
      values ['s]. It is the only way to read an extension column's values or
      compute on them: no {!Col} handle binds one. A library's claim to an
      extension's name is a convention, as in Arrow; what it controls is ['e],
      whose values come only from [dec] and [enc].

      Without a declaration, an extension column moves through the order-free
      structural operations: [keep], [Expr.each], filtering, gathering,
      appending, joining and grouping by key identity on its storage. Its order,
      and computation on its values, need a declaration. *)

  type ('e, 's) t
  (** The type for declarations of extensions whose values read as ['e] and are
      stored as ['s]. *)

  val v :
    name:string ->
    ?metadata:string ->
    ordered:bool ->
    's Type.t ->
    dec:('s -> 'e) ->
    enc:('e -> 's) ->
    ('e, 's) t
  (** [v ~name ~metadata ~ordered storage ~dec ~enc] declares the extension type
      [Type.ext ~name ~metadata storage] with:
      - [dec], which reads a stored value as ['e]: [Query.values] and
        [Column.values] decode with it;
      - [enc], which stores an ['e]: literals, {!Expr.const} values and
        {!Expr.is_in} values encode with it. It must be injective;
      - [ordered], a promise that the order of stored values is the order of the
        values, as for an epoch stored normalized. With [~ordered:false],
        {!Expr.min}, {!Expr.max}, {!Expr.arg_min}, {!Expr.arg_max},
        {!Expr.rank}, {!Expr.cut} and [<] are problems on its values. Sort and
        join keys are names, which no declaration binds: sort an extension
        column by its storage, derived first.

      [metadata] defaults to [""].

      Raises [Invalid_argument] as {!Type.ext} does. *)

  val col : ('e, 's) t -> string -> ('e, Expr.row) Expr.t
  (** [col e name] is the column [name] read through [e]. It binds only a column
      whose type is [e]'s: the same name, metadata and storage type, so a
      declaration of metres never binds a column of kilograms. *)

  val storage : ('e, 's) t -> ('e, 'sh) Expr.t -> ('s, 'sh) Expr.t
  (** [storage e x] is the stored values of [x], an expression of [e]'s type, as
      [e]'s storage type. *)

  val wrap : ('e, 's) t -> ('s, 'sh) Expr.t -> ('e, 'sh) Expr.t
  (** [wrap e x] is [x]'s values as values of [e]'s type: [x] meets [e]'s
      storage type, which must contain [x]'s type. [wrap e (storage e x)] has
      [x]'s values. *)
end

module Source : sig
  (** Sources: tables read in batches.

      A source is data that a query reads when it runs, such as a file: its
      schema is known when it is built, and its rows arrive in batches. Formats
      build sources with {!v}, and a program's own format is a peer of the
      shipped ones. [Query.of_source] makes a query of one.

      A source is a contract between talon and its author:
      - {b Pushdown.} Before reading, talon asks the source about each conjunct
        of the filters on it, translated into a {!Pred.t}, and the source
        answers with no IO whether it can apply it ({!answer}).
      - {b Request.} Talon then asks for the source's parts with a {!request}:
        the columns it reads, the conjuncts the source can use, and a limit.
        Each place in a plan that reads the source is a read of its own, with
        its own request.
      - {b Pull.} Talon opens a part's {!reader}, pulls batches with [next], and
        calls [close] once it has no more use for the reader.

      The rows of a source are those of its parts in part order, then batch
      order. They are fixed by the data and the request, never by the machine's
      core count. *)

  (** {1:pushdown Pushdown} *)

  (** The type for a source's answers about a conjunct. *)
  type answer =
    | Exact
        (** The source applies the conjunct exactly: it yields no row that fails
            it, so talon removes the conjunct from the plan. *)
    | Inexact
        (** The source may use the conjunct to skip rows that fail it, as row
            group statistics do, and never skips a row that passes it. Talon
            applies the conjunct again. *)
    | Unsupported  (** The source ignores the conjunct. Talon applies it. *)

  (** Predicates on one row: the filters that sources apply.

      A predicate compares columns of the source against values. It means what
      its filter means in talon: a comparison uses talon's total order
      ({!Type.compare_value}) and is null when the column's value is null, and
      {!Not}, {!And} and {!Or} are Kleene's, so a row passes a predicate iff the
      predicate is [true] on it. A source answers {!Exact} only for a predicate
      whose meaning it honours entirely: for example, a float column's minimum
      and maximum prune a row group only when the source knows the group holds
      no NaN, since NaN passes [>], [>=] and [<>]. *)
  module Pred : sig
    type value =
      | Value : 'a Type.t * 'a -> value
          (** The type for values. [Value (ty, v)] is [v] at the type [ty] that
              the comparison is made in, which holds [v] ({!Type.holds}). It is
              the column's type, or a type that contains it ({!Type.common}),
              such as [string] for a categorical column, which then compares by
              text rather than by dictionary position. For an extension column
              it is the storage type, and [v] is the value's storage. [v] is a
              value of [ty] exactly, as [ty] stores it: a [float32] value is
              rounded to [float32]. *)

    (** The type for predicates. *)
    type t =
      | Cmp of string * [ `Eq | `Ne | `Lt | `Le | `Gt | `Ge ] * value
          (** [Cmp (c, op, v)] compares the column [c] to [v] with [op]: [`Eq]
              is [=], [`Ne] is [<>], [`Lt] is [<], [`Le] is [<=], [`Gt] is [>]
              and [`Ge] is [>=]. It is null where [c] is null. *)
      | In of string * value list
          (** [In (c, vs)] is [true] iff the column [c] is the same key as one
              of [vs], which have one type, and [false] elsewhere, null [c]
              included. [In (c, [])] is [false]. *)
      | Null of string  (** [Null c] is [true] iff the column [c] is null. *)
      | Valid of string
          (** [Valid c] is [true] iff the column [c] is not null. *)
      | And of t list
          (** [And ps] is [false] if one of [ps] is, else null if one is null,
              else [true]. [And []] is [true]. *)
      | Or of t list
          (** [Or ps] is [true] if one of [ps] is, else null if one is null,
              else [false]. [Or []] is [false]. *)
      | Not of t  (** [Not p] is the negation of [p], null where [p] is. *)
  end

  (** {1:reading Reading} *)

  type request = {
    columns : string list;
        (** The columns talon reads, distinct, in schema order. It may be empty,
            as when a query only counts rows. *)
    filters : Pred.t list;
        (** The conjuncts that the source answered {!Exact} or {!Inexact}, in
            plan order. The source applies each it answered {!Exact}. *)
    limit : int option;
        (** [Some n] when talon reads no more than the first [n] rows that the
            request yields, so the source may stop after them. It is [None]
            unless every conjunct of the filters on the source is {!Exact}. *)
  }
  (** The type for what talon asks of a source when it runs. *)

  type reader = {
    next : unit -> (table option, Error.t) result;
        (** [next ()] is [Ok (Some b)] with the next batch [b] of the part,
            whose columns are the request's, in its order, with the source's
            types; [Ok None] at the end of the part; or [Error e]. A batch may
            have no rows. A batch with other columns or types raises
            [Invalid_argument] when talon reads it. *)
    close : unit -> unit;
        (** [close ()] releases the reader. Talon calls it once, when [next] has
            returned [Ok None] or [Error _], when it needs no more of the part's
            rows, or when the run fails, and calls [next] no more afterwards. *)
  }
  (** The type for readers of a part's batches. *)

  type part = {
    rows : int option;
        (** [Some n] promises that the part yields exactly [n] rows for the
            request, and talon trusts it to skip the part without reading it;
            [None] if unknown. *)
    open_ : unit -> (reader, Error.t) result;
        (** [open_ ()] opens a reader on the part. Talon opens a part at most
            once, and parts must not depend on one another. *)
  }
  (** The type for parts: the units a source reads, such as files or row groups.
  *)

  (** {1:sources Sources} *)

  type t
  (** The type for sources. Two sources are the same iff they are one value. *)

  val v :
    name:string ->
    schema:Schema.t ->
    ?rows:int ->
    ?sorted:Order.t list ->
    ?pushdown:(Pred.t -> answer) ->
    (request -> (part list, Error.t) result) ->
    t
  (** [v ~name ~schema ?rows ?sorted ?pushdown parts] is the source called
      [name] that yields rows of the columns [schema], with:
      - [rows], the number of rows the source yields for a request without
        filters, which plans print, and which turns a slice from the end into
        one from the start. Defaults to unknown.
      - [sorted], the order the rows come in, by talon's total order, which lets
        an ordered join skip a sort. Talon checks it as rows arrive, and a row
        out of order fails the run. Defaults to no order.
      - [pushdown], which answers for one conjunct whether the source applies
        it. Talon calls it when it plans a run, any number of times: it must be
        pure and do no IO. Defaults to {!Unsupported} for every conjunct.

      Talon calls [parts] once for each place in the optimized plan that reads
      the source, with that place's request, and [parts] gives the source's
      parts in order, or the [Error] that fails the run. A source that can be
      read only once, such as a stream, documents that it reads once, and its
      [parts] is an [Error] after its first call: a plan that needs it in
      several places reads a table that [run] makes of it first.

      Plans print a source as [name] followed by its number of columns, and by
      [rows] when given, as in [parquet "carriers.parquet" (2 columns)], so
      [name] says what the source reads: [csv "flights.csv"].

      Raises [Invalid_argument] if [name] is empty, is not valid UTF-8 or holds
      a control character, if [rows] is negative, or if a key of [sorted] names
      no column of [schema], names a column twice, or names a column whose type
      is or contains an extension type, which has no order of its own. *)
end

module Join : sig
  (** Joins: conditions, kinds and counts.

      [left |> Query.join ~on right] pairs the rows of [left] and [right] that
      the condition [on] matches. A condition is a value, a conjunction of
      {e atoms} built with {!( && )} inside [Join.( … )]:
      [Join.(keys [ "ticker" ] && closest (ge "ts" "quote_ts"))]. Atoms name a
      left column, then a right column.

      {b Algorithms.} Each condition runs as one algorithm, with the cost that
      [Query.join] states, and a conjunction that no algorithm runs raises when
      it is built. The conditions are:
      - {e equality}: one or more equality atoms ({!keys}, {!eq});
      - {e inequality}: equality atoms and one or more inequality atoms ({!lt},
        {!le}, {!gt}, {!ge}), which all compare to one right column. A range on
        two right columns is a join on one of them, then a [filter];
      - {e closest} and {e nearest}: equality atoms and one {!closest} or
        {!nearest} atom;
      - {!position}, alone;
      - {!all}, every pair, which is also the condition with no atom: [all && c]
        is [c].

      {b Keys.} Equality atoms match by key identity ({!Type.compare_value},
      null being one key), so null keys match. The columns of an inequality,
      {!closest} or {!nearest} atom order by talon's total order, and a null in
      one of them fails the run.

      {b Columns.} A {!Semi} or {!Anti} join has the left columns. Another join
      has the left columns, then the right columns but those of equality atoms,
      in order: the key of [eq l r] appears once, as [l]. In a {!Full} join that
      key has the common type of [l] and [r] ({!Type.common}), since an
      unmatched right row gives it [r]'s value; every other column keeps its
      type. A name on both sides is a problem, and talon adds no suffixes:
      rename one side first. *)

  (** {1:conditions Conditions} *)

  type cond
  (** The type for conditions: conjunctions of atoms that an algorithm runs. *)

  val keys : string list -> cond
  (** [keys ns] matches the rows whose columns [ns] are the same keys on both
      sides: it is [eq n n] for each [n] of [ns].

      Raises [Invalid_argument] if [ns] is empty, since a join on no key is
      {!all}, written so, or if [ns] names a column twice. *)

  val eq : string -> string -> cond
  (** [eq l r] matches the rows whose left column [l] and right column [r] are
      the same key. [l] and [r] must meet ({!Type.common}). *)

  val lt : string -> string -> cond
  (** [lt l r] matches the rows whose left column [l] is less than the right
      column [r]. [l] and [r] must meet, at a type that orders: one that neither
      is nor contains an extension type. *)

  val le : string -> string -> cond
  (** [le l r] matches where [l] is at most [r], like {!lt}. *)

  val gt : string -> string -> cond
  (** [gt l r] matches where [l] is greater than [r], like {!lt}. *)

  val ge : string -> string -> cond
  (** [ge l r] matches where [l] is at least [r], like {!lt}. *)

  val closest : ?within:('a, Expr.row) Expr.t -> cond -> cond
  (** [closest ?within c] matches each left row with the right row that best
      satisfies the inequality [c]: [closest (ge "ts" "quote_ts")] is the latest
      quote at or before [ts]. Under {!ge} and {!gt} that is the right row with
      the greatest right column, and under {!le} and {!lt} the one with the
      least. Among right rows tied on it, the last in right order wins. [within]
      keeps the match only if the two columns differ by at most [within], a
      literal of the columns' difference:
      - the columns' common type for integer, float and decimal columns,
        [Expr.int 5], [Expr.float 0.5];
      - a span of the columns' unit for datetimes, durations and clocks, and of
        whole days for dates, [Expr.span (Time.Span.s 5)].

      [within] defaults to no bound. Over other types there is no difference, so
      [within] is a problem. A [within] that is not a literal, or that is
      negative, is a problem.

      Raises [Invalid_argument] if [c] is not one inequality atom. *)

  val nearest : ?within:('a, Expr.row) Expr.t -> string -> string -> cond
  (** [nearest ?within l r] matches each left row with the right row whose
      column [r] is nearest to its column [l], in either direction. A tie in
      distance goes to the smaller key, and among right rows tied on the key the
      last in right order wins. [within] bounds the distance as for {!closest}.
      [l] and [r] must have a difference, as for {!closest}'s [within]. *)

  val position : cond
  (** [position] matches row i of the left with row i of the right. *)

  val all : cond
  (** [all] matches every pair of rows. *)

  val ( && ) : cond -> cond -> cond
  (** [c0 && c1] matches the pairs that both [c0] and [c1] match.

      Raises [Invalid_argument] if no algorithm runs the conjunction: one that
      holds {!position} and another atom, two {!closest} or {!nearest} atoms,
      one of them and an inequality atom, or inequality atoms on two right
      columns; or if it holds one equality atom twice. *)

  (** {1:kinds Kinds and counts} *)

  (** The type for join kinds: which rows a join keeps. *)
  type kind =
    | Inner  (** The matched pairs. *)
    | Left
        (** The matched pairs, and each left row without a match, the right
            columns null. *)
    | Full
        (** As {!Left}, then each right row without a match, the left columns
            null except the keys, which take the right row's. *)
    | Semi  (** Each left row with a match, once, with the left columns only. *)
    | Anti  (** Each left row without a match, with the left columns only. *)

  (** The type for the number of matches each row of a side must have. A row
      with another number fails the run, naming the side, the key and the count.
  *)
  type count =
    | Any  (** No constraint. *)
    | At_most_one  (** Zero or one. *)
    | One  (** Exactly one. *)
    | At_least_one  (** One or more. *)
end

module Query : sig
  (** Queries: descriptions of tables to compute.

      A query is a plan: a table or a {!Source.t}, transformed by verbs.
      Building one reads no data. A pipeline is written inside [Query.( … )],
      and the expressions of its verbs inside [Expr.( … )]:
      {[
      Query.(
        of_source flights
        |> filter Expr.(delay > float 15.)
        |> aggregate ~by:[ "carrier" ] Expr.[ "mean_delay" := mean delay ]
        |> join
             ~on:(Join.keys [ "carrier" ])
             ~each_left:One (of_source carriers)
        |> sort [ Order.desc "mean_delay" ])
      ]}

      {b Order is contract.} Each verb states the order of its rows, and no verb
      has an ordering flag. Where a verb states a cost, n is the number of rows
      of its input and m that of a join's right input. A verb that {e streams}
      transforms its input batch by batch; one that {e blocks} holds the columns
      it reads until its input ends.

      {b Problems.} A verb checks its arguments against its input's schema when
      it is applied: it binds its expressions ({!Expr} says how), resolves its
      names and selectors, and infers its schema, which {!schema} then returns.
      It collects every problem it finds and raises one [Invalid_argument] whose
      message reports them all:
      {v
      aggregate: 3 problems
        ~by: no column "carier". Did you mean "carrier"?
        "mean_delay" := mean dep_dly
          no column "dep_dly". Did you mean "dep_delay"?
        "late" := mean carrier
          Col.float reads float16, float32 or float64, but "carrier" is string.
        input (19 columns): year int16, month int8, day int8, dep_time int32, sched_dep_time int32, dep_delay float64, arr_time int32, sched_arr_time int32, …
      v}
      - The first line names the verb and counts its problems.
      - An output or a predicate with problems follows, as it was written, with
        its problems below it.
      - A problem of another argument starts with that argument, as [~by:] does.
      - A problem between arguments, such as two outputs of one name, stands
        alone.
      - Problems come in the order of the verb's arguments.
      - The last lines give each input's schema: its number of columns and its
        first eight columns, as {!Schema.pp} formats them, then […] if there are
        more. A join's inputs are [left] and [right], and [append]'s are [input]
        and [rest].

      A name that the input lacks comes with the input's names nearest to it,
      within edit distance 2, or else with all of them.

      {b User functions.} The functions inside a verb's expressions and
      selectors ([Expr.across], [Expr.each], [Sel.where], [Expr.nx],
      [Expr.batch]) run when the verb is applied. They must be pure, and an
      exception they raise propagates from the verb. *)

  type t
  (** The type for queries. *)

  (** {1:leaves Tables and sources} *)

  val of_table : table -> t
  (** [of_table t] is the query of the rows of the table [t]. It costs O(1). *)

  val of_source : Source.t -> t
  (** [of_source s] is the query of the rows of the source [s], which it reads
      when it runs. It costs O(1). *)

  val schema : t -> Schema.t
  (** [schema q] is the names and types of the columns of [q]'s rows. The verb
      that made [q] resolved it, so it costs O(1) and reads no data. *)

  (** {1:verbs Verbs} *)

  val select : Expr.row Expr.out list -> t -> t
  (** [select os q] is [q]'s rows with exactly the columns [os], in order:
      {[
      select
        Expr.[ keep Sel.(names [ "carrier" ]); "late" := delay > float 15. ]
      ]}
      It keeps [q]'s row count and order. Each output reads [q]'s columns, never
      another output. It streams in O(n), except that {!Expr.over},
      {!Expr.rolling} and {!Expr.rank} over the input's rows need the whole
      input: the verb then blocks.

      Its problems are those of binding [os], and two outputs of one name. *)

  val derive : Expr.row Expr.out list -> t -> t
  (** [derive os q] is [q]'s rows with each of [q]'s columns and the outputs
      [os]: an output replaces the column of its name in place, with the
      output's type, and the others follow [q]'s columns, in order. It keeps
      [q]'s row count and order. Each output reads [q]'s columns, never another
      output. It streams in O(n), except that {!Expr.over}, {!Expr.rolling} and
      {!Expr.rank} over the input's rows need the whole input: the verb then
      blocks.

      Its problems are those of binding [os], and two outputs of one name. *)

  val filter : (bool, Expr.row) Expr.t -> t -> t
  (** [filter p q] is the rows of [q] on which [p] is [true], in order, so a row
      on which [p] is null is dropped. Its schema is [q]'s. [p] is a [bool]
      column: an OCaml value, as [const f $ x] is, takes that type. It streams
      in O(n), except that {!Expr.over}, {!Expr.rolling} and {!Expr.rank} over
      the input's rows need the whole input: the verb then blocks.

      Its problems are those of binding [p], and an extension's values, even
      when they read as [bool]: compute [p] from [Ext.storage]. *)

  val sort : Order.t list -> t -> t
  (** [sort ks q] is [q]'s rows ordered by the keys [ks] in turn, each by
      talon's total order in its direction, nulls last unless
      {!Order.nulls_first} (see {!Order}). It is stable: rows equal on every key
      keep [q]'s order. Its schema is [q]'s. It blocks, in O(n log n).

      Its problems are a key that names no column, a key that names the column
      of an earlier key, a key on a column whose type holds an extension type,
      which has no order, and a key on an extension column, which orders only
      through its declaration: sort its storage, derived first, as the problem's
      message shows:
      {[
      derive Expr.[ "k" := Ext.storage e (Ext.col e "t") ]
      |> sort [ Order.asc "k" ]
      |> select Expr.[ keep Sel.(all - names [ "k" ]) ]
      ]} *)

  val slice : offset:int -> length:int -> t -> t
  (** [slice ~offset ~length q] is [q]'s rows at the positions [offset] to
      [offset + length - 1] that [q] has, in order, a negative [offset] counting
      from the end: [slice ~offset:0 ~length:10 q] is [q]'s first ten rows, and
      [slice ~offset:(-10) ~length:10 q] its last ten. Its schema is [q]'s. With
      [offset >= 0] it stops reading at the last row it keeps; with a negative
      [offset] it holds [q]'s last [-offset] rows.

      Its problem is a negative [length]. *)

  val aggregate : by:string list -> Expr.agg Expr.out list -> t -> t
  (** [aggregate ~by os q] is one row per group of [q]'s rows that have the same
      keys in the columns [by], by key identity (null is one key, and NaN is
      one), in order of first appearance. Each row holds the columns [by], then
      the outputs [os] reduced over the group's rows in [q]'s order. [~by:[]]
      makes one group of every row, so one row, even when [q] has none. It
      blocks, in O(n) expected time, holding the columns [by] and those that
      [os] read.

      Its problems are a key that names no column or is named twice, those of
      binding [os], two outputs of one name, and an output of a key's name. *)

  val join :
    ?kind:Join.kind ->
    ?each_left:Join.count ->
    ?each_right:Join.count ->
    on:Join.cond ->
    t ->
    t ->
    t
  (** [join ?kind ?each_left ?each_right ~on right left] pairs [left]'s rows
      with the rows of [right] that [on] matches, written
      [left |> join ~on right], with:
      - [kind], the rows kept ({!Join.kind}). Defaults to {!Join.Inner}. A right
        join is a left join with the arguments swapped.
      - [each_left] and [each_right], the number of matches each row of [left]
        and of [right] must have; another number fails the run. Both default to
        {!Join.Any}.

      The rows come in [left]'s order, each left row followed by its matches in
      [right]'s order; a {!Join.Full} join then appends [right]'s unmatched rows
      in order. Over {!Join.position}, {!Join.Inner} and {!Join.Semi} keep
      min(n, m) rows, {!Join.Left} keeps n, {!Join.Full} max(n, m), and
      {!Join.Anti} [left]'s rows past m. Its columns are those {!Join}
      describes. An equality or {!Join.position} join blocks on both inputs; a
      join on {!Join.all} blocks on [right] and streams [left]. It runs in:
      - O(n + m + matches) for an equality join;
      - O((n + m) log m + matches log matches) for an inequality join;
      - O((n + m) log m) for {!Join.closest} and {!Join.nearest}.

      Its problems are, for each atom of [on] in order: a column missing on its
      side; columns that do not meet, or, outside equality, do not order; a
      [within] or a {!Join.nearest} over columns without a difference; and a
      [within] that is not a literal, is negative, is not of the difference's
      kind, or that the difference's type does not hold. Then, in a {!Join.Full}
      join, a left column that is the left of two equality atoms, and, except in
      a {!Join.Semi} or {!Join.Anti} join, the joined columns that have one
      name. *)

  val append : t -> t -> t
  (** [append rest q] is [q]'s rows, then [rest]'s, written [q |> append rest].
      The two have the same columns, matched by name, of equal types, and the
      schema is [q]'s. Appending other columns is [Kit.union], which derives the
      missing ones as nulls. It costs O(1) over tables and streams over sources.

      Its problems are a column that only [q] has, one that only [rest] has, and
      a column whose types differ. *)

  val unnest : string list -> t -> t
  (** [unnest cs q] is one row per element of the list columns [cs], in order,
      each column of [cs] replaced in place by its elements, of the list's
      element type, and [q]'s other columns repeated. Several columns zip: their
      lists have one length in each row, a null list counting as empty, and
      unequal lengths fail the run. A row whose lists are null or empty gives no
      row. It streams in O(n + elements).

      Its problems are an empty [cs], a name that names no column or is named
      twice, and a column that is not a list. *)

  (** {1:running Running} *)

  val fold : t -> init:'a -> ('a -> table -> 'a) -> ('a, Error.t) result
  (** [fold q ~init f] runs [q]: it reads [q]'s sources, computes [q]'s rows in
      batches and folds [f] over them, in order, from [init]. Each batch has at
      least one row. The batches [f] sees depend on the batches of [q]'s inputs;
      their rows in order do not. It optimizes [q] first ({!optimize}).

      [Error e] where the run fails: a source's error, or one found in the data,
      such as a cast that loses a value, [only] with two values or a join
      assertion. A run fails as evaluating its optimized plan one row at a time
      would, at the first row, in that order, where a step fails, whether an
      error or an exception. Which failure is reported depends on the plan and
      its inputs' values in order, never on batches or cores. [e] names the
      step, as {!pp} prints it, and the row of the step's input.

      A function of [q]'s expressions may be called on rows past a limit. Every
      reader [fold] opens is closed when it returns, also when [f], a function
      of [q]'s expressions or a source raises; the exception then propagates.

      Running under a transformation of nx, such as compilation or
      vectorization, is not specified yet: talon runs on the caller's fiber, and
      a step that reads values on the host, such as a filter, a sort, a group, a
      join or text, raises whatever nx raises when those values are traced. *)

  val run : t -> (table, Error.t) result
  (** [run q] is [q]'s rows as one batch: [fold] into a table, then one copy
      into a single batch. Its columns are canonical: offsets from [0], values
      that are exactly the rows', validities at bit offset [0], so their
      {!Column.layout}s do not depend on batches either. *)

  val values : ('a, Expr.row) Expr.t -> t -> ('a array, Error.t) result
  (** [values e q] is [e] on each row of [q], decoded to OCaml:
      [values Expr.(const mk $ carrier $ option origin) q]. An extension value
      is decoded with its declaration's [dec]. [q] reads only the columns [e]
      reads.

      [Error e] as for {!run}, and where [e] is null on a row, or a value is
      outside what OCaml reads ({!Column.values}): read nullable values through
      {!Expr.option}. [e] then names the step [values] followed by [e].

      Raises [Invalid_argument] with a report like a verb's, named [values], if
      [e] does not bind against [q]'s schema. *)

  (** {1:optimizing Optimizing} *)

  val optimize : t -> t
  (** [optimize q] is the plan that running [q] runs: [q] with the same schema
      and the same rows, rewritten to read and compute less. Running a query
      optimizes it first, so [optimize] serves to print the plan that runs and
      to compare plans. It reads no data, it calls the sources' [pushdown], and
      [optimize (optimize q)] is [optimize q].

      {b Results.} The rewrites change no byte of a result, including those
      under its nulls, and add no failure: an operation that can fail meets only
      rows that [q] shows it. They may remove failures: a value that no row and
      no column of the result reads is not computed, so a failure, or an
      exception from a user function, that only such a value meets does not
      happen. Of the failures that remain, a run reports the one at the earliest
      row of the optimized plan.

      {b Constants.} Arithmetic, comparisons and [not] of literals are
      evaluated as a run evaluates them, and become the literal they compute.
      One whose evaluation fails, or gives a value OCaml's type does not hold
      (an [int64] past [max_int]), stays and evaluates at run time. [&&], [||],
      [is_null], [if_], [coalesce], and [store] of a literal fold alike.
      [a && false], [a || true], [a && true], [a || false], an [if_] on a
      literal and a [coalesce] with literals simplify alike. A filter whose
      predicate is [true] goes, and one whose predicate is [false] or null
      becomes [slice ~offset:0 ~length:0].

      {b Predicates.} A filter's predicate splits into its {e conjuncts}, the
      operands of its [&&]s. A conjunct moves toward the sources when it is
      {e row-local}, its value at a row depending on that row alone: it has no
      {!Expr.over}, {!Expr.rolling}, {!Expr.shift} or {!Expr.rank}. It moves
      past each step that keeps the rows it sees and the values it reads:
      - a [sort];
      - a [filter] whose predicate is row-local, with which it merges;
      - a [select] or a [derive] whose outputs are row-local, when each column
        the conjunct reads is a column of the step's input kept unchanged, under
        its name or another ([select ["y" := x]]);
      - an [aggregate] with keys, when the conjunct reads only keys and none of
        their types holds floats, since [-0.] and [0.] are one key with two
        values;
      - an [unnest], when the conjunct reads no unnested column;
      - an [append], into both inputs;
      - a [join] whose [~each_left] and [~each_right] are [Any], on a condition
        other than {!Join.position}: into the left input when the conjunct reads
        only left columns and the join is [Inner], [Left], [Semi] or [Anti], and
        into the right input when it reads only right columns, the join is
        [Inner] and the condition has no {!Join.closest} or {!Join.nearest}
        atom.

      It stops at a [slice], at a [Full] join, at a join with an assertion, at a
      step with an output or predicate that is not row-local, and at a table.
      Pushdown never crosses a join with an assertion: a conjunct entering one
      input changes the matches of the other input's rows, and so could make
      that side's assertion fail on rows the plan shows it. A conjunct that can
      fail, one with a [cast] that narrows, [Str.parse], temporal arithmetic or
      parsing, [of_option], [$] or [batch], passes only the steps that show it
      the rows they receive: a [sort], a [select] or a [derive], an
      [aggregate]'s keys and an [append]. Below a [filter], a [join] or an
      [unnest] it would meet rows that [q] never shows it.

      When it reaches a source, a conjunct that compares a column with a
      literal, tests it with {!Expr.is_in} or {!Expr.is_null}, or combines such
      tests with [&&], [||] and [not], is a {!Source.Pred.t}, and it is offered
      to the source's [pushdown]: an [Exact] conjunct leaves the plan and goes
      into the source's request, an [Inexact] one goes into the request and
      stays, and an [Unsupported] one stays. The conjuncts that stay at one
      place form filters in plan order, a conjunct that can fail in a filter
      above those it must not pass.

      {b Slices.} A slice moves below a [select] or a [derive] whose outputs are
      row-local, and above the conjuncts that move. Two slices from the start,
      [offset >= 0], merge into one. A slice from the start of an [append]
      limits both its inputs to [offset + length] rows. A slice from the start
      directly above a source gives it a limit of [offset + length] rows. A
      conjunct that stays is a filter between them, so a source with an
      [Inexact] or [Unsupported] conjunct gets no limit. A slice from the end
      gets one too when the source states its rows and is handed no conjunct,
      the slice then counting from the start. A slice from the start directly
      above a [sort] runs as a selection of its first [offset + length] rows,
      which sorts only the rows whose first key is at most the
      [offset + length]th row's: [sort] then [slice] is a top-k.

      {b Projections.} Each source reads only the columns that a step reads or
      that the result has, and each [select], [derive] and [aggregate] drops the
      outputs that nothing reads. A [select] that keeps its input's columns
      unchanged, in order, goes, and so does a [derive] left without outputs,
      and a [select] that only keeps columns below a [select] or an [aggregate],
      which read their input by name. Where an input of an [append] has columns
      that the [append] does not take, [select [keep (names …)]] keeps those it
      takes.

      {b Sharing.} Subplans that are the same steps with equal arguments over
      physically the same tables and sources become one value, which a run runs
      once; equal subexpressions of one step are one expression, which it
      computes once. Each place that reads a source is a read of its own, with
      the columns and the conjuncts of that place, so places with equal requests
      are one read. A step that a plan reaches more than once runs once: a run
      computes it as far as its furthest reader reads, and holds its rows, of
      the columns its readers read, from its slowest reader to its furthest.

      The guide's pipeline, optimized, reads two of the CSV file's columns,
      which answers [Unsupported] to every conjunct:
      {v
      query → carrier string, mean_delay float64, flights int64, name string
      sort [desc "mean_delay"]
      └ join ~on:(keys ["carrier"]) ~each_left:One
        ├ aggregate ~by:["carrier"] ["mean_delay" := mean dep_delay;
        │                            "flights" := rows]
        │ └ filter (dep_delay > 15.)
        │   └ csv "flights.csv" (19 columns) ~columns:["dep_delay"; "carrier"]
        └ parquet "carriers.parquet" (2 columns)
      v} *)

  (** {1:comparing Comparing and formatting} *)

  val equal : t -> t -> bool
  (** [equal q0 q1] is [true] iff [q0] and [q1] are the same plan: the same
      steps with equal arguments, expressions compared by their identity, tables
      by key identity row by row, and sources physically, with equal requests
      (see {!pp}). Equal queries have equal schemas. *)

  val pp : Format.formatter -> t -> unit
  (** [pp ppf q] formats [q]'s plan, which reads no data: the line [query →] and
      [q]'s schema as {!Schema.pp} formats it, then [q]'s last step, with the
      steps it reads below it in a tree:
      {v
      query → carrier string, mean_delay float64, flights int64, name string
      sort [desc "mean_delay"]
      └ join ~on:(keys ["carrier"]) ~each_left:One
        ├ aggregate ~by:["carrier"] ["mean_delay" := mean dep_delay;
        │                            "flights" := rows]
        │ └ filter (dep_delay > 15.)
        │   └ csv "flights.csv" (19 columns)
        └ parquet "carriers.parquet" (2 columns)
      v}
      A step formats as the call of the verb that made it, without its input or
      module paths, its expressions as {!Expr.pp} formats them, its keys as they
      are written inside [Order.( … )] and its condition as it is written inside
      [Join.( … )]:
      - outputs as ["name" := e], with selectors resolved, a run of columns kept
        unchanged under their names as one [keep (names ["a"; "b"])], and a run
        of every field of a record [r], in order, as [unpack r];
      - [~kind], [~each_left] and [~each_right] after [~on], when they are not
        their defaults;
      - a table as [table (4 columns, 16 rows)], and a source as its name and
        its number of columns, then its number of rows when it states one:
        [parquet "carriers.parquet" (2 columns, 1491 rows)];
      - a source that is asked for less than all of it, as an optimized plan
        asks ({!optimize}), then as its request: [~columns] when it reads fewer
        than all its columns, [~filters] when it is handed conjuncts, and
        [~limit] when it has one:
        [parquet "f.parquet" (19 columns) ~columns:["dep_delay"; "carrier"]
         ~filters:[dep_delay > 15.] ~limit:10].

      A join's left input comes before its right, and [append]'s [q] before
      [rest]. A step that the plan reaches more than once, as equal subplans are
      after {!optimize}, is labelled [#1], [#2], … in the order the printer
      first meets them: the first time as [#1] followed by the step and its
      inputs, and each later time as [#1] alone:
      {v
      query → k string, n int64, m int64
      join ~on:(keys ["k"])
      ├ #1 aggregate ~by:["k"] ["n" := rows]
      │ └ csv "x.csv" (3 columns) ~columns:["k"]
      └ select [keep (names ["k"]); "m" := n]
        └ #1
      v}
      A step too long for the margin continues on the next lines, indented under
      it. Lines fit the formatter's margin counted from column 0, since a
      formatter does not tell its current indentation: inside an indented box
      they overrun the margin by that indentation. *)
end

module Kit : sig
  (** Compositions of the verbs and expressions.

      Each value of [Kit] is a composition written against talon's public
      signature alone, and its documentation shows the definition. A plan built
      with [Kit] prints as the verbs that make it, and its problems are those of
      its verbs, which report them as they report any other plan's. A function
      raises [Invalid_argument] itself only for an argument that is wrong
      whatever the schema, or for a requirement that no verb states. *)

  (** {1:queries Queries} *)

  val head : int -> Query.t -> Query.t
  (** [head n q] is the first [n] rows of [q]:
      {[
      Query.slice ~offset:0 ~length:n q
      ]} *)

  val tail : int -> Query.t -> Query.t
  (** [tail n q] is the last [n] rows of [q]:
      {[
      Query.slice ~offset:(-n) ~length:n q
      ]} *)

  val top_k : int -> Order.t list -> Query.t -> Query.t
  (** [top_k k keys q] is the first [k] rows of [q] in the order of [keys]:
      {[
      Query.(q |> sort keys |> slice ~offset:0 ~length:k)
      ]}
      A slice from the start of a sort runs as a selection of its first rows,
      which sorts only the rows that precede or tie with the [k]th on the first
      key. *)

  val distinct : Query.t -> Query.t
  (** [distinct q] is the first of each set of [q]'s rows that are the same on
      every column, by key identity (null is one key, and NaN is one), in order:
      {[
      match List.map fst (Schema.columns (Query.schema q)) with
      | [] -> head 1 q
      | names -> Query.aggregate ~by:names [] q
      ]}
      A query without columns has one distinct row if it has rows. The first of
      each set of rows that are the same on the columns [ks] alone is
      [Query.filter Expr.(over ~by:ks Kit.index = int 0) q]. *)

  val count_by : string list -> Query.t -> Query.t
  (** [count_by ks q] is one row per group of [q]'s rows that have the same keys
      in the columns [ks], in order of first appearance: the columns [ks], then
      ["count"], the group's number of rows as [int64]:
      {[
      Query.aggregate ~by:ks Expr.[ "count" := rows ] q
      ]} *)

  val value_counts : string -> Query.t -> Query.t
  (** [value_counts c q] is each distinct value of [q]'s column [c], null
      included, with its number of rows, most frequent first and ties in order
      of first appearance:
      {[
      count_by [ c ] q |> Query.sort [ Order.desc "count" ]
      ]} *)

  val describe : Query.t -> Query.t
  (** [describe q] summarizes each integer or float column of [q], in order, as
      one row of the columns ["column"], its name; ["count"], its number of
      values that are not null; ["nulls"], its number of nulls; ["mean"];
      ["std"]; ["min"]; ["q25"], ["median"] and ["q75"], its quartiles; and
      ["max"]. ["count"] and ["nulls"] are [int64] and the others [float64]. The
      statistics are {!Expr}'s reductions, which skip nulls:
      {[
      let stats name x =
        Query.aggregate ~by:[]
          Expr.
            [
              "column" := string name;
              "count" := count x;
              "nulls" := rows - count x;
              "mean" := mean x;
              "std" := std x;
              "min" := cast Type.float64 (min x);
              "q25" := quantile 0.25 x;
              "median" := median x;
              "q75" := quantile 0.75 x;
              "max" := cast Type.float64 (max x);
            ]
          q
      in
      s0 |> Query.append s1 |> … |> Query.append sk
      ]}
      where [s0] to [sk] are [stats n (Col.int n)] or [stats n (Col.float n)]
      for each integer or float column [n]. Without such a column, [describe q]
      is [head 0 (stats "" Expr.(store Type.float64 null))], which has no rows.
      It reads [q] once per integer or float column. *)

  val null_count : Query.t -> Query.t
  (** [null_count q] is one row with, for each column of [q], in order and under
      its name, its number of nulls as [int64]:
      {[
      Query.aggregate ~by:[]
        Expr.[ each Sel.all { column = (fun n x -> n := rows - count x) } ]
        q
      ]} *)

  val drop : Sel.t -> Query.t -> Query.t
  (** [drop sel q] is [q] without the columns that [sel] selects:
      {[
      Query.select Expr.[ keep Sel.(all - sel) ] q
      ]} *)

  val rename : (string * string) list -> Query.t -> Query.t
  (** [rename pairs q] is [q] with each column [old] of [pairs] renamed to the
      [name] it pairs with, in place:
      {[
      let name n = Option.value ~default:n (List.assoc_opt n pairs) in
      Query.select
        Expr.
          [
            each
              Sel.(all + names (List.map fst pairs))
              { column = (fun n x -> name n := x) };
          ]
        q
      ]}
      An [old] that [q] lacks is a problem of the [select], through
      {!Sel.names}, and so is a name that two columns would take.

      Raises [Invalid_argument] if [pairs] renames a column twice. *)

  val complete : string list -> Query.t -> Query.t
  (** [complete ks q] is [q] with a row for each combination of the values of
      the columns [ks] that [q] lacks, its other columns null. Each column's
      values come in order of first appearance, combinations vary the last
      column fastest, and each combination is followed by its rows of [q], in
      order:
      {[
      let values k = Query.select Expr.[ keep Sel.(names [ k ]) ] q |> distinct in
      let combos =
        List.fold_left
          (fun acc k -> acc |> Query.join ~on:Join.all (values k))
          (values k0) ks'                               (* ks = k0 :: ks' *)
      in
      combos
      |> Query.join ~kind:Left ~on:(Join.keys ks) q
      |> Query.select Expr.[ keep (Sel.names (columns of q)) ]
      ]}
      It has [q]'s schema. Its key columns hold the combinations' values, so
      where a key holds floats, a row of [q] holds the first of its equal values
      to appear in [q]: [-0.] or [0.].

      Raises [Invalid_argument] if [ks] is empty or names a column twice. *)

  val one_hot : string -> Query.t -> Query.t
  (** [one_hot c q] is [q] with its categorical column [c] replaced, in place,
      by one [bool] column per category, in dictionary order, named
      [c ^ "_" ^ category]: [true] where [c] is that category, [false] where it
      is another, and null where [c] is null:
      {[
      Query.select
        Expr.(
          [ keep (Sel.names before) ]
          @ List.map
              (fun cat -> c ^ "_" ^ cat := Col.string c = string cat)
              categories
          @ [ keep (Sel.names after) ])
        q
      ]}
      The categories are in [c]'s type, so the columns are known before any data
      is read; a string column is cast to a categorical type first.

      If [q] lacks [c], it is [Query.select Expr.[ keep Sel.(names [ c ]) ] q]'s
      problem, reported with the nearest names.

      Raises [Invalid_argument] if [c] is not categorical. *)

  val union : Query.t -> Query.t -> Query.t
  (** [union rest q] is [q]'s rows, then [rest]'s, written
      [q |> Kit.union rest], with [q]'s columns, then those that only [rest]
      has, in order; a column is null on the side that lacks it:
      {[
      let nulls from into =
        List.filter_map
          (fun (n, Type.Any t) ->
            match Schema.find into n with
            | Some _ -> None
            | None -> Some Expr.(n := store t null))
          (Schema.columns from)
      in
      let pad os q = if os = [] then q else Query.derive os q in
      let s = Query.schema q and r = Query.schema rest in
      pad (nulls r s) q |> Query.append (pad (nulls s r) rest)
      ]}
      A column that both have, of two types, is a problem of the [append]. *)

  (** {1:runs Runs} *)

  val categorize : string list -> Query.t -> (Query.t, Error.t) result
  (** [categorize cs q] is [q] with each text column of [cs] cast, in place, to
      the categorical type whose dictionary is the column's distinct non-null
      values in byte order, the order of {!Order.asc} on text, which does not
      depend on the order or the batches of [q]'s rows. It runs one query per
      column, after building them all:
      {[
      let words c =
        let x = Col.string c in
        Query.select Expr.[ c := cast Type.string x ] q
        |> Query.filter Expr.(not (is_null x))
        |> distinct
        |> Query.sort [ Order.asc c ]
        |> Query.values x
      in
      Query.derive
        (List.map
           (fun (c, ws) -> Expr.(c := cast (Type.categorical ws) (Col.string c)))
           dictionaries)                     (* the [(c, words c)] of each [c] *)
        q
      ]}
      [cs] empty is [Ok q]. A column of [q] that is missing or is not text is a
      problem of the [select]. A categorical column takes the dictionary of the
      values it holds, in byte order, since it is read as text.

      [Error e] if a run fails, or if a column holds more than 2{^ 31} - 1
      distinct strings, the most a dictionary holds.

      Raises [Invalid_argument] if [cs] names a column twice. *)

  (** {1:expressions Expressions} *)

  val cumulative : ('a, Expr.agg) Expr.t -> ('a, Expr.row) Expr.t
  (** [cumulative r] is, at each row of the frame, the reduction [r] over the
      frame's rows from the first to that one: [cumulative (sum x)] is a running
      sum. It takes [r]'s type and null rules: a null row adds nothing to a
      reduction that skips nulls, and a row before the frame's first value has
      [r] over no values, [0] for a sum and null for a maximum. Inside
      [Expr.over ~by ~order] the frame is each partition, in its order.
      {[
      Expr.rolling (Window.rows ~before:max_int ~after:0) r
      ]}
      A growing [rows], [count], [sum], [min], [max], [first], [last] or [ewm]
      costs one pass over the frame; over the input's rows it blocks, as
      [Expr.rolling] does. *)

  val index : (int, Expr.row) Expr.t
  (** [index] is each row's position in its frame, from [0], as [int64]: the
      position that [Expr.arg_min] and [Expr.arg_max] give.
      {[
      Expr.(cumulative rows - int 1)
      ]} *)

  val arg :
    (int, Expr.agg) Expr.t -> ('a, Expr.row) Expr.t -> ('a, Expr.agg) Expr.t
  (** [arg p v] is [v] at the position [p] of the frame, and null where [p] or
      that value is null: [arg (arg_max score) name] is the name with the best
      score.
      {[
      Expr.(first (if_ (index = over p) v null))
      ]} *)

  val fill_forward : ('a, Expr.row) Expr.t -> ('a, Expr.row) Expr.t
  (** [fill_forward x] is [x] with each null replaced by the last value before
      it in the frame, and null before the first value. It applies to every
      type, an extension read with [Expr.each] included.
      {[
      cumulative (Expr.last x)
      ]} *)
end
