(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tables: named, typed columns.

    {!Type}s say what columns store and {!Kind}s what their cells read as in
    OCaml. {!Binary}, {!Decimal}, {!Time} and {!Record} are the OCaml values
    that cells read as, and {!Schema}s name and type a table's columns. *)

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
      backslashes preceded by a backslash and control bytes written as [\x] and
      two hexadecimal digits. *)
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
