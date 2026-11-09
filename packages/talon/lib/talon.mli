(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tables: named, typed columns.

    {!Type}s say what columns store and {!Kind}s what their cells read as in
    OCaml. {!Binary}, {!Time} and {!Record} are the OCaml values that cells read
    as, {!Schema}s name and type a table's columns, and {!Tz} reads the time
    zone database.

    A {!Query} is the centre: a plan over a table or a {!Source}, transformed by
    verbs, whose schema is known before any data is read. Its verbs take
    {!Expr}essions, read through {!Col} handles and {!Ext} declarations; {!Sel}
    chooses columns, {!Order} sorts, and {!Join} conditions pair rows. Plan
    problems raise [Invalid_argument]; failures in data are {!Error} values. *)

type t
(** The type for tables: named, typed columns of equal length. *)

type table := t

module Binary = Binary
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

  val date : Time.date t
  (** [date] reads [date] columns. *)

  val instant : Time.instant t
  (** [instant] reads [datetime] columns of every unit whose clock is UTC
      ({!Time.section-stamps}). A value outside the range of {!Time.instant}
      fails when it is read. *)

  val wall : Time.wall t
  (** [wall] reads [datetime] columns of every unit whose clock is a wall clock
      ({!Time.section-stamps}). A value outside the range of {!Time.wall} fails
      when it is read. *)

  val span : Time.span t
  (** [span] reads [duration] and [clock] columns of every unit. A time of day
      reads as its span from midnight. A value outside the range of {!Time.span}
      fails when it is read. *)

  val list : 'a t -> 'a array t
  (** [list k] reads list columns whose elements [k] reads. *)

  val tensor : ('a, 'b) Nx.dtype -> ('a, 'b, Nx.host) Nx.t t
  (** [tensor dt] reads tensor columns of dtype [dt] and any cell shape. *)

  (** {1:comparing Comparing and formatting} *)

  val provably_equal : 'a t -> 'b t -> ('a, 'b) Stdlib.Type.eq option
  (** [provably_equal k0 k1] is [Some Equal] iff [k0] and [k1] are the same
      kind, neither being nor containing the extension kind: lists are equal
      when their elements are, and tensors when their dtypes are. It is [None]
      otherwise, in particular for two extension kinds. *)

  val pp : Format.formatter -> 'a t -> unit
  (** [pp ppf k] formats [k] by the name of the value that builds it: [bool],
      [int], [float], [string], [binary], [date], [instant], [wall] and [span].
      The record kind formats as [record] and the extension kind as [ext]. A
      list kind formats its element kind in brackets, as in [list[float]], and a
      tensor kind its dtype, as in [tensor[float32]]. *)
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

      Under [open Talon], this module shadows [Stdlib.Type]. *)

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
    | Bool : bool t  (** Booleans, stored as {!Nx.bit}, eight to a byte. *)
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
    | String : string t
        (** UTF-8 text. Text is validated when it enters talon, and invalid
            bytes belong in {!constructor-Binary}. *)
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
    | Datetime : { unit_ : unit_; clock : 'c Time.clock } -> 'c Time.stamp t
        (** Stamps ({!Time.section-stamps}), stored as int64 ticks of [unit_]
            since 1970-01-01 00:00:00 of [clock]: instants when [clock] is
            [Time.Utc] and wall clocks when it is [Time.Local]. A tick outside
            the range of {!Time.stamp} fails when it is read. *)
    | List : 'a t -> 'a array t
        (** Lists of values of the element type, read as arrays. *)
    | Record : (string * any) list -> Record.t t
        (** Records with the given fields in order, Arrow's struct. Field names
            are distinct UTF-8 strings. *)
    | Tensor : ('a, 'b) Nx.dtype * int iarray -> ('a, 'b, Nx.host) Nx.t t
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

  val string : string t
  (** [string] is {!String}. *)

  val binary : Binary.t t
  (** [binary] is {!constructor-Binary}. *)

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

  val instant : unit_ -> Time.instant t
  (** [instant u] is {!Datetime} of instants in the unit [u]. *)

  val wall : unit_ -> Time.wall t
  (** [wall u] is {!Datetime} of wall clocks in the unit [u]. *)

  val list : 'a t -> 'a array t
  (** [list t] is {!List} of [t]. *)

  val record : (string * any) list -> Record.t t
  (** [record fields] is {!constructor-Record} with [fields], in order. A record
      type may have no fields, and a field's name may be empty.

      Raises [Invalid_argument] if two fields have the same name, or a name is
      not valid UTF-8. *)

  val tensor : ('a, 'b) Nx.dtype -> int array -> ('a, 'b, Nx.host) Nx.t t
  (** [tensor dt shape] is {!Tensor} of [dt] with a copy of [shape].

      Raises [Invalid_argument] if [shape] is empty or holds a negative
      dimension. *)

  val ext : name:string -> ?metadata:string -> 'a t -> ext t
  (** [ext ~name ?metadata storage] is {!constructor-Ext} named [name], with
      [metadata] (defaults to [""]) and stored as [storage]. Format readers
      build the types of extension columns with it.

      Raises [Invalid_argument] if [name] is empty or not valid UTF-8, or if
      [storage] is an extension type. *)

  (** {1:kinds Kinds} *)

  val kind : 'a t -> 'a Kind.t
  (** [kind t] is the kind that [t]'s cells read as: {!Kind.bool}, {!Kind.int}
      for the integer types, {!Kind.float} for the float types, {!Kind.string}
      for [String] and [Categorical], {!Kind.binary}, {!Kind.date}, {!Kind.span}
      for [Clock] and [Duration], {!Kind.instant} and {!Kind.wall} for
      [Datetime] by its clock,
      [Kind.list (kind e)] for [List e], {!Record.kind} for every [Record], and
      [Kind.tensor dt] for [Tensor (dt, _)]. For [Ext] it is the extension kind,
      which reads nothing (see {!Kind.provably_equal}). *)

  (** {1:values Values} *)

  val holds : 'a t -> 'a -> bool
  (** [holds t v] is [true] iff [t] holds [v], that is iff [v] can be stored as
      [t]:
      - an integer is in the type's range;
      - a float is NaN, infinite, or rounds to a finite value at the type's
        precision;
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
      - integers, dates, spans and instants order by value;
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
      - a [Categorical] in [String], and in a categorical whose dictionary
        begins with its own;
      - a [Clock] in a clock of a finer unit.

      So [int8] and [uint8] do not meet, while [int8], [uint8] and [int16] meet
      at [int16]. [uint64] does not meet [int64], although every [uint64] that
      OCaml reads is an [int64]. [Clock] and [Duration] never meet, and neither
      do datetimes or durations that differ in unit, since a coarser unit stores
      ticks a finer one cannot: cast first. A wall clock and an instant never
      meet. *)

  (** {1:predicates Equality and formatting} *)

  val equal : 'a t -> 'b t -> bool
  (** [equal t0 t1] is [true] iff [t0] and [t1] are the same type: the same
      constructor with equal arguments. Dictionaries are equal when they hold
      the same strings in the same order, and record types when they have the
      same field names with equal types in the same order. *)

  val pp : Format.formatter -> 'a t -> unit
  (** [pp ppf t] formats [t] as schemas and plans show it:
      - [bool], [int8] to [uint64], [float16] to [float64], [string], [binary]
        and [date];
      - [categorical["AA", "B6"]], with at most the first eight strings of the
        dictionary, followed by an ellipsis and the dictionary's size when it
        holds more: [categorical["a", "b", "c", "d", "e", "f", "g", "h", … 26]];
      - [clock[ns]], [duration[ms]], [datetime[us]] for a wall clock and
        [datetime[us, UTC]] for an instant, with the units [s], [ms], [us] and
        [ns];
      - [list[float64]], [record[carrier string, delay float64]] and
        [tensor[float32, 3×4]];
      - [ext[ymir.epoch, float64]], and [ext[units.mass "kg", float64]] when the
        metadata is not empty.

      A field name or an extension name formats as is when it is
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

      A column is an Arrow array over nx buffers. Its {e validity} is an
      {!Nx.bit_t}, true at each row that holds a value; its absence means no row
      is null. Its values are laid out by its type:
      - one element per row of a primitive nx array: {!Nx.bit} for [bool], the
        integer and float types, [int32] positions in the dictionary for
        categoricals, [int32] days for dates and [int64] ticks for clocks,
        durations and datetimes. A tensor column is one [(rows, …shape)] array
        of its dtype;
      - offsets into a child for byte strings, text and lists: text is a list of
        bytes;
      - one child per field for records.

      An extension column is laid out as its storage. The values under a null
      are unspecified and deterministic: a reader that needs defined values
      masks them with the validity, as Arrow requires. Under a null record, its
      fields' slots are unspecified, their values and validity alike. Columns are immutable,
      and share their buffers with the tensors and layouts that read them: a
      column holds values of its own over that memory, so donating a tensor it
      took or gave ends that tensor alone, and the donation computes into fresh
      memory ({!Nx.donate}).

      A column is {e canonical} when its buffers hold exactly its rows: offsets
      from [0], values exactly the rows', a validity only when a row is null,
      and every buffer of elements narrower than a byte ({!Nx.bit}, the int4
      dtypes) from the first bit of its storage with every bit past its last
      element clear. Two canonical columns whose rows hold the same values,
      those under their nulls included, have the same layout, byte for byte.

      A column's buffers are bytes on the host, which formats, text kernels and
      C stubs read in place: every nx value a column takes or gives, tensors,
      validities, offsets and positions, is a host value ({!Nx.host}). A caller
      places a value of another device set on the host first, with
      {!Nx.place}[ Nx.Host.on]. *)

  type t
  (** The type for columns. *)

  val type_ : t -> Type.any
  (** [type_ c] is the type of [c]'s values. *)

  val length : t -> int
  (** [length c] is the number of rows of [c]. *)

  val null_count : t -> int
  (** [null_count c] is the number of null rows of [c]. The first call may read
      [c]'s validity once; later calls cost O(1). It raises [Invalid_argument]
      for a validity [Rune.jit] or [Rune.vmap] traces
      ({!Query.section-compiled}), whose bits have no elements to count. *)

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
      integer outside OCaml's [int], a stamp or a span outside {!Time}'s range,
      a list with a null element. *)

  val options : 'a Kind.t -> t -> 'a option array
  (** [options k c] is like {!values}, with [None] for each null. *)

  (** {1:tensors Tensors and ragged arrays} *)

  val of_tensor : ?validity:Nx.host Nx.bit_t -> ('a, 'b, Nx.host) Nx.t -> t
  (** [of_tensor ?validity x] is the column of [x]'s rows:
      - for a 1-D [x], of the type of [x]'s dtype: [bool] for [bit] and [bool],
        [int8] to [uint64], [float16] to [float64];
      - for [x] of shape [(n, …shape)], a tensor column of [x]'s dtype and cell
        shape [shape].

      It shares [x] when [x]'s dtype is the storage of its type; a 1-D [bool]
      [x] is packed into {!Nx.bit} once.

      [validity] marks the rows that hold a value; it defaults to every row.

      Raises [Invalid_argument] if [x] is a scalar, if [x] is 1-D of a dtype
      that no scalar type stores ([bfloat16], the float8 and int4 dtypes,
      complex), or if [validity]'s shape is not [x]'s rows. *)

  val to_tensor : ('a, 'b) Nx.dtype -> t -> ('a, 'b, Nx.host) Nx.t
  (** [to_tensor dt c] is [c]'s values as stored, in O(1) after one read of the
      null count the first time: numbers, booleans as {!Nx.bit}, the days or
      ticks of temporal values, the codes of a categorical (its dictionary is in
      its type), and [(rows, …shape)] for a tensor column. It shares [c]'s
      buffer, which must not be written.

      Raises [Invalid_argument] if [dt] is not [c]'s storage dtype ({!Nx.cast}
      converts the result), if [c] is not stored one element per row, or if [c]
      has a null; under [Rune.jit], for a traced validity
      ({!Query.section-compiled}), the null raises when the call returns. *)

  val validity : t -> Nx.host Nx.bit_t option
  (** [validity c] is true at row [i] iff row [i] of [c] holds a value. [None]
      means no row is null. [Some v] may be true everywhere. *)

  val ragged : ('a, 'b) Nx.dtype -> t -> ('a, 'b, Nx.host) Nx_ragged.t
  (** [ragged dt c] is [c]'s rows as a ragged array without a copy, one row per
      row of [c]: a list column's offsets and its elements, stored as [dt], or a
      [string] or [binary] column's offsets and bytes, [dt] being {!Nx.uint8}.
      An extension of such a type is read as its storage. It reads the offsets
      once, as {!Nx_ragged.v} checks them, and no element. It shares [c]'s
      buffers, which must not be written.

      Raises [Invalid_argument] if [c] is neither a list whose elements are
      stored as [dt] nor text or bytes with [dt] {!Nx.uint8}, or if a row of [c]
      or an element of a row is null. *)

  val of_ragged :
    ?validity:Nx.host Nx.bit_t -> ('a, 'b, Nx.host) Nx_ragged.t -> t
  (** [of_ragged ?validity r] is the list column whose rows are [r]'s rows, null
      where [validity] is false: a [list] of the element type {!of_tensor} gives
      [r]'s values, such as [list[int32]] for {!Nx.int32} values, and of a
      tensor type for values of more than one axis. It shares [r]'s buffers,
      packing [bool] values as {!of_tensor} does. [validity] defaults to every
      row valid. [ragged dt (of_ragged r)] has [r]'s rows, [dt] being the dtype
      the column stores [r]'s values in.

      Raises [Invalid_argument] if [validity]'s shape is not [r]'s number of
      rows, or if [r]'s dtype has no talon type, as {!of_tensor} does. *)

  (** {1:layout Layouts}

      A layout is a column's Arrow buffers, as formats read and write them. *)

  (** The type for layouts. *)
  type layout =
    | Fixed of {
        validity : Nx.host Nx.bit_t option;
        values : Nx.host Nx.packed;
      }  (** One element per row, or one cell for a tensor column. *)
    | Varsize of {
        validity : Nx.host Nx.bit_t option;
        offsets : Nx.host Nx.int64_t;
        child : t;
      }
        (** Row [r] is the child's rows [offsets.{r}] to [offsets.{r + 1} - 1]:
            the elements of a list, or, for [string] and [binary], the bytes, a
            [uint8] child without nulls. *)
    | Children of {
        validity : Nx.host Nx.bit_t option;
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
      time of day outside the day. A row is checked only where it is not null.
      [reason] is a phrase, as in [invalid UTF-8 at byte 3]. A child holds its
      own values, so only [l]'s own values are checked.

      Raises [Invalid_argument] if [l] does not lay out [ty]: values of another
      dtype or cell shape than [ty]'s storage, a validity of a shape other than
      [[|n|]] for [n] rows, offsets that are not 1-D, start below [0], decrease
      or reach past the child, a child of another type (for text, a [uint8]
      column with a null), or fields of other names, types or lengths than
      [ty]'s; under [Rune.jit], for traced offsets of [binary]
      ({!Query.section-compiled}), offsets out of place raise when the call
      returns, naming [Nx_ragged.v]. *)

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
      - [string]: the bytes, valid UTF-8; [binary]: the bytes;
      - a categorical: one of the dictionary's strings;
      - [date]: [YYYY-MM-DD], a year outside [0000] to [9999] signed and of at
        least four digits, as {!Time.Date.pp} writes it: [-0044-03-15];
      - [datetime[u]] (a wall clock) and [datetime[u, UTC]] (an instant): a date
        of any year, [T] or a space, [hh:mm:ss] and an optional fraction of one
        to nine digits, then, for an instant only, [Z] or [±hh:mm]; a whole
        number of [u] in [u]'s range, the int64 ticks, which bound the year.

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
      - a datetime with the fewest fraction digits that are exact, and [Z] for
        an instant, every tick of its unit included, past {!Time.Date}'s years;
      - an integer in full, past OCaml's [int] included.

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
(** [column t name] is the column [name] of [t], canonical ({!Column}): [t]'s
    own when it already is.

    Raises [Invalid_argument] if [t] has no column [name]. *)

val take : Nx.host Nx.int64_t -> t -> t
(** [take indices t] is the rows of [t] at [indices], in order, as one batch:
    [take (Nx.Rng.permutation ~key (rows t)) t] shuffles every column.

    Raises [Invalid_argument] if [indices] is not 1-D or holds an index outside
    \[[0];[rows t - 1]\]; under [Rune.jit], for traced [indices]
    ({!Query.section-compiled}), the index outside raises when the call
    returns. *)

val to_tensor : ('a, 'b) Nx.dtype -> string list -> t -> ('a, 'b, Nx.host) Nx.t
(** [to_tensor dt names t] is the [(rows t, List.length names)] matrix whose
    column [j] is the column [List.nth names j] of [t], each value converted to
    [dt] as {!Nx.cast} converts it. It is the one copy a columnar layout forces.
    The columns are numeric or boolean, and have no null.

    Raises [Invalid_argument] if a name is not a column of [t], if a column is
    neither numeric nor boolean, or has a null; under [Rune.jit], for a traced
    validity ({!Query.section-compiled}), the null raises when the call
    returns. *)

val to_dense :
  ('a, 'b) Nx.dtype ->
  axes:(string * int) list ->
  string ->
  t ->
  ('a, 'b, Nx.host) Nx.t * Nx.host Nx.bit_t option
(** [to_dense dt ~axes value t] is [t]'s column [value] scattered into a
    tensor, and its validity. [axes] lists each axis as its key column and
    extent, outermost first, and cells lie in row-major order. A key is a
    categorical code, whose dictionary has the extent's size, or an integer in
    \[[0];[extent - 1]\]. A cell no row names holds zero, and so does a null
    value; the validity clears each such cell and is [None] when every cell
    holds a value. The value column is numeric or boolean, converted as
    {!Nx.cast} converts it.

    Raises [Invalid_argument] if [axes] is empty or names a column twice, if an
    extent is negative or a categorical's differs from its dictionary's size,
    if a key column is neither categorical nor an integer or holds a null or a
    key outside its extent, if two rows name one cell, if [value] is neither
    numeric nor boolean, or if the cells exceed OCaml's [int]. *)

val of_dense :
  ?validity:Nx.host Nx.bit_t ->
  axes:string list ->
  string ->
  ('a, 'b, Nx.host) Nx.t ->
  t
(** [of_dense ~validity ~axes value x] is the table of one row per element of
    [x], in row-major order: an [int64] key column per name of [axes], then the
    elements in column [value], null where [validity] is clear. The column
    shares [x]'s memory where {!Nx.flatten} returns a view. {!to_dense} with
    the same axes and extents returns [x], with zeros where [validity] is
    clear.

    Raises [Invalid_argument] if [axes] is empty or has not [x]'s dimensions,
    if [validity]'s shape is not [x]'s, or as {!v} does. *)

val equal : t -> t -> bool
(** [equal t0 t1] is [true] iff [t0] and [t1] have equal schemas and the same
    keys row by row, by key identity ({!Type.compare_value}, null being one more
    key), whatever their batches.

    Raises [Invalid_argument] over traced columns ({!Query.section-compiled}).
*)

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
    such as [2024-03-15T09:30:00.5Z] for an instant; durations and clocks
    as {!Time.Span.pp} formats them, lists as OCaml lists, and records and
    tensors as [<record>] and [<tensor>].

    Raises [Invalid_argument] if a limit is negative or [l.width] is [0], and
    over traced columns or under [Rune.grad] ({!Query.section-compiled}). *)

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
      [Expr.each] and [Kit.drop] take selectors; keys stay [string list].

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
      - {!window}[ r]: at each row, the rows of the enclosing frame around it
        ({!section-windows});
      - {!per_list}[ ls r]: at each row, the elements of its lists [ls], in
        list order ({!section-lists}).

      A frame's order is its input order, and no expression reorders values
      without returning them to their rows.

      {b Nulls.} Elementwise operations are null where an operand is null,
      except where stated. Comparisons are Kleene: a comparison with null is
      null. Reductions skip nulls except {!rows}, {!count} and {!n_unique}. *)

  (** {1:types Expressions and outputs} *)

  type row
  (** The shape of expressions with one value per row of their frame. *)

  type agg
  (** The shape of expressions with one value per frame. *)

  type ('a, +'s) t
  (** The type for expressions of shape ['s] whose values read as ['a]. *)

  type +'s out
  (** The type for outputs of shape ['s]: named expressions, as [select],
      [derive] and [aggregate] take them. *)

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

  val wall : Time.wall -> (Time.wall, 's) t
  (** [wall t] is the wall clock [t]. *)

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
  (** [cut edges a] is the bin of [a] among [edges], as [int64]. The bin is [i]
      where [edges.(i) <= a < edges.(i + 1)] in talon's total order, so [k + 1]
      edges bound [k] bins, each holding its lower edge, and adjacent bins share
      an edge. It is null where [a] is null, below the first edge, or at or
      above the last, which NaN is: [cut [| 0.; 0.5; 1. |] x] is [0] for [0.2],
      [1] for [0.5] and null for [1.].

      Each edge is a literal of [a]'s type, which must hold it and may round it,
      so [cut edges a = int i] exactly where
      [a >= edges.(i) && a < edges.(i + 1)]. As that type stores them, the edges
      must ascend strictly and hold no NaN; [-0.] and [0.] are one edge. [a]'s
      type is one that {!( < )} orders. [edges] is copied.

      Raises [Invalid_argument] if [edges] has fewer than two elements. *)

  val cast : 'b Type.t -> ('a, 's) t -> ('b, 's) t
  (** [cast ty a] converts [a]'s values to [ty]:
      - between [bool], integer and float types: integers take exact values
        only, so a fractional, infinite or NaN float, or a value out of range,
        is a data error; floats round to nearest; [bool] takes [0] and [1] only,
        and gives [0] and [1];
      - between [string] and categorical types: the text is kept, and a
        categorical takes only the strings of its dictionary;
      - between datetimes of one clock, between durations, and between clocks:
        values are kept, and a value that is not a whole number of the new unit
        is a data error;
      - between clocks and durations: a time of day is its span from midnight,
        converted as between durations, and a duration outside \[[0], 1 day) is
        a data error for a clock;
      - between a duration and [int64]: the integer is the duration's ticks in
        its own unit, exactly, for every [int64]. To read another unit, cast
        between durations first, as
        [cast Type.int64 (cast (Type.duration Ms) d)] does for milliseconds;
      - between dates and wall-clock datetimes: a date becomes the datetime of
        its midnight, and a datetime becomes the day it falls on, so
        [1969-12-31T23:59:59] becomes [1969-12-31]. A value out of the new
        type's range is a data error;
      - lists element by element, records with the same field names field by
        field, and tensors of the same shape element by element, as [Nx.cast]
        does.

      Any other pair is a problem that names the function that converts it, if
      one does: {!Str.parse} and {!Temporal.parse} for text to values, and
      {!Temporal.format} for dates, clocks and datetimes to text, and
      {!Temporal.wall} or {!Temporal.localize} between the clocks. An instant
      falls on a day only in a zone, so it casts neither to nor from a date.
      [a] needs a column type. *)

  type fn = { f : 'a 'b. ('a, 'b, Nx.host) Nx.t -> ('a, 'b, Nx.host) Nx.t }
  (** The type for elementwise nx functions of one argument that preserve its
      dtype, such as [Nx.exp], on host values as columns hold. *)

  val nx : fn -> ('a, 's) t -> ('a, 's) t
  (** [nx { f } a] applies [f] to [a]'s values with nx's semantics, IEEE's for
      floats: [nx { f = Nx.exp } x]. [a] has an nx dtype: an integer, float or
      [bool] type, [bool] computing as {!Nx.bool}. When the verb is applied,
      talon calls [f] once on a traced value of that dtype and records the
      operations [f] performs; an [f] that moves, reduces or reshapes its
      argument is a problem, and so is one that ignores it, which would lose its
      nulls. *)

  type fn2 = {
    f2 :
      'a 'b.
      ('a, 'b, Nx.host) Nx.t -> ('a, 'b, Nx.host) Nx.t -> ('a, 'b, Nx.host) Nx.t;
  }
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
      floats and durations. Integer sums wrap as nx's integers do; a duration
      sum that overflows is a data error. Other types are a problem. *)

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
  (** [mean a] is the arithmetic mean of [a]'s values, as [float64]: their
      sum divided by their count. Integers of at most 32 bits sum exactly in
      [int64] in a frame of fewer than 2{^31} values, and the sum rounds once
      to [float64]; other values sum in [float64] in row order. [a] is an
      integer or float expression, as for {!std}, {!var}, {!median} and
      {!quantile}; other types are a problem. *)

  val std : ('a, row) t -> (float, agg) t
  (** [std a] is the sample standard deviation of [a]'s values, dividing by n -
      1, and null for fewer than two values. It is infinite only where the
      standard deviation exceeds [max_float]. Its bits depend on the frame's
      values in row order alone. Over traced values, and under [Rune.jit], it
      is a two-pass form, equal within rounding.

      Raises [Invalid_argument] over a [float64] operand under [Rune.jit], or
      over traced [float64] values under [Rune.vmap], as that form reads them
      to detect overflow. *)

  val var : ('a, row) t -> (float, agg) t
  (** [var a] is the sample variance of [a]'s values, like {!std}. It is
      infinite only where the variance exceeds [max_float]. *)

  val median : ('a, row) t -> (float, agg) t
  (** [median a] is [quantile 0.5 a]. *)

  val quantile : float -> ('a, row) t -> (float, agg) t
  (** [quantile p a] is the [p]-quantile of [a]'s values, interpolating linearly
      between the two nearest ranks.

      Raises [Invalid_argument] if [p] is not in \[[0];[1]\]. *)

  val n_unique : ('a, row) t -> (int, agg) t
  (** [n_unique a] is the number of distinct keys of [a], null being one key, as
      [int64]. *)

  val arg_min : ('a, row) t -> (int, agg) t
  (** [arg_min a] is the zero-based position in the frame of [a]'s first least
      value, as [int64]. An extension type needs an ordered declaration. *)

  val arg_max : ('a, row) t -> (int, agg) t
  (** [arg_max a] is the position of [a]'s first greatest value, like
      {!arg_min}. *)

  val pivot : (string, row) t -> ('a, row) t -> (Record.t, agg) t
  (** [pivot key v] is, per frame, one record with a field per category of
      [key]'s categorical type, in dictionary order, named as the category. The
      field holds [v] at the frame's row whose [key] is that category, and is
      null where the frame has no such row. A null [v] in a cell is a null
      field, so a missing cell and a null value look alike; a [bool] [true]
      pivoted beside [v] tells them apart.

      The record's type, and the plan's schema, follow from [key]'s type alone
      and never from its values: cast [key] to the categorical naming the
      fields, or read them from the data with {!Kit.categorize}. Every non-null
      value of [v] lands in a cell of its row, or the run fails: a row with a
      null [key] and a null [v] holds no cell, so a left join's unmatched row
      leaves its frame's fields null; a null [key] with a non-null [v] has no
      field to hold the value and is a data error, as is any value outside the
      dictionary at the cast. Two rows of one frame with one key are a data
      error; reduce them first with {!Query.aggregate} on the frame's keys and
      [key]. Filtering before [pivot] removes the groups left with no row.

      A data error reports the failing frame's first row, naming the key.

      Cost: O(n + g k) for n rows, g frames and k categories. Under {!over} or
      {!window} a frame's record holds its k cells, so the rows together hold
      their frames' k cells.

      Its problems are a [key] whose type is not categorical, and a [v] with no
      column type, or none reachable. [aggregate] with it blocks, holding the
      columns it reads. Raises [Invalid_argument] under [Rune.jit], and under
      [Rune.vmap] over traced keys ({!Query.section-compiled}): its failures
      read them. *)

  val collect : ('a, row) t -> ('a array, agg) t
  (** [collect a] is [a]'s values in frame order as a list of [a]'s type, with
      one element per row of the frame and a null element for a null value. It
      is never null, and is the empty list over no rows. [a] must have a column
      type, and an extension column is read with {!each}.

      {b Cost.} O(n). Under {!over} each row holds its partition's list, and
      the lists together hold the sum of the partitions' squared sizes. Under
      {!window} every row holds its window's list, and memory holds them all
      together.

      Raises [Invalid_argument] under [Rune.jit]
      ({!Query.section-compiled}). *)

  (** {1:frames Frames}

      {!over} evaluates an expression in each {e partition} of the enclosing
      frame, the rows whose columns [by] are the same keys
      ({!Type.compare_value}), each partition ordered by [order]. A reduction
      is broadcast over its partition's rows. A row expression such as
      {!shift} or {!rank} runs within its partition and returns its values to
      their rows. *)

  val over : ?by:string list -> ?order:Order.t list -> ('a, 's) t -> ('a, row) t
  (** [over ~by ~order e] is [e] evaluated in each partition of the enclosing
      frame ({!section-frames}). Its type is [e]'s. [by] defaults to no
      columns, which is one partition. [order] defaults to none, which is the
      frame's order.

      At the top of [derive] or [filter] the frame is the whole input, so
      [over (mean x)] is the input's mean and it blocks the pipeline
      ({!Query}).

      Its problems are a column named twice in [by] or in [order], and an
      [order] key on a column whose type is or holds an extension type.

      Raises [Invalid_argument] over traced columns ({!Query.section-compiled})
      if [by] is nonempty. *)

  val shift : int -> ('a, row) t -> ('a, row) t
  (** [shift n a] is [a]'s value [n] rows earlier in the frame, or [-n] rows
      later if [n] is negative, and null past the frame's edges. *)

  val rank : ('a, row) t -> (int, row) t
  (** [rank a] is the 1-based rank of [a]'s value among the frame's non-null
      values in talon's total order, ties taking the lowest rank (SQL's [RANK]),
      as [int64], and null where [a] is null. An extension type needs an ordered
      declaration. *)

  (** {2:windows Windows}

      A {e window} is, at each row of a frame, a frame of the rows around it,
      and a reduction over it gives the row's value. Each row has a {e key},
      the value of an expression [on] or, without [on], the row's place in the
      frame from [0]. Row [i]'s window holds the rows of its frame whose key
      [k] has [ki - before < k <= ki + after] in talon's total order, where
      [ki] is row [i]'s key and the {e bounds} [before] and [after] are
      expressions of the frame, so they may differ by row. At a row whose key
      or bound is null, the value is null. Every window leaves out the rows
      whose key is null. *)

  val window :
    ?on:('k, row) t ->
    ?before:('d, row) t ->
    ?after:('d, row) t ->
    ?exclude_key:bool ->
    ('a, agg) t ->
    ('a, row) t
  (** [window ~on ~before ~after ~exclude_key r] is, at each row, [r] over the
      row's window ({!section-windows}). Its type is [r]'s. Without [before]
      the window has no lower limit, and [after] defaults to [0]. With
      [~exclude_key:true] it leaves out the rows whose key is [ki], which
      without [on] is the row itself.

      [window (sum x)] is the running sum, and [window ~on:ts ~before:(span
      (Time.Span.days 7)) ~exclude_key:true (sum x)] sums the seven days before
      [ts]. Without [on] the key is the row's place, so {!over}[ ~order] sets
      it, as in [over ~order:[ Order.asc "day" ]
      (window ~before:(int 28) (mean x))], the mean of a day and the 27 before
      it.

      It blocks the pipeline as {!over} does. {!section-window_keys} and
      {!section-window_floats} give its keys and bounds, and
      {!section-window_cost} its cost.

      Its problems are an [on] of another type than {!section-window_keys}
      allows, a bound of another kind, type or unit, and {!over}, {!shift},
      {!rank} or [window] inside [r], outside a {!per_list}'s body. *)

  (** {3:window_keys Keys and bounds}

      [on] is an integer, a float, a date, a datetime, a duration or a clock.
      A bound is a number of [on]'s kind, a duration of [on]'s unit or a
      coarser one for a temporal [on], and an integer without [on]. A window
      holds its rows in key order, the rows of one key in the frame's order,
      which {!over}[ ~order] sets.

      [ki - before] and [ki + after] compute in the type where [on] and its
      bounds meet. They are exact for integers, and for temporal keys, which
      compute in ticks of [on]'s unit (seconds for a date). An exact result
      past the type's range lies below or above every key. *)

  (** {3:window_floats Float keys and results}

      Float bounds follow IEEE 754 and round once, and NaN is the greatest
      key, so that:
      - an infinite or NaN key's window is empty under finite bounds, and
        without [before] holds every key up to its own;
      - at a finite key, [~before:(float infinity)] reaches every key above
        [neg_infinity] and [~after:(float infinity)] every key up to
        [infinity];
      - a NaN [before] holds no key, and a NaN [after] every key from the
        window's lower limit on, NaN included.

      A float sum's bits depend on the window's rows, their places among the
      rows of the enclosing frame, all partitions included, and whether the
      window has [before] or [~exclude_key:true] with [after]. They may differ
      from {!Query.aggregate}'s over the same rows. With [~exclude_key:true]
      and [after], [var] and [std] are {!Query.aggregate}'s over the window's
      rows. Otherwise, with [before], they come from {!Nx.moments_ranges} over
      the enclosing frame's values, so that their bits depend on the window's
      values and their places among them, and without [before] they may differ
      from both. *)

  (** {3:window_cost Cost}

      With [on], the frame's rows are sorted once by key. {!only}, the [arg_]
      reductions, and {!min} and {!max} of values other than integers and
      booleans also sort their values. Without [before], {!count}, {!sum},
      {!mean}, {!first}, {!last}, {!var} and {!std} then take one scan, O(n)
      for n rows, and {!min}, {!max}, {!only} and the [arg_] reductions take
      that scan too. With [before], all of these take O(n log n), as {!var}
      and {!std} without it do over float64 values whose variance overflows
      or underflows. {!median}, {!quantile} and {!n_unique} reduce every
      window's rows, in time proportional to the windows' total size, which
      is O(n{^ 2}) without [before]. They hold in memory at most 2{^ 20} of
      those rows, or one window's rows when a window has more. A window with
      [~exclude_key:true] and an explicit [after] costs in all of these as
      one with [before], but its {!var} and {!std} reduce every window's
      rows, as {!median} does. *)

  (** {2:lists Lists}

      A list has one {e position} per element, and a null list none, so its
      length is 0. A row's lists are read over a frame with one row per
      position, in list order. An output's column holds the list's element at
      the position, and every other column of the enclosing frame the row's
      value. Several outputs are read in lockstep, so position [j] holds
      element [j] of each, and lists of one row must have one length. Frames
      nest, so a [per_list] of an element that is itself a list reads that
      inner list. {!rows} over a list's frame is the list's length. {!over},
      {!shift}, {!rank} and {!window} run within the list, and each must read
      an output of its list. *)

  val per_list : row out list -> ('a, agg) t -> ('a, row) t
  (** [per_list ls r] is, at each row, [r] over the frame of the elements of
      the lists that the outputs [ls] compute ({!section-lists}). Its type is
      [r]'s. A list's norm is:
      {[
      let xs = Col.v Kind.(list float) "xs" and x = Col.float "x"
      let norm = per_list [ "x" := xs ] (sum (x *. x)) ** float 0.5
      ]}
      [per_list [ "t" := spikes ] (mean (t -. shift 1 t))] is a spike train's
      mean interval. {!section-list_values} gives its nulls and float results,
      and {!section-list_cost} its cost.

      Where its outputs hold no {!over}, {!shift}, {!rank} or {!window}, its
      value at a row reads that row alone, so it streams. It may stand in a
      {!window}'s operand. A failure in [r] is at the row of its list.

      Its problems are a hiding output ({!section-list_outputs}), an output
      that is not a list, no output, two outputs of one name, and an {!over},
      {!shift}, {!rank} or {!window} in [r] that reads no output.

      Raises [Invalid_argument] under [Rune.jit] when it reads a row's list,
      and under [Rune.vmap] where the lists' offsets or validity are traced
      ({!Query.section-compiled}).
  *)

  (** {3:list_outputs Outputs}

      An output must not hide a column of the enclosing frame other than a
      list it outputs unchanged, since [r] could not read the hidden column.
      [keep] therefore outputs a list under its own name, and [r] reads its
      element there.

      An {!over}, {!shift}, {!rank} or {!window} that reads only the row's
      values, which do not change along the list, would look like the
      enclosing frame's, so it is a problem. *)

  (** {3:list_values Nulls and floats}

      A row where an output is null has no frame, whatever the lengths of its
      other lists, so [r] does not run there, and the value is null, that of
      {!count} and {!rows} included. Lists of one row whose lengths differ are
      a data error at that row when none of them is null. An empty
      list's frame has no rows, so reductions give their value over no values
      ({!section-reductions}). A null element is a null value of the frame,
      which a reduction reads as it reads any null.

      A float {!sum} adds a list's values in list order, a float {!mean} sums
      them in [float64] in list order and divides by their count, and {!var}
      and {!std} sum squared deviations from the list's mean in list order, less
      the term that takes out the mean's rounding. *)

  (** {3:list_cost Cost}

      A reduction in [r] costs O(n + e) for n rows and e elements, as a
      reduction of e rows costs, and a {!window} costs as {!section-window_cost}
      states. {!median}, {!quantile}, {!rank}, a {!window}[ ~on] and an
      [over ~order] in [r] sort, and a column of the enclosing frame that [r]
      reads is copied once per element.
  *)

  (** {1:records Records}

      A record expression computes a value of a {!Type.constructor-Record}
      type from named expressions of the record's shape. A record and its
      fields have separate nulls, and {!is_null} of a record tests the record
      alone. *)

  val list : ('a, 's) t list -> ('a array, 's) t
  (** [list es] is the list of [es]' values at each row, in order, as a list of
      the type where the types of [es] meet ({!Type.common}), with a null
      element for each null value. It is never null: [list [ x; null ]] where
      [x] is null is a list of two null elements. A literal or [null] element
      takes that type, and {!store} gives one where nothing does. A list
      operand stays one element, so a list of lists nests and never flattens.

      It computes under [Rune.jit] and [Rune.vmap], since the lists' sizes
      follow from the columns' lengths, but reading a list raises under
      [Rune.jit] as for any list column: in a result, and under {!per_list}
      and {!Query.unnest} ({!Query.section-compiled}).

      Its problems are an empty [es], and operands that do not meet. *)

  val record : 's out list -> (Record.t, 's) t
  (** [record os] is the record of the outputs [os], its fields named as they
      are, in order: [record [ "x" := x; keep (Sel.names [ "flux" ]) ]]. It is
      never null, even where every field is. [if_ c (record os) null] is null
      where [c] is not [true].

      Its problems are those of binding [os], and two outputs of one name. *)

  val field : 'a Kind.t -> string -> (Record.t, 's) t -> ('a, 's) t
  (** [field k n r] is the field [n] of [r] read as [k], null where [r] is
      null or the field is: [field Kind.float "flux" band]. [k] binds the field
      as a {!Col} handle binds a column, so [field] reads fields of types
      without an extension, and {!within} reads any.

      Its problems are a field that [r] lacks and one that [k] does not read.
  *)

  val within : (Record.t, row) t -> 's out -> 's out
  (** [within r o] is the outputs [o] bound over [r]'s fields, so that inside
      [o] a name reads the field of that name and no column of the frame, null
      where [r] is null or the field is. [within r (keep Sel.all)] spreads a
      record, [within r (each Sel.all { column = (fun n x -> "b_" ^ n := x) })]
      renames, and [within r ("snr" := Col.float "flux" /. Col.float "err")]
      computes. To combine a field with a column of the frame, read the field
      with {!field} outside [within].

      Its problems are those of binding [o] against [r]'s fields, a name that
      [r] lacks included, an [r] that is not a record, and, under
      {!Query.derive} and {!Query.unnest}, an output named as a column of the
      input, which would replace that column. *)

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

  (** {1:text Text} *)

  (** Text.

      Text counts and slices Unicode scalar values. Categorical values are text.
  *)
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

    val matches : pattern -> (string, 's) t -> (bool, 's) t
    (** [matches p a] is [true] iff [p] matches [a]. *)

    val split : string -> (string, 's) t -> (string array, 's) t
    (** [split sep a] is the pieces of [a] between the occurrences of the
        literal [sep], found left to right without overlap, as a [list[string]]:
        k occurrences give k + 1 pieces, so a leading, trailing or repeated
        [sep] gives an empty piece, and the empty text one empty piece.
        [split "_" s] of ["sub-01_ses-02"] is [["sub-01"; "ses-02"]], and the
        pieces joined by [sep] are [a].

        Raises [Invalid_argument] if [sep] is empty or not valid UTF-8. *)

    val replace : string -> by:string -> (string, 's) t -> (string, 's) t
    (** [replace sub ~by a] is [a] with each occurrence of the literal [sub]
        replaced by [by], found left to right without overlap:
        [replace "aa" ~by:"b"] of ["aaa"] is ["ba"].

        Raises [Invalid_argument] if [sub] is empty, or if [sub] or [by] is not
        valid UTF-8. *)

    val trim : (string, 's) t -> (string, 's) t
    (** [trim a] is [a] without the white space at its start and end. White
        space is the scalar values that have the White_Space property in Unicode
        18.0.0's PropList.txt: U+0009 to U+000D, U+0020, U+0085, U+00A0, U+1680,
        U+2000 to U+200A, U+2028, U+2029, U+202F, U+205F and U+3000. U+200B and
        U+FEFF, which have no width, are kept, and text of white space alone
        trims to the empty text. *)

    val lower : (string, 's) t -> (string, 's) t
    (** [lower a] is [a] in lowercase by Unicode's Default Case Conversion
        (Unicode 18.0.0, chapter 3, 3.13): each scalar value maps to its full
        lowercase mapping, from SpecialCasing.txt where it has an unconditional
        entry and from UnicodeData.txt otherwise, so the text can change byte
        length (U+0130 lowers to [i] and U+0307, three bytes for two). U+03A3
        lowers to U+03C2, the final form, where a Cased scalar precedes it and
        no Cased scalar follows it, Case_Ignorable scalars being passed over on
        both sides as Table 3-17's Final_Sigma states, and to U+03C3 otherwise.
        No language's tailoring is applied: Turkish, Azeri and Lithuanian text
        lowers as any other. *)

    val upper : (string, 's) t -> (string, 's) t
    (** [upper a] is [a] in uppercase by the same rules, from the full uppercase
        mappings: U+00DF becomes [SS], and U+FB03 [FFI]. Uppercase mapping
        depends on no context, and no language's tailoring is applied. *)

    val concat : string -> (string, 's) t list -> (string, 's) t
    (** [concat sep es] is the texts of [es] joined by the literal [sep], as
        [String.concat sep] joins strings: [concat "_" [ session; unit ]] of
        ["s01"] and ["u3"] is ["s01_u3"]. It is null where one of [es] is null,
        and [coalesce [ e; string "" ]] reads a null [e] as the empty text.
        [concat sep []] is the empty text.

        Raises [Invalid_argument] if [sep] is not valid UTF-8. *)

    val parse : 'a Type.t -> (string, 's) t -> ('a, 's) t
    (** [parse ty a] is the value of type [ty] that the text [a] writes, in the
        forms that [talon.csv] reads: [true] and [false]; decimal integers with
        an optional sign; decimal or scientific floats, [inf], [-inf] and [nan];
        ISO 8601 dates; and ISO 8601 datetimes, with an offset or [Z] for an
        instant and without one for a wall clock. Any other text, and a value
        that [ty] does not hold, is a data error. [ty] is a [bool], integer,
        float, [date] or [datetime] type, or a categorical, which takes only the
        strings of its dictionary. *)
  end

  (** {1:time Time} *)

  (** Temporal values.

      A datetime column holds instants or wall clocks ({!Time.section-stamps}),
      and a zone is an argument only of the conversions {!wall} and {!localize}.
      {!field}, and {!floor} or {!offset} by a calendar step, read a wall clock
      as it is; on an instant they are a problem that names {!wall}.

      A zone's clock can repeat or skip a reading. The readings of a {e fold}
      occur twice, on the offset before the change and then on the one after,
      and those of a {e gap} never occur. A {e policy} resolves such a reading:
      - [`Earlier] is the earlier instant, a fold's first occurrence or a gap's
        reading moved back by the gap's length;
      - [`Later] is the later instant, a fold's second occurrence or a gap's
        reading moved forward by the gap's length;
      - [`Null] is null, and [`Fail] a data error naming the reading. *)
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

    val div : (Time.span, 's) t -> (Time.span, 's) t -> (float, 's) t
    (** [div a b] is how many of [b] fit in [a], as [float64], for example
        [div (diff ts signup) (span (Time.Span.days 1))]. Both operands are
        durations, each converted to [float64] and the coarser scaled to the
        finer's unit, so a quotient of values that [float64] holds exactly
        rounds once. A zero [b] gives an infinity or NaN as IEEE 754 divides
        them. A clock is a problem; cast it to a duration first. It is null
        where an operand is. *)

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

    val field : field -> ('a, 's) t -> (int, 's) t
    (** [field f a] is the field [f] of [a], as [int64]:
        [field `Year orderdate]. [a] is a date, a clock or a wall clock. A date
        has no time-of-day field and a clock no calendar field: asking for one
        is a problem. *)

    val floor : Time.step -> ('a, 's) t -> ('a, 's) t
    (** [floor step a] is the first value of the period of [step] that holds
        [a], a date or a datetime. Periods are counted from 1970-01-01 00:00,
        and weeks from Monday 1970-01-05. A date floors by calendar steps only,
        a wall clock by either, and an instant by exact steps; an exact step is
        whole ticks of the datetime's unit. Its type is [a]'s.

        Raises [Invalid_argument] if [step] is not positive. *)

    val offset : Time.step -> ('a, 's) t -> ('a, 's) t
    (** [offset step a] is [a], a date or a datetime, moved by [step]: an exact
        step adds ticks, and a calendar step moves the calendar date of a wall
        clock; a day of the month past the month's end becomes its last day. A
        date moves by calendar steps only, a wall clock by either, and an
        instant by exact steps of whole ticks of its unit. Its type is [a]'s. *)

    val wall : Tz.zone -> (Time.instant, 's) t -> (Time.wall, 's) t
    (** [wall z a] is the reading of [z]'s wall clock at the instant [a], in
        [a]'s unit: [a] plus the offset of [z] at [a] ({!Tz.offset_s}). A result
        out of range is a data error. *)

    (** The type for what {!localize} and {!localize_ordered} do with a reading
        that a zone repeats or skips, as the section's preamble describes. *)
    type policy = [ `Earlier | `Later | `Null | `Fail ]

    val localize :
      Tz.zone ->
      ambiguous:policy ->
      gap:policy ->
      (Time.wall, 's) t ->
      (Time.instant, 's) t
    (** [localize z ~ambiguous ~gap a] is the instant at which [z]'s wall clock
        reads [a], in [a]'s unit: [a - o] for the offset [o] of {!Tz.local}'s
        [Unique o]. A reading that [z] repeats follows [ambiguous], and one it
        skips [gap]. On a gap, [`Later] is [a - before] and [`Earlier] is
        [a - after] in {!Tz.local}'s terms. [localize z ~ambiguous ~gap
        (wall z a)] is [a] where [z] repeats no reading. An instant out of range
        is a data error. *)

    val localize_ordered :
      Tz.zone ->
      ambiguous:policy ->
      gap:policy ->
      (Time.wall, row) t ->
      (Time.instant, row) t
    (** [localize_ordered z ~ambiguous ~gap a] is {!localize} where the order of
        each frame's rows ({!section-frames}), which must be time order, decides
        the pass of a reading that [z] repeats. The rows of one fold,
        consecutive apart from nulls, read as the fold's first pass up to the
        first row strictly below its predecessor, and as the second from it. Two
        equal rows that are a fold's only rows read one per pass when the row
        on each side lies the fold's length from them, as a logger of that
        cadence writes them. Other rows of a fold follow [ambiguous]. Order
        finds the pass only at a cadence at or below the fold's length, and at
        that cadence a duplicate whose other pass is lost reads as the two
        passes. Nulls and rows moved by [gap] are not a previous row. An equal
        instant is allowed. A row not moved by [gap] whose instant is before
        the previous row's is a data error, as is an instant out of range. *)

    val parse : string -> 'a Type.t -> (string, 's) t -> ('a, 's) t
    (** [parse fmt ty a] reads the text [a] in the format [fmt] as a value of
        [ty], a [date], [datetime] or [clock] type. [fmt] holds the directives
        [%Y] (the year), [%m], [%d], [%H], [%M], [%S] (two digits each), [%f]
        (one to nine digits of a fraction of a second), [%z] ([Z] or a [±hh:mm]
        offset) and [%%]; other characters match themselves. A year is four
        digits, or a sign and at least four digits, as {!Time.Date.pp} writes
        it. A field that [fmt] leaves out is that of 1970-01-01 00:00:00; a date
        reads the day alone, and a clock the time of day. An instant needs
        [%z] in [fmt], and a wall clock takes none. Text that does not
        match, and a value that [ty] does not hold, are data errors.

        Raises [Invalid_argument] if [fmt] holds another directive. *)

    val format : string -> ('a, 's) t -> (string, 's) t
    (** [format fmt a] writes [a], a date, datetime or clock, in the format
        [fmt], as {!parse} reads it: a datetime as its clock reads, a clock on
        1970-01-01; [%f] writes nine digits and [%z] writes [Z]. [%z] needs an
        instant.

        Raises [Invalid_argument] as {!parse} does. *)
  end

  (** {1:hashing Hashing}

      {!hash} is the xxHash64 of each row's value, in the xxHash
      specification's sense, with a seed. Its result is the 64-bit hash
      shifted right by two bits, so it lies in \[[0];[max_int]\] and [mod]
      keeps it non-negative. From 1.0 the definition below is a frozen
      format, so a split made with one release holds in every later one.

      A value's input to the hash is decided by what its kind reads, whatever
      its storage:
      - a boolean is the integer [0] or [1], and an integer its value as a
        64-bit two's complement word, so every width of one number hashes
        alike and a [uint64] beyond [int64] hashes as the word it stores;
      - a float is its [float64] value's bits, [-0.] as [0.] and every NaN as
        the word [0x7FF8000000000000];
      - a date is its [int64] days from 1970-01-01, and a clock, a duration
        or a datetime its nanoseconds, wrapped to 64 bits;
      - a word above is hashed as eight little-endian bytes; a string, a
        categorical's text and a [binary] as its bytes;
      - a null at any depth hashes the eight bytes of the word
        [0x9E3779B185EBCA87];
      - a record hashes, as one input, its fields' 64-bit hashes, before the
        shift, as eight little-endian bytes each, in field order, with field
        names left out.

      The seed is the OCaml [int] sign-extended to 64 bits, [0] by default, the
      same at every depth of a record. {!hash} over a list, a tensor or an
      extension is a problem. Exact casts within a kind keep the hash, so
      [hash (cast ty a) = hash a] for them. *)

  val hash : ?seed:int -> ('a, 's) t -> (int, 's) t
  (** [hash ~seed a] is the hash of each row of [a] ({!section-hashing}), as
      [int64], and is never null. A column whose bytes nx cannot read, a traced
      one other than a categorical, raises [Invalid_argument] naming
      [Expr.hash] ({!Query.section-compiled}). *)

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
        {!nx} and {!nx2} as [<fn>];
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

  val date : string -> (Time.date, Expr.row) Expr.t
  (** [date name] is [v Kind.date name]. *)

  val instant : string -> (Time.instant, Expr.row) Expr.t
  (** [instant name] is [v Kind.instant name], which binds every datetime type
      of instants. *)

  val wall : string -> (Time.wall, Expr.row) Expr.t
  (** [wall name] is [v Kind.wall name], which binds every datetime type of wall
      clocks. *)

  val span : string -> (Time.span, Expr.row) Expr.t
  (** [span name] is [v Kind.span name], which binds every duration and clock
      type. *)
end

module Ext : sig
  (** Extension declarations.

      An extension type is a named type stored as another ({!Type.val-ext}). A
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
      - [enc], which stores an ['e]: literals, {!Expr.const} values,
        {!Expr.is_in} values and {!Expr.cut} edges encode with it. It must be
        injective;
      - [ordered], a promise that the order of stored values is the order of the
        values, as for an epoch stored normalized. With [~ordered:false],
        {!Expr.min}, {!Expr.max}, {!Expr.arg_min}, {!Expr.arg_max},
        {!Expr.rank}, {!Expr.cut} and [<] are problems on its values. Sort and
        join keys are names, which no declaration binds.

      [metadata] defaults to [""].

      Raises [Invalid_argument] as {!Type.val-ext} does. *)

  val col : ('e, 's) t -> string -> ('e, Expr.row) Expr.t
  (** [col e name] is the column [name] read through [e]. It binds only a column
      whose type is [e]'s: the same name, metadata and storage type, so a
      declaration of metres never binds a column of kilograms. *)
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
      {!Pred.Not}, {!Pred.And} and {!Pred.Or} are Kleene's, so a row passes a
      predicate iff the predicate is [true] on it. A source answers {!Exact}
      only for a predicate whose meaning it honours entirely: for example, a
      float column's minimum and maximum prune a row group only when the source
      knows the group holds no NaN, since NaN passes [>], [>=] and [<>]. *)
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

    val filter : Schema.t -> t -> (bool, Expr.row) Expr.t option
    (** [filter s p] is the filter that [p] is over a table of the columns [s]
        describes: it is [true], [false] or null where [p] is, so
        [Query.filter] of it keeps the rows that pass [p]. A source that
        answers {!Exact} can apply a conjunct with it, on each batch it reads.
        It is [None] if [p] names a column that [s] lacks, or one whose type is
        or contains an extension type, which a filter reads only with its
        declaration ({!Ext.col}). *)
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
      - [sorted], the order the rows come in, by talon's total order. Talon
        checks it as rows arrive, and a row out of order fails the run. Defaults
        to no order.
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

  (** {1:paths Paths}

      A directory of a partitioned dataset names partition values in its path,
      as Hive writes them: [year=2024/month=01/part-0.parquet]. This layout has
      no specification, and what the partition values mean, like their types,
      is the caller's: see {!hive}. *)

  val hive : string list -> (table, Error.t) result
  (** [hive paths] is the table of the keys and values in [paths], one row per
      path in its order: the column ["path"], then one [string] column per key
      of the first path, in the order they appear in it. A path is split at
      each [/], and every part but the last is a {e directory}; a directory
      holding [=] is a key and a value, split at its first [=] and
      percent-decoded by bytes, where only a pair of hex digits decodes. A
      value [__HIVE_DEFAULT_PARTITION__] is null. Each file's keys are matched
      by name across [paths].

      Values are text. Types are the caller's: {!Expr.Str.parse} them, so one
      directory cannot change a column's type for its siblings.

      [Error e] on a path that is not valid UTF-8; an empty key or the key
      ["path"], which holds the file's path; a key or value not valid UTF-8 once
      decoded; a key repeated in one path; or paths whose keys differ. An
      invalid percent sequence is not an error and stays literal. It is pure
      and reads no file: [paths] are any strings. *)
end

(** {1:decimals Decimals in files}

    A decimal of a file, the number [unscaled × 10{^-scale}] of at most a
    precision's digits, has no talon type. A file format leaves its column {e
    undeclared} until the reader declares it with the format's [with_type], as
    [float64], the float nearest each value, or, up to 18 digits, as [int64],
    each value's unscaled integer, a value outside [int64] failing when it is
    read. An [int64] means what its scale says, so a format that declares one
    reads only a file whose column has that scale, and any other fails when
    read. A list or record that holds a decimal is undeclared as a whole. A
    source of a format that leaves a column undeclared refuses to be read,
    naming the column and the types it reads as. *)

module Join : sig
  (** Joins of two queries by conditions on their columns.

      [left |> Query.join ~on right] pairs the rows of [left] and [right] that
      the condition [on] matches. A condition is a value, a conjunction of
      {e atoms} built with {!( && )} inside [Join.( … )], as in
      [Join.(keys [ "ticker" ] && eq "day" "date")]. An atom names a left
      column, then a right column. A {!type-kind} says which rows are kept,
      and a {!type-count} how many matches each row must have. Each condition
      runs as one algorithm ({!section-algorithms}).

      {b Matching.} A join keeps exactly the pairs of rows on which every atom
      holds. It is the join on {!all}, restricted to those pairs. An equality
      atom holds by key identity ({!Type.compare_value}), so null keys match.
      A comparison holds where {!Expr.( < )} or its sibling is [true] on the
      two values, in talon's total order (NaN after every number and [-0.]
      equal to [0.]), and never at a null. A value cannot fail a join, and a
      null or a period whose values come in either order ({!overlaps})
      changes only which pairs match.

      {b Columns.} A {!Semi} or {!Anti} join has the left columns. Another
      join has the left columns, then the right columns without those of
      equality atoms, in order, so the key of [eq l r] appears once, as [l].
      In a {!Full} join that key has the common type of [l] and [r]
      ({!Type.common}), since an unmatched right row gives it [r]'s value.
      Every other column keeps its type. Except in a {!Semi} or {!Anti} join,
      a name on both sides is a problem, and talon adds no suffixes. Rename
      one side first.

      {b Order.} A join's rows come in [left]'s order, each left row followed
      by its matches in [right]'s order, and a {!Full} join then appends
      [right]'s unmatched rows in order. *)

  (** {1:conditions Conditions} *)

  type cond
  (** The type for conditions, which are conjunctions of atoms. *)

  val keys : string list -> cond
  (** [keys ns] matches the rows whose columns [ns] are the same keys on both
      sides. It is [eq n n] for each [n] of [ns].

      Raises [Invalid_argument] if [ns] is empty, since a join on no key is
      {!all}, written so, or if [ns] names a column twice. *)

  val eq : string -> string -> cond
  (** [eq l r] matches the rows whose left column [l] and right column [r] are
      the same key. [l] and [r] must meet ({!Type.common}). *)

  val lt : string -> string -> cond
  (** [lt l r] matches the rows whose left column [l] comes before the right
      column [r], as {!Expr.( < )} compares. [l] and [r] must meet
      ({!Type.common}) at a type with an order, which an extension type
      lacks. *)

  val le : string -> string -> cond
  (** [le l r] is {!lt}, as {!Expr.( <= )} compares. *)

  val gt : string -> string -> cond
  (** [gt l r] is {!lt}, as {!Expr.( > )} compares. *)

  val ge : string -> string -> cond
  (** [ge l r] is {!lt}, as {!Expr.( >= )} compares. *)

  val inside : string -> string * string -> cond
  (** [inside p (s, e)] matches the rows whose left column [p] is from the
      right column [s], included, to the right column [e], excluded. It is
      [ge p s && lt p e], so it matches only rows whose [e] is after their
      [s]. *)

  val overlaps : string * string -> string * string -> cond
  (** [overlaps (ls, le) (rs, re)] matches the rows whose left period
      [(ls, le)] and right period [(rs, re)] overlap, as SQL-92's OVERLAPS
      predicate defines it (ISO/IEC 9075:1992, 8.11), in talon's total order.
      A period's {e start} is the lesser of its two values and its {e stop}
      the greater; with one value null, its start is the other one and its
      stop is unknown, and a period of two nulls overlaps nothing. Two periods
      overlap where their starts are equal and neither stop is unknown, or
      where one's start is after the other's
      start and before the other's known stop. So [(0, 2)] and [(2, 3)] do
      not overlap, [(5, 5)] overlaps [(3, 8)] and [(5, 9)] but not [(1, 5)],
      and [(8, 3)] is [(3, 8)]. The four columns must meet at a type with an
      order. *)

  val position : cond
  (** [position] matches row i of the left with row i of the right. For a
      left side of n rows and a right side of m, {!Inner} and {!Semi} keep
      min(n, m) rows, {!Left} keeps n, {!Full} max(n, m), and {!Anti} the left
      rows past m. *)

  val all : cond
  (** [all] matches every pair of rows. *)

  val ( && ) : cond -> cond -> cond
  (** [c0 && c1] matches the pairs that both [c0] and [c1] match.

      Raises [Invalid_argument] if the conjunction holds {!position} and
      another atom, or holds one atom twice. *)

  (** {1:algorithms Algorithms}

      - A condition of equality atoms alone ({!keys}, {!eq}) runs as an
        {e equality} join, in O(n + m + matches) for sides of n and m rows.
      - Equality atoms, if any, with at least one comparison ({!lt}, {!le},
        {!gt}, {!ge}, {!inside}) or {!overlaps} run as a {e range} join
        ({!section-range}).
      - {!position} takes no other atom and runs as its own algorithm.
      - {!all}, the condition with no atom, runs every pair, so [all && c] is
        [c]. *)

  (** {2:range Range joins}

      A range join sorts one side's rows within their {e equality keys}, the
      columns of its equality atoms, by the {e band column}. Each row of the
      other side searches that order for its {e band}, the rows the bounding
      atoms select. The band column is the column that comparisons name most,
      where a left and a right column of one name are two columns. On a tie
      between the sides it is the right column, and on a tie within a side
      the one written first. If the condition has an {!overlaps}, the first
      one bounds the band and each side is sorted by its periods' starts.
      Otherwise the comparisons that name the band column bound the band,
      when they meet that column at the type of the first such comparison.
      Every other comparison or {!overlaps} is tested on each pair the bands
      read. *)

  (** {2:range_cost Cost of a range join}

      A side needs no sort when its rows come grouped by their equality keys,
      each group ascending in the band column, which the join checks by reading
      them. The right side of an {!overlaps} needs none when its rows come
      grouped that way by their periods' starts. A range join whose bands read
      p pairs, on a side of s rows searched from a side of o rows, costs
      O(s log s + o log s + p), less the sort when the side needs none.
      Matches out of the join's order are reordered once. An {!overlaps}
      join sorts and searches each side. When every atom but the equality
      atoms bounds the band, a {!Semi} or {!Anti} join costs
      O(s log s + o log s), and the checks of [each_left] and [each_right]
      add no p term. *)

  (** {1:kinds Kinds and counts} *)

  (** The type for join kinds, which say which rows a join keeps. A right join
      is a left join with the arguments swapped. *)
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
      selectors ([Expr.across], [Expr.each], [Sel.where], [Expr.nx]) run when
      the verb is applied. They must be pure, and an exception they raise
      propagates from the verb. *)

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
      {!Expr.rank} and {!Expr.window} over the input's rows need the whole
      input: the verb then blocks.

      Its problems are those of binding [os], and two outputs of one name. *)

  val derive : Expr.row Expr.out list -> t -> t
  (** [derive os q] is [q]'s rows with each of [q]'s columns and the outputs
      [os]: an output replaces the column of its name in place, with the
      output's type, and the others follow [q]'s columns, in order. It keeps
      [q]'s row count and order. Each output reads [q]'s columns, never another
      output. It streams in O(n), except that {!Expr.over}, {!Expr.rank} and
      {!Expr.window} over the input's rows need the whole input: the verb then
      blocks.

      Its problems are those of binding [os], two outputs of one name, and an
      output of {!Expr.within} named as a column of [q]. *)

  val filter : (bool, Expr.row) Expr.t -> t -> t
  (** [filter p q] is the rows of [q] on which [p] is [true], in order, so a row
      on which [p] is null is dropped. Its schema is [q]'s. [p] is a [bool]
      column: an OCaml value, as [const f $ x] is, takes that type. It streams
      in O(n), except that {!Expr.over}, {!Expr.rank} and {!Expr.window} over
      the input's rows need the whole input: the verb then blocks.

      Over traced columns ({!section-compiled}) it raises when [p] reads one.

      Its problems are those of binding [p], and an extension's values, even
      when they read as [bool]. *)

  val sort : Order.t list -> t -> t
  (** [sort ks q] is [q]'s rows ordered by the keys [ks] in turn, each by
      talon's total order in its direction, nulls last unless
      {!Order.nulls_first} (see {!Order}). It is stable: rows equal on every key
      keep [q]'s order. Its schema is [q]'s. It blocks, in O(n log n).

      Its problems are a key that names no column, a key that names the column
      of an earlier key, a key on a column whose type holds an extension type,
      which has no order, and a key on an extension column, which orders only
      through its declaration. *)

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
      makes one group of every row, so one row, even when [q] has none.

      It takes O(n) expected time. A stream of several batches holds O(groups
      + largest batch) rows, a state of one row per group, where each output
      reduces with {!rows}, [count], [sum], [mean], [var], [std], [min], [max],
      [first], [last], [only], [arg_min] or [arg_max] of an operand that reads
      its own row alone and holds no [per_list]. Its values and failures do
      not depend on the batches. Otherwise it blocks, holding the columns [by]
      and those that [os] read: for any other reduction, a [sum] of durations,
      an OCaml function or value ([$], [const], [of_option]), an extension
      value, a {!Expr.within}, an output that can fail outside a reduction, or
      a traced batch ({!section-compiled}). Over traced columns it raises
      unless [by] is empty. With a [var] or [std] it raises [Invalid_argument]
      for a traced batch that follows two untraced batches.

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
      [left |> join ~on right]. [kind] says which rows are kept ({!Join.kind})
      and defaults to {!Join.Inner}. [each_left] and [each_right] are the
      numbers of matches each row of [left] and of [right] must have
      ({!Join.type-count}), {!Join.Any} by default. Its rows and columns are
      those {!Join} describes.

      A join on {!Join.all} blocks on [right] and streams [left], and the
      others block on both inputs. The cost of an equality or range join is in
      {!Join.section-algorithms}.

      Its problems are, for each atom of [on] in order, a column missing on
      its side and the atom's own ({!Join.eq}, {!Join.lt}, {!Join.overlaps}).
      Then, in a {!Join.Full} join, a left column that is the left of two
      equality atoms, and the joined columns that have one name, as {!Join}
      states.

      Raises [Invalid_argument] over traced columns ({!section-compiled}) on
      an equality or range join. The others run, their counts included. *)

  val append : t -> t -> t
  (** [append rest q] is [q]'s rows, then [rest]'s, written [q |> append rest].
      The two have the same columns, matched by name. Its schema is [q]'s names,
      in order, each at the type where the column's two types meet
      ({!Type.common}): an [int32] column and an [int64] one append as [int64],
      a categorical and a [string] as [string]. Each side's values widen to it
      exactly. Appending other columns is [Kit.union], which derives the missing
      ones as nulls. It costs O(1) over tables of one schema and streams over
      sources.

      Its problems are a column that only [q] has, one that only [rest] has, and
      each column whose two types do not meet, such as [int8] and [uint8]. *)

  val unnest : ?keep_empty:bool -> Expr.row Expr.out list -> t -> t
  (** [unnest ?keep_empty os q] is the rows of the frames of the lists that the
      outputs [os] compute ({!Expr.section-lists}), concatenated in [q]'s row
      order, each row's positions in list order. Its columns are [q]'s with
      [os], placed as {!derive} places them, an output holding its list's
      element and every other column repeating the row's value.

      A row without positions gives no row. With [~keep_empty:true] (default
      [false]) it gives one row with every output null, so a null list and an
      empty one merge, a padded null reads as a null element, and {!Expr.rows}
      counts that row as one. Lengths that differ, a null beside a non-empty
      list included, are a data error at the row.

      It streams, and blocks where {!derive} would, in O(n + m) for m
      positions, plus the size of each repeated list or text column once per
      position of its row. {!optimize} narrows the repeated columns.

      Its problems are those of binding [os], no output, two outputs of one
      name, an output that is not a list, and an output of {!Expr.within}
      named as a column of [q].

      Raises [Invalid_argument] over traced columns under [Rune.jit], and
      under [Rune.vmap] where the lists' offsets or validity are traced
      ({!section-compiled}). *)

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
      step, as {!pp} prints it, and the row of the step's input. Over traced
      columns ({!section-compiled}) a step that can fail on a value, such as a
      cast that may lose one, raises at the trace.

      A function of [q]'s expressions may be called on rows past a limit of
      [q] as written: the optimized plan may read them, as a filter it moves
      below a sort reads every row. A failure on such a row, an error or an
      exception, ends the run where the optimized plan, evaluated one row at a
      time, reads the row, whatever the batches, and is never met elsewhere.
      Every reader [fold] opens is closed when it returns, also when [f], a
      function of [q]'s expressions or a source raises; the exception then
      propagates.
  *)

  val run : t -> (table, Error.t) result
  (** [run q] is [q]'s rows as one batch: [fold] into a table, then one copy
      into a single batch. Its columns are canonical ({!Column}), so their
      {!Column.type-layout}s do not depend on batches, except that a column
      whose validity is traced ({!section-compiled}) keeps it, even with no
      null, and that a record's fields under its nulls are unspecified. *)

  val values : ('a, Expr.row) Expr.t -> t -> ('a array, Error.t) result
  (** [values e q] is [e] on each row of [q], decoded to OCaml:
      [values Expr.(const mk $ carrier $ option origin) q]. An extension value
      is decoded with its declaration's [dec]. [q] reads only the columns [e]
      reads.

      [Error e] as for {!run}, and where [e] is null on a row, or a value is
      outside what OCaml reads ({!Column.values}): read nullable values through
      {!Expr.option}. [e] then names the step [values] followed by [e].

      Raises [Invalid_argument] with a report like a verb's, named [values], if
      [e] does not bind against [q]'s schema, and over traced columns or under
      [Rune.grad] ({!section-compiled}), as nx refuses to read their arrays
      out. *)

  (** {1:compiled Compiled, mapped and differentiated runs}

      A query may run in a function that [Rune.jit] compiles, [Rune.vmap] maps
      or [Rune.grad] differentiates, over columns of its arrays
      ({!Column.of_tensor}). Under [Rune.jit] and [Rune.vmap] those arrays are
      {e traced}, in the sense that their elements exist only when the program
      runs; under [Rune.jit] so is every result computed from a table the
      function captures, so steps on its text raise. A step over traced columns
      computes eager's values when every size it computes follows from its
      inputs' sizes, and {!Expr.std} and {!Expr.var} compute them within
      rounding. Otherwise it raises [Invalid_argument] at the trace with
      nx's message for the read, such as
      [Nx.positions: the value is traced by Rune.jit], and its doc says so.
      Under [Rune.grad] every step runs, {!values} and display aside. *)

  (** {1:optimizing Optimizing} *)

  val optimize : t -> t
  (** [optimize q] is the plan that running [q] runs: [q] with the same schema
      and the same rows, rewritten to read and compute less. Running a query
      optimizes it first, so [optimize] serves to print the plan that runs and
      to compare plans. It reads no data, it calls the sources' [pushdown], and
      [optimize (optimize q)] is [optimize q].

      {b Results.} The rewrites change no value and no null of a result, and add
      no failure: an operation that can fail meets only rows that [q] shows it.
      They may remove failures: a value that no row and no column of the result
      reads is not computed, so a failure, or an exception from a user function,
      that only such a value meets does not happen. Of the failures that remain,
      a run reports the one at the earliest row of the optimized plan.

      {b Constants.} Arithmetic, comparisons, [not], {!Expr.is_in} and the text
      operations other than {!Expr.Str.split} and {!Expr.Str.concat}, of
      literals, are evaluated as a run evaluates them, and become the literal
      they compute. One whose
      evaluation fails, or gives a value OCaml's type does not hold (an [int64]
      past [max_int]), stays and evaluates at run time. [&&], [||], [is_null],
      [if_], [coalesce], and [store] of a literal fold alike. [a && false],
      [a || true], [a && true], [a || false], an [if_] on a literal and a
      [coalesce] with literals simplify alike. A filter whose predicate is
      [true] goes, and one with a conjunct [false] or null becomes
      [slice ~offset:0 ~length:0], since a row passes only if every conjunct
      is [true].

      {b Predicates.} A filter's predicate splits into its {e conjuncts}, the
      operands of its [&&]s. A conjunct moves toward the sources when it is
      {e row-local}, its value at a row depending on that row alone: outside
      the bodies of its {!Expr.per_list}s, it has no {!Expr.over},
      {!Expr.shift}, {!Expr.rank} or {!Expr.window}. It moves past
      each step that keeps the rows it sees and the values it reads:
      - a [sort];
      - a [filter] whose predicate is row-local, with which it merges;
      - a [select] or a [derive] whose outputs are row-local, when each column
        the conjunct reads is a column of the step's input kept unchanged, under
        its name or another ([select ["y" := x]]), or an output that is a
        literal or null, and the renaming hands no {!Expr.per_list} body's read
        of a column to one of its outputs. Such an output is read as its value
        and the conjunct folds, which simplifies a predicate over constant
        columns: over a fold of files with partition columns derived as
        literals, a conjunct that folds to [false] or null empties that branch
        before its source is read, and [false || x] becomes [x], which the
        source can use;
      - an [aggregate] with keys, when the conjunct reads only keys and none of
        their types holds floats, since [-0.] and [0.] are one key with two
        values;
      - an [append], into both inputs;
      - a [join] whose [~each_left] and [~each_right] are [Any], on a condition
        other than {!Join.position}: into the left input when the conjunct reads
        only left columns and the join is [Inner], [Left], [Semi] or [Anti], and
        into the right input when it reads only right columns and the join is
        [Inner].

      It stops at a [slice], at an [unnest], at a [Full] join, at a join with an
      assertion, at a step with an output or predicate that is not row-local,
      and at a table.
      Pushdown never crosses a join with an assertion: a conjunct entering one
      input changes the matches of the other input's rows, and so could make
      that side's assertion fail on rows the plan shows it. A conjunct that can
      fail, one with a [cast] that narrows, [Str.parse], temporal arithmetic or
      parsing, [of_option] or [$], or a {!Expr.per_list} of several lists or
      with {!Expr.only} or a duration {!Expr.sum}, passes only the steps that
      show it the rows
      they receive: a [sort], a [select] or a [derive], an [aggregate]'s keys
      and an [append]. Below a [filter] or a [join] it would meet rows that [q]
      never shows it.

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
      that the result has, each [select], [derive] and [aggregate] drops the
      outputs that nothing reads, and an [unnest] repeats only the columns read
      above it, though it keeps every output, which sets its rows. A [select]
      that keeps its input's columns unchanged, in order, goes, and so does a
      [derive] left without outputs,
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
      - outputs as ["name" := e], with selectors resolved, and a run of columns
        kept unchanged under their names as one [keep (names ["a"; "b"])];
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
      whatever the schema, or for a requirement that no verb states.
      {!time_range} scales a step by a position, which no verb computes. *)

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
      A query without columns has one distinct row if it has rows. *)

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

  val longer : name:string -> value:string -> Sel.t -> Query.t -> Query.t
  (** [longer ~name ~value sel q] is [q] with the columns of [sel] replaced by
      two columns: [name], the categorical whose dictionary is the selected
      names in order, and [value], their cells at the type where their types
      meet ({!Type.common}), one row per cell, nulls included. The other columns
      come first, repeated, in [q]'s order, then [name] and [value], each row's
      cells in column order. It is:
      {[
      let ns = (* the names [sel] selects, in order *) in
      let others = (* [q]'s columns but [ns] *) in
      let cells = Array.of_list ns in
      Query.unnest
        Expr.[
          name := store Type.(list (categorical cells)) (const cells);
          value := list (List.map (Col.v k) ns);
        ]
        q
      |> Query.select Expr.[ keep (Sel.names (others @ [ name; value ])) ]
      ]}
      with [k] the kind of the first selected column. It streams in O(n k),
      and building and optimizing its plan take O(k{^ 2}) time. Filtering with
      {!Expr.is_null} after it drops null cells. Selected columns of different
      kinds are a problem of the handle; widen them first with {!Expr.cast}.

      Raises [Invalid_argument] if [sel] selects no column, if [name] or [value]
      is a column [sel] does not select, or if the first selected column holds
      an extension, which no {!Col} handle reads. *)

  val wider : by:string list -> key:string -> value:string -> Query.t -> Query.t
  (** [wider ~by ~key ~value q] is one row per group of [q]'s rows that have the
      same [by], in order of first appearance, holding [by], then one column
      per category of [key]'s categorical type, in dictionary order, holding
      [value] at the row of that group and category, null where there is none
      ({!Expr.pivot}). With [by] empty it is one row, even from no rows. It is:
      {[
      Query.aggregate ~by
        Expr.[
          each (Sel.names [ value ])
            { column = (fun n x -> n := pivot (Col.string key) x) };
        ]
        q
      |> Query.filter Expr.(not (is_null (Col.v Record.kind value)))
      |> Query.select
           Expr.[
             keep (Sel.names by);
             within (Col.v Record.kind value) (keep Sel.all);
           ]
      ]}
      The columns come from [key]'s type, so the plan's schema is known before
      any data is read. Cast a key to the categorical type whose dictionary
      names the columns, or read them from the data with a run:
      {[
      let open Result.Syntax in
      let* q = Kit.categorize [ "element" ] q in
      Ok (Kit.wider ~by:[ "station"; "date" ] ~key:"element" ~value:"value" q)
      ]}
      An unused category is a column of nulls; {!categorize} a filtered query
      to read only its categories. {!Expr.pivot} states the failures: duplicates
      in a cell, a null [key] holding a value, and filtering before it. After
      {!longer} ~name ~value sel, [wider ~by ~key:name ~value] restores the
      input when the selected columns have one type, [by] is a key of the
      input, lists its other columns in their order, and is non-empty or the
      input has a row, and those columns precede the selected ones. Equality
      there is by key identity ({!Type.compare_value}).

      It blocks in O(n + groups k), and building and optimizing its plan take
      O(k{^ 2}) time in its number of columns. Its problems are those of its
      [aggregate] and [select]: a [key] that is not categorical, a category
      named as a column of [by]. *)

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
      A column that both have, of two types, takes the type where they meet, and
      one whose types do not meet is a problem of the [append]. *)

  val union_all : Query.t list -> Query.t
  (** [union_all qs] is the rows of [qs], in order, with the columns of the
      first query, then each column that a later one adds, in order, null on
      the queries that lack it. Each column has the type where all of its
      types meet ({!Type.common}), which no order of [qs] can change: columns
      of [int8], [uint8] and [int16] make [int16], where a fold of {!union} in
      that order fails. Every query is cast to those types and padded before
      any is appended, in a balanced tree of [append]s, so its plan takes work
      linear in the length of [qs], beyond a comparison of each leaf with those
      of its name and shape. A filter on a column stored at its common
      type reaches every such query, whatever the others' types; a query whose
      column was cast is read and filtered after the cast. A column whose types
      do not meet is a problem of the [append]s. [union_all []] is a query of no
      column and no row, the identity of {!union}: [q |> union (union_all [])]
      has [q]'s rows and columns. *)

  (** {1:windows Windows}

      These compose {!Expr.window}, so its problems on [r] apply to [x] and
      [r]. *)

  val cumulative : ('a, Expr.agg) Expr.t -> ('a, Expr.row) Expr.t
  (** [cumulative r] is [r] over the frame's rows up to and including each:
      {[
      Expr.window r
      ]} *)

  val fill_forward : ('a, Expr.row) Expr.t -> ('a, Expr.row) Expr.t
  (** [fill_forward x] is [x], or where [x] is null its last earlier value in
      the frame:
      {[
      Expr.(window (last x))
      ]} *)

  val fill_backward : ('a, Expr.row) Expr.t -> ('a, Expr.row) Expr.t
  (** [fill_backward x] is [x], or where [x] is null its next value in the
      frame:
      {[
      Expr.(window ~before:(int 1) ~after:(int max_int) (first x))
      ]} *)

  val interpolate :
    ?on:('k, Expr.row) Expr.t -> (float, Expr.row) Expr.t ->
    (float, Expr.row) Expr.t
  (** [interpolate ~on x] is [x] where it holds a value. Elsewhere it is the
      value at the row's key on the line through the last value at or before
      that key and the first value after it, in [float64]. Where a row of the
      same key holds a value, it is that value. It is null before the first
      value and after the last. Keys are [on]'s values cast to [float64], an
      integer or a float, or without [on] the rows' places. Inside
      [Expr.( … )] it is:
      {[
      let line ~at ~x0 ~k0 ~x1 ~k1 =
        coalesce
          [ x; if_ (at = k0) x0 (x0 +. (x1 -. x0) *. (at -. k0) /. (k1 -. k0)) ]
      in
      match on with
      | Some on ->
          let k = cast Type.float64 on in
          let held v = if_ (is_null x) null v in
          let upto r = window ~on:k r in
          let past r =
            window ~on:k ~before:(float 0.) ~after:(float infinity) r
          in
          line ~at:k ~x0:(upto (last x)) ~k0:(upto (last (held k)))
            ~x1:(past (first x)) ~k1:(past (first (held k)))
      | None ->
          (* [c] values reach the row; the [c]th starts the run of rows of
             count [c], which ends before the next value. *)
          let place r = cast Type.float64 r in
          let c = window (count x) in
          let k0 = place (window ~on:c ~exclude_key:true rows + int 1) in
          let k1 = k0 +. place (window ~on:c ~before:(int 1) rows) in
          line ~at:(place (window rows)) ~x0:(window (last x)) ~k0
            ~x1:(window ~before:(int 0) ~after:(int max_int) (first x)) ~k1
      ]} *)

  (** {1:asof As-of joins}

      An {e as-of join} pairs each row of a query [left] with at most one row
      of a query [right]. [l] is a column of [left] and [r] a column of
      [right]. The {e matches} of a left row are the right rows with its
      values in the columns [by] and with [r <= l] for {!Backward} or
      [r >= l] for {!Forward}. Under [~strict:true] they leave out those whose
      [r] equals [l]. Its {e pick} is the match nearest [l] in a
      {!type-direction}. [l] and [r] compare as {!Expr.( < )} does, so a null
      key matches nothing and NaN is the greatest value, and [by] matches by
      key identity, as {!Join.keys} does. *)

  (** The type for the directions of an as-of join. *)
  type direction =
    | Backward
        (** The match of the greatest [r], at or before [l], and among matches
            of equal [r] the last in [right]'s order, as {!Expr.last} takes it
            in a frame sorted stably by [r]. *)
    | Forward
        (** The match of the least [r], at or after [l], and among matches of
            equal [r] the first in [right]'s order, as {!Expr.first} takes it
            in a frame sorted stably by [r]. *)
    | Nearest
        (** Of the backward and the forward pick, the one nearer [l], and the
            backward pick on a tie. It takes keys that meet at an integer, a
            float, a date, a datetime, a duration or a clock type. The
            distance between two values is [0] where they are equal, infinite
            where one is infinite or NaN, and else the exact absolute value of
            their difference. Between datetimes or durations the difference is
            a span ({!Expr.Temporal.diff}). Where a left row has both picks, a
            span past a duration's range is a data error. *)

  val join_asof :
    ?by:string list ->
    ?direction:direction ->
    ?strict:bool ->
    on:string * string ->
    Query.t ->
    Query.t ->
    Query.t
  (** [join_asof ~by ~direction ~strict ~on:(l, r) right left] is each row of
      [left], in order, with its pick in [right] ({!section-asof}), written
      [left |> Kit.join_asof ~on right]. [direction] defaults to {!Backward}
      and [strict] to [false]. Its columns are [left]'s, then [right]'s without
      [by], null where a row has no pick. Neither side need be sorted. A
      tolerance is applied after the join ({!section-asof_tolerance}).
      {!section-asof_definition} shows its definition.

      A name on both sides, [r] included when it is [l], is a problem of the
      join.

      Raises [Invalid_argument] if [by] names a column twice, or if
      [direction] is {!Nearest} and the keys meet at another type. *)

  (** {2:asof_tolerance Tolerances}

      A tolerance is a comparison [within] on the pick's key, such as
      [Expr.(Col.int r >= Col.int l - int 5)]. [Query.filter within] keeps
      the rows picked within it and drops the others. To keep every left row
      and null the picks beyond it, derive
      [each sel { column = (fun n x -> n := if_ within x null) }], where [sel]
      selects [right]'s columns without [by]. *)

  (** {2:asof_definition Definition}

      Inside [Expr.( … )], where [key], [side], [j] and [row] are names that
      neither side has and a name in an expression reads its column,
      {!join_asof} is:
      {[
      let tagged q c s number =
        Query.select [ keep (Sel.names by); key := c; side := int s; number ] q
      in
      let pick ~forward x =
        let ahead = if forward = strict then Order.asc else Order.desc in
        let along = over ~by ~order:[ Order.asc key; ahead side ] in
        if forward then
          along (window ~before:(int 1) ~after:(int max_int) (first x))
        else along (window (last x))
      in
      let picks =
        tagged left l 1 (j := store Type.int64 null)
        |> Query.append
             (tagged right r 0 (j := if_ (is_null r) null (window rows)))
        |> Query.derive [ row := if_ (is_null key) null picked ]
        |> Query.filter (side = int 1)
        |> Query.select [ keep (Sel.names [ row ]) ]
      in
      left
      |> Query.join ~on:Join.position picks
      |> Query.join ~kind:Left ~on:(Join.keys [ row ])
           (drop (Sel.names by) right |> Query.derive [ row := window rows ])
      |> drop (Sel.names [ row ])
      ]}
      [picked] is [pick ~forward:false j] for {!Backward},
      [pick ~forward:true j] for {!Forward}, and for {!Nearest}
      [if_ (is_null jb) ja (if_ (is_null ja) jb (if_ (nearer ka kb) ja jb))],
      where [ja] and [jb] are the backward and forward picks of [j], [ka] and
      [kb] those of [if_ (is_null j) null key] where both picks exist, and
      [nearer] compares the distances to [key]. It sorts the n + m keys once,
      four times for {!Nearest}, and reads each side at most twice. *)

  (** {1:intervals Interval sets}

      These read each row of a query as a {e period} of its columns [(s, e)],
      with its start and stop as {!Join.overlaps} defines them. A period holds
      the values from its start, included, to its stop, excluded. A period
      whose stop equals its start holds its start alone, a point whose length
      is zero. A period with a null value holds none. The rows of one key in
      the columns [by] (key identity, {!Type.compare_value}) form a {e set},
      the values that its periods of positive length hold, so a zero-length
      period adds nothing to a set and takes nothing from one. *)

  (** {2:normal_form Normal form}

      Each function returns a set in {e normal form}, which has its keys in
      ascending order, nulls last, and for each key the fewest periods that
      hold the set's values, sorted by start, each starting before it stops
      and stopping before the next one starts. Its columns are [by], then [s]
      and [e], at the type where the inputs' columns meet. Every period of a
      normal form has positive length, though {!Join.overlaps} matches a
      zero-length period as its point. {e Impulses}, events of zero duration,
      are matched against a set by a join on {!Join.overlaps}, as
      [events |> Query.join ~on:Join.(keys by && overlaps (t, t) (s, e)) set]
      (without [by], the {!Join.overlaps} alone), or by the {!Join.Anti} join
      on that condition for the impulses outside the set. *)

  (** {2:sweep Sweep}

      Each function is one sweep over the starts and stops of the periods of
      [q] and [rest], which counts the periods of each that hold a value, [a]
      and [b], and keeps the values where [holds a b] is [true]. Inside
      [Expr.( … )], where [at], [a], [b] and [held] are names that neither
      [by] nor the period has and a name in an expression reads its column,
      it is:
      {[
      let events q (wa, wb) =
        let start = if_ (s <= e) s e and stop = if_ (s <= e) e s in
        let edge x (wa, wb) =
          Query.select
            [ keep (Sel.names by); at := x; a := int wa; b := int wb ]
            (Query.filter (start < stop) q)
        in
        edge start (wa, wb) |> Query.append (edge stop (-wa, -wb))
      in
      let running c = over ~by (window (sum c)) in
      events q (1, 0)
      |> Query.append (events rest (0, 1))
      |> Query.sort (List.map Order.asc (by @ [ at ]))
      |> Query.aggregate ~by:(by @ [ at ]) [ a := sum a; b := sum b ]
      |> Query.derive [ held := holds (running a) (running b) ]
      |> Query.filter (held <> coalesce [ over ~by (shift 1 held); bool false ])
      |> Query.derive [ e := over ~by (shift (-1) at) ]
      |> Query.filter held
      |> Query.select [ keep (Sel.names by); s := at; keep (Sel.names [ e ]) ]
      ]}
      It sorts the starts and stops once.

      Each raises [Invalid_argument] if [by] names a column twice. *)

  val merge_intervals : ?by:string list -> string * string -> Query.t -> Query.t
  (** [merge_intervals ~by (s, e) q] is [q]'s sets in normal form
      ({!section-normal_form}), so periods that overlap merge, and so do
      periods that touch, where one's stop is the next one's start. It is the
      sweep ({!section-sweep}) without [rest] and with [holds a _ = a > 0].
      The union of two sets is the merge of their [Query.append]. *)

  val intersect_intervals :
    ?by:string list -> string * string -> Query.t -> Query.t -> Query.t
  (** [intersect_intervals ~by (s, e) rest q] is the values that both [q]'s
      and [rest]'s set of a key hold, written
      [q |> Kit.intersect_intervals (s, e) rest], in normal form
      ({!section-normal_form}). It is the sweep ({!section-sweep}) with
      [holds a b = a > 0 && b > 0]. *)

  val subtract_intervals :
    ?by:string list -> string * string -> Query.t -> Query.t -> Query.t
  (** [subtract_intervals ~by (s, e) rest q] is the values that [q]'s set of a
      key holds and [rest]'s does not, written
      [q |> Kit.subtract_intervals (s, e) rest], in normal form
      ({!section-normal_form}). It is the sweep ({!section-sweep}) with
      [holds a b = a > 0 && b = 0]. *)

  (** {1:random Randomness}

      These derive their rows from {!Expr.hash} with a seed, so a plan returns
      the same rows on every run, whatever its batches and the number of
      cores. Without [~seed] one is drawn from {!Nx.Rng}'s scope when the plan
      is built, so a plan value keeps it and a rebuilt plan draws another. A
      split that must repeat across programs gives a [~seed].

      {!folds} balances the number of groups, and rows follow group sizes. It
      depends on the set of groups, so a changed set moves some groups between
      folds. A split that must hold as the table grows filters on a group's
      hash alone, as in
      [hash ~seed (record [ keep (Sel.names by) ]) mod int m < int j]. *)

  val shuffle : ?seed:int -> Query.t -> Query.t
  (** [shuffle ~seed q] is [q]'s rows in the order of the hash of each row's
      place in [q], counted across its batches:
      {[
      Query.derive Expr.[ key := hash ~seed (window rows) ] q
      |> Query.sort [ Order.asc key ]
      |> drop (Sel.names [ key ])
      ]}
      It reads no value, so equal rows take independent places, and it blocks.
  *)

  val sample :
    ?seed:int -> [ `Rows of int | `Fraction of float ] -> Query.t -> Query.t
  (** [sample ~seed size q] draws rows of [q] without replacement. [`Rows n]
      is the first [n] rows of {!shuffle}'s order, or all of them where [q] has
      fewer. [`Fraction f] keeps, in [q]'s order, the rows whose place hashes
      below [f * 2{^ 62}], so at one seed a smaller fraction keeps a subset of
      a larger one:
      {[
      Query.filter Expr.(hash ~seed (window rows) < int cutoff) q
      ]}
      It blocks. Bootstrapping with replacement is {!Talon.take} of
      {!Nx.Rng.randint} indices.

      Raises [Invalid_argument] if [n] is negative or [f] is not in
      \[[0];[1]\]. *)

  val folds : ?seed:int -> by:string list -> int -> Query.t -> Query.t
  (** [folds ~seed ~by k q] is [q] with a last column ["fold"], an [int64] in
      \[[0];[k - 1]\], that gives every row of a group of [by] one fold, by
      key identity (null is one group and NaN one). Groups rank by the hash of
      the record of [by]'s columns, in that order, and the [r]th of [g] groups
      takes fold [⌊k (r - 1) / g⌋], so fold sizes in groups differ by at most
      one. It blocks, and reads every key before the first fold is known.

      Raises [Invalid_argument] if [by] is empty or names a column twice, if
      [k] is below [1] or if [q] already has a column ["fold"]; a run raises it
      if [q] has fewer than [k] groups. *)

  (** {1:ranges Time ranges} *)

  val time_range :
    'a Type.t -> start:'a -> stop:'a -> Time.step -> string -> t
  (** [time_range ty ~start ~stop step name] is the table of one column [name]
      of [ty], the values [offset (k * step) start] below [stop], in order, for
      [k] from [0], so that adjacent ranges meet at [stop]. A day past a month's
      end becomes that month's last day, and each step counts from [start], so
      2024-01-31 by one month gives 2024-02-29, then 2024-03-31. It is empty
      when [start >= stop].

      It is a table where the rest of [Kit] is a plan, since the [k]th value
      scales a step by [k], which no verb computes.

      Raises [Invalid_argument] if [step] is not positive, if
      {!Expr.Temporal.offset} does not move [ty] by [step], or if the range
      holds [2{^ 62}] steps or more. *)

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
           (fun (c, ws) ->
             Expr.(c := cast (Type.categorical ws) (Col.string c)))
           dictionaries)
        (* the [(c, words c)] of each [c] *)
        q
      ]}
      [cs] empty is [Ok q]. A column of [q] that is missing or is not text is a
      problem of the [select]. A categorical column takes the dictionary of the
      values it holds, in byte order, since it is read as text.

      [Error e] if a run fails, or if a column holds more than 2{^ 31} - 1
      distinct strings, the most a dictionary holds.

      Raises [Invalid_argument] if [cs] names a column twice. *)
end

(**/**)

(* What Talon_parquet and Talon_arrow share about decimal columns. *)

module Decimal : sig
  (** Decimal columns of files.

      A decimal is an integer, its {e unscaled} value, and a scale: the number
      [unscaled × 10{^-scale}]. Talon has no decimal type, and a file format
      that stores decimals has its reader declare, for each column, one of two
      types that hold its values ({!section-decimals}). This module is the one
      place that states which, and how a file's bytes read as them. *)

  type bytes = (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t
  (** The type for the bytes a decimal is read from. *)

  val reads_as : precision:int -> Type.any -> bool
  (** [reads_as ~precision t] is [true] iff a decimal of at most [precision] digits
      reads as [t]: [float64], and, up to 18 digits, [int64]. *)

  val scales_agree : Type.any -> format:int option -> file:int option -> bool
  (** [scales_agree t ~format ~file] is [false] iff a leaf declared [t] by a
      format whose field has decimal scale [format], read from a file whose
      field has decimal scale [file], [None] for a field that is not a decimal,
      would read integers that mean other numbers: [t] is [int64] and the two
      differ. A [float64] applies the file's own scale. *)

  val other_scale : string
  (** [other_scale] words the failure {!scales_agree} detects, after a column's
      name. *)

  val reads_text : precision:int -> string
  (** [reads_text ~precision] states the types a decimal of [precision] digits
      reads as, for error messages: [float64 or int64], or [float64] past 18
      digits. *)

  val undeclared : fn:string -> column:string -> text:string -> flat:bool -> 'a
  (** [undeclared ~fn ~column ~text ~flat] raises [Invalid_argument] with the
      message for a decimal column left undeclared: [fn] is the function that
      needs its type, [column] its name and [text] its file type, which is a
      decimal when [flat] and holds one otherwise. *)

  val unscaled :
    bytes ->
    first:int ->
    len:int ->
    big:bool ->
    (int64, Bigarray.int64_elt, Bigarray.c_layout) Bigarray.Array1.t ->
    int ->
    bool
  (** [unscaled b ~first ~len ~big out i] writes to [out.{i}] the two's complement
      integer of the [len] bytes of [b] from [first], big-endian if [big] and
      little-endian otherwise, and is [true], or is [false] if it is outside
      [int64]. [len] is positive. *)

  val digits : bytes -> first:int -> len:int -> big:bool -> string
  (** [digits b ~first ~len ~big] is the decimal text of the two's complement
      integer of the [len] bytes of [b] from [first], big-endian if [big] and
      little-endian otherwise, of any length. [len] is positive. *)

  val floats :
    int -> scale:int -> (int -> string) -> (Nx.host Nx.float64_t, string) result
  (** [floats n ~scale digits] is the [n] decimals of unscaled values [digits i]
      (decimal text) and scale [scale] as [float64]: the float nearest each value,
      ties to even, or an infinity past [float64]'s range. A value the text does
      not parse is [Error why]. *)
end

(* For talon's tests. *)

val duration_sum : int64 array -> copies:int -> int64 option
(** [duration_sum xs ~copies] is the sum of [copies] copies of each tick of
    [xs], as a [duration[ns]] {!Expr.sum} computes it, or [None] if it leaves
    [int64]: a frame of [copies * Array.length xs] values, held as [xs]. *)
