(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The reference interpreter: plans run one row at a time.

    A plan is written in the small language below, which {!query} translates to
    the verbs and {!run} evaluates over decoded [option] values, comparing them
    with {!Talon_next.Type.compare_value}. The operands of an operation meet at
    their common type ({!Talon_next.Type.common}), and a literal or [Null]
    stands only beside an operand that reads a column, so that it takes that
    operand's type. Extension columns are read as their storage.

    {b Frames.} A step evaluates its expressions over a frame of rows, node by
    node: its outputs in order, and an expression every operand, branches
    included, before its node, from left to right, each over every row of the
    frame. A reduction is one value for its frame, which a row expression above
    it repeats on each of the frame's rows. A step whose expressions are local
    to their row evaluates over one row at a time; any other step, and
    [aggregate] over each group, over all its input's rows.

    {b Failures.} Rows flow through the plan, each step pulling the rows it
    needs from its input: a slice from the start pulls its input's first
    [offset + length] rows, none when [length] is [0], and a slice from the end,
    a {!Sort} and a step that is not local all of them. A {!Sort} orders its
    rows stably, each key by {!Talon_next.Type.compare_value} in its direction,
    nulls last unless first. A frame fails at the earliest row where a node
    fails, and of two failures at a row, at the first evaluated: a value that a
    {!Map}, {!Bind} or {!Cast} type does not hold, a text that {!Parse} does not
    read, or an exception that a function raises, at its row of the step's
    input; [Only] over two values at the frame's first row. A failed value is
    null. A step emits its rows before a frame that fails, and the run ends with
    the failure: an error at the row, or the exception. *)

open Talon_next

type iop = Add | Sub | Mul | Div | Mod
type fop = Fadd | Fsub | Fmul | Fdiv
type cmp = [ `Eq | `Ne | `Lt | `Le | `Gt | `Ge ]

(** The type for reductions of values ['a] to a value ['b]. [Sum] and [Mean]
    take integers of at most 32 bits, which they compute exactly. *)
type (_, _) reduction =
  | Count : ('a, int) reduction
  | Sum : (int, int) reduction
  | Mean : (int, float) reduction
  | Min : ('a, 'a) reduction
  | Max : ('a, 'a) reduction
  | First : ('a, 'a) reduction
  | Last : ('a, 'a) reduction
  | Only : ('a, 'a) reduction
  | Median : ('a, float) reduction  (** Of integers or floats. *)
  | Quantile : float -> ('a, float) reduction  (** Of integers or floats. *)
  | N_unique : ('a, int) reduction
  | Arg_min : ('a, int) reduction
  | Arg_max : ('a, int) reduction

type key = { name : string; desc : bool; nulls_first : bool }
(** The type for the keys of {!Over}'s order. *)

(** The type for expressions of values ['a] and shape ['s]. *)
type ('a, 's) term =
  | Col : 'a Type.t * string -> ('a, Expr.row) term
  | Lit : 'a Type.t * 'a -> ('a, 's) term
      (** A literal of the type it meets. *)
  | Null : 'a Type.t -> ('a, 's) term
  | Int : iop * (int, 's) term * (int, 's) term -> (int, 's) term
      (** Meeting at [int8] to [int32], [uint8] to [uint32]. *)
  | Float : fop * (float, 's) term * (float, 's) term -> (float, 's) term
      (** Meeting at [float32] and [float64]. *)
  | Cmp : cmp * ('a, 's) term * ('a, 's) term -> (bool, 's) term
  | And : (bool, 's) term * (bool, 's) term -> (bool, 's) term
  | Or : (bool, 's) term * (bool, 's) term -> (bool, 's) term
  | Not : (bool, 's) term -> (bool, 's) term
  | If : (bool, 's) term * ('a, 's) term * ('a, 's) term -> ('a, 's) term
  | Is_null : ('a, 's) term -> (bool, 's) term
  | Coalesce : ('a, 's) term list -> ('a, 's) term
  | Is_in : 'a list * ('a, 's) term -> (bool, 's) term
  | Store : 'a Type.t * ('a, 's) term -> ('a, 's) term
      (** At a type that contains the operand's. *)
  | Map : int Type.t * ('a -> int) * ('a, 's) term -> (int, 's) term
      (** [Map (ty, f, a)] is [store ty (const f $ a)], at a type of OCaml
          [int]s. [a] reads a column. *)
  | Bind :
      int Type.t * ('a option -> int option) * ('a, 's) term
      -> (int, 's) term
      (** [Bind (ty, f, a)] is [store ty (of_option (const f $ option a))]. *)
  | Cast : int Type.t * (int, 's) term -> (int, 's) term
      (** [Cast (ty, a)] is [cast ty a] between integer types: [a]'s value,
          where [ty] holds it. *)
  | Length : (string, 's) term -> (int, 's) term
      (** [Str.length], an [int64]. *)
  | Substring : int * int * (string, 's) term -> (string, 's) term
      (** [Substring (offset, length, a)] is [Str.slice ~offset ~length a]. *)
  | Parse : int Type.t * (string, 's) term -> (int, 's) term
      (** [Parse (ty, a)] is [Str.parse ty a] at an integer type: a sign and
          decimal digits, whose value [ty] holds. *)
  | Field : Expr.Temporal.field * (Time.date, 's) term -> (int, 's) term
      (** [Field (f, a)] is [Temporal.field f a] of a date, an [int64]: [`Year],
          [`Month], [`Day] or [`Yearday]. *)
  | Rows : (int, Expr.agg) term
  | Reduce : ('a, 'b) reduction * ('a, Expr.row) term -> ('b, Expr.agg) term
  | Over : string list * key list * ('a, 's) term -> ('a, Expr.row) term
  | Shift : int * ('a, Expr.row) term -> ('a, Expr.row) term
  | Rank : ('a, Expr.row) term -> (int, Expr.row) term

type 'a expr = ('a, Expr.row) term
(** The type for row expressions. *)

(** The type for outputs. *)
type 's out =
  | Out : string * ('a, 's) term -> 's out
  | Keep : string list -> Expr.row out

(** The type for join conditions. *)
type on =
  | Keys of (string * string) list
      (** [Keys [(l, r); …]] is [eq l r && …], at least one atom. *)
  | Position
  | All

(** The type for plans. *)
type plan =
  | Table of Talon_next.t
  | Select of Expr.row out list * plan
  | Derive of Expr.row out list * plan
  | Filter of bool expr * plan
  | Slice of { offset : int; length : int; plan : plan }
  | Append of plan * plan  (** [Append (q, rest)] is [q |> append rest]. *)
  | Aggregate of string list * Expr.agg out list * plan
  | Join of {
      kind : Join.kind;
      each_left : Join.count;
      each_right : Join.count;
      on : on;
      left : plan;
      right : plan;
    }
      (** A nested loop over the left's rows, then the right's. An equality join
          reads all its left's rows, then all its right's; a join on [All] all
          its right's, then its left's as it needs them. The assertions check
          the left's rows in order, then the right's: a failure is at the first
          row of that side whose number of matches the count does not allow, its
          reason naming its keys as {!key_text} writes them. *)
  | Sort of key list * plan
  | Source of Source.t * Talon_next.t
      (** [Source (s, t)] reads [s], whose rows are [t]'s. *)

val key_text : 'a Type.t -> ('a option -> string) option
(** [key_text ty] writes values of [ty] as the run's messages write a key, [∅]
    for [None], or is [None] if the reference cannot write them: for the types
    that [Expr] writes no literal of, and instants. *)

val literal : 'a Type.t -> ('a -> ('a, 's) Expr.t) option
(** [literal ty] makes the literals of [ty], if [Expr] writes them. *)

val type_of : ('a, 's) term -> 'a Type.t
(** [type_of e] is the type of [e]'s values. *)

val expr : ('a, 's) term -> ('a, 's) Expr.t
(** [expr e] is [e] in [Expr]. *)

val query : plan -> Query.t
(** [query p] is [p] built with the verbs. *)

val schema : plan -> (string * Type.any) list
(** [schema p] is the columns of [p]'s rows. *)

(** The type for a column of values. *)
type column = Column : 'a Type.t * 'a option array -> column

val decode : Column.t -> column
(** [decode c] is [c]'s values, an extension's as its storage's. *)

val run : plan -> ((string * column) list, int * string) result
(** [run p] is [p]'s rows, column by column, as {!decode} reads them, or
    [Error (row, reason)] where it fails at the row [row] of a step's input,
    [reason] as the run says it. *)

val values : 'a expr -> plan -> ('a array, int * string) result
(** [values e p] is [e] on each of [p]'s rows, or [Error (row, reason)] where
    [p] fails, or where [e] fails or is null at [p]'s row [row]. *)
