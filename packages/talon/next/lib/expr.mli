(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Expressions: typed computations over the columns of a frame.

    [Talon_next.Expr] documents the expressions that users write. This interface
    adds their {{!repr}representation}, their {{!binding}binding} to a schema
    and the {{!analyses}analyses} that serve the verbs, the optimizer and the
    evaluator. *)

(** {1:exprs Expressions} *)

type row
type agg
type ('a, +'s) t
type +'s out

val ( := ) : string -> ('a, 's) t -> 's out
val keep : Sel.t -> row out
val across : 'a Kind.t -> Sel.t -> (string -> ('a, row) t -> 's out) -> 's out

type 's column = { column : 'a. string -> ('a, row) t -> 's out }

val each : Sel.t -> 's column -> 's out
val int : int -> (int, 's) t
val float : float -> (float, 's) t
val bool : bool -> (bool, 's) t
val string : string -> (string, 's) t
val instant : Time.instant -> (Time.instant, 's) t
val span : Time.span -> (Time.span, 's) t
val date : Time.date -> (Time.date, 's) t
val null : ('a, 's) t
val ( + ) : (int, 's) t -> (int, 's) t -> (int, 's) t
val ( - ) : (int, 's) t -> (int, 's) t -> (int, 's) t
val ( * ) : (int, 's) t -> (int, 's) t -> (int, 's) t
val ( / ) : (int, 's) t -> (int, 's) t -> (int, 's) t
val ( mod ) : (int, 's) t -> (int, 's) t -> (int, 's) t
val ( +. ) : (float, 's) t -> (float, 's) t -> (float, 's) t
val ( -. ) : (float, 's) t -> (float, 's) t -> (float, 's) t
val ( *. ) : (float, 's) t -> (float, 's) t -> (float, 's) t
val ( /. ) : (float, 's) t -> (float, 's) t -> (float, 's) t
val ( ** ) : (float, 's) t -> (float, 's) t -> (float, 's) t
val ( = ) : ('a, 's) t -> ('a, 's) t -> (bool, 's) t
val ( <> ) : ('a, 's) t -> ('a, 's) t -> (bool, 's) t
val ( < ) : ('a, 's) t -> ('a, 's) t -> (bool, 's) t
val ( > ) : ('a, 's) t -> ('a, 's) t -> (bool, 's) t
val ( <= ) : ('a, 's) t -> ('a, 's) t -> (bool, 's) t
val ( >= ) : ('a, 's) t -> ('a, 's) t -> (bool, 's) t
val ( && ) : (bool, 's) t -> (bool, 's) t -> (bool, 's) t
val ( || ) : (bool, 's) t -> (bool, 's) t -> (bool, 's) t
val not : (bool, 's) t -> (bool, 's) t
val if_ : (bool, 's) t -> ('a, 's) t -> ('a, 's) t -> ('a, 's) t
val is_null : ('a, 's) t -> (bool, 's) t
val coalesce : ('a, 's) t list -> ('a, 's) t
val is_in : 'a list -> ('a, 's) t -> (bool, 's) t
val cut : 'a array -> ('a, 's) t -> (int, 's) t
val cast : 'b Type.t -> ('a, 's) t -> ('b, 's) t

type fn = { f : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

val nx : fn -> ('a, 's) t -> ('a, 's) t

type fn2 = { f2 : 'a 'b. ('a, 'b) Nx.t -> ('a, 'b) Nx.t -> ('a, 'b) Nx.t }

val nx2 : fn2 -> ('a, 's) t -> ('a, 's) t -> ('a, 's) t
val rows : (int, agg) t
val count : ('a, row) t -> (int, agg) t
val sum : ('a, row) t -> ('a, agg) t
val min : ('a, row) t -> ('a, agg) t
val max : ('a, row) t -> ('a, agg) t
val first : ('a, row) t -> ('a, agg) t
val last : ('a, row) t -> ('a, agg) t
val only : ('a, row) t -> ('a, agg) t
val mean : ('a, row) t -> (float, agg) t
val std : ('a, row) t -> (float, agg) t
val var : ('a, row) t -> (float, agg) t
val median : ('a, row) t -> (float, agg) t
val quantile : float -> ('a, row) t -> (float, agg) t
val ewm : alpha:float -> ('a, row) t -> (float, agg) t
val n_unique : ('a, row) t -> (int, agg) t
val arg_min : ('a, row) t -> (int, agg) t
val arg_max : ('a, row) t -> (int, agg) t
val collect : ('a, row) t -> ('a array, agg) t
val over : ?by:string list -> ?order:Order.t list -> ('a, 's) t -> ('a, row) t
val rolling : Window.t -> ('a, agg) t -> ('a, row) t
val shift : int -> ('a, row) t -> ('a, row) t
val rank : ('a, row) t -> (int, row) t
val const : 'a -> ('a, 's) t
val ( $ ) : ('a -> 'b, 's) t -> ('a, 's) t -> ('b, 's) t
val option : ('a, 's) t -> ('a option, 's) t
val of_option : ('a option, 's) t -> ('a, 's) t
val store : 'b Type.t -> ('b, 's) t -> ('b, 's) t

val batch :
  (('a, 'b) Nx.t -> ('c, 'd) Nx.t) ->
  (('a, 'b) Nx.t, row) t ->
  (('c, 'd) Nx.t, row) t

val record : 's out list -> (Record.t, 's) t
val field : 'a Kind.t -> string -> (Record.t, 's) t -> ('a, 's) t
val unpack : (Record.t, row) t -> row out

module Str : sig
  type pattern

  val literal : string -> pattern
  val prefix : string -> pattern
  val suffix : string -> pattern
  val pieces : string list -> pattern
  val length : (string, 's) t -> (int, 's) t
  val slice : offset:int -> length:int -> (string, 's) t -> (string, 's) t
  val lower : (string, 's) t -> (string, 's) t
  val upper : (string, 's) t -> (string, 's) t
  val matches : pattern -> (string, 's) t -> (bool, 's) t
  val parse : 'a Type.t -> (string, 's) t -> ('a, 's) t
end

module Temporal : sig
  val add : ('a, 's) t -> (Time.span, 's) t -> ('a, 's) t
  val diff : ('a, 's) t -> ('a, 's) t -> (Time.span, 's) t

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

  val field : field -> ?zone:Tz.zone -> ('a, 's) t -> (int, 's) t
  val floor : ?zone:Tz.zone -> Time.step -> ('a, 's) t -> ('a, 's) t
  val offset : ?zone:Tz.zone -> Time.step -> ('a, 's) t -> ('a, 's) t

  type policy = [ `Earlier | `Later | `Null | `Fail ]

  val localize :
    Tz.zone ->
    ambiguous:policy ->
    gap:policy ->
    (Time.instant, 's) t ->
    (Time.instant, 's) t

  val windows :
    ?zone:Tz.zone ->
    every:Time.step ->
    period:Time.step ->
    ('a, 's) t ->
    ('a array, 's) t

  val parse : string -> 'a Type.t -> (string, 's) t -> ('a, 's) t
  val format : string -> ('a, 's) t -> (string, 's) t
end

val pp : Format.formatter -> ('a, 's) t -> unit
(** [pp] formats an expression as [Talon_next.Expr.pp] says. Besides, a bound
    expression formats as written: its typings format as nothing, {!cut} edges
    format sorted, and a traced lift formats as the nx operations it records, by
    their names: [exp x]. *)

val pp_arg : Format.formatter -> ('a, 's) t -> unit
(** [pp_arg ppf e] formats [e] as a function's argument: as {!pp} does, in
    parentheses unless it is atomic: a handle, a literal that is not negative,
    {!null} or {!rows}. [filter (dep_delay > 15.)] prints its predicate so. *)

val pp_out : Format.formatter -> 's out -> unit
(** [pp_out ppf o] formats [o] as written: ["z" := …], [keep (prefix "wk")],
    [across float (prefix "wk") <fn>], [each all <fn>]. *)

(** {1:repr Representation}

    An unbound expression is a tree, as the constructors build it, with an
    identity of its own: nothing compares unbound expressions. Bound expressions
    are hash-consed: two live bound expressions are {!same} iff they are
    structurally equal, so comparing them is O(1). Literal values compare by
    value, floats by their bits. An {!is_in} value list and {!cut} edges compare
    by {!Type.compare_value} when their operand has a column typing, and
    physically otherwise. Every other OCaml value inside a node (functions,
    {!const} values, extension declarations, zones) compares physically. The
    table is weak and safe to use from any domain. *)

(** The type for arithmetic: [Mod] applies to [int]s only, and [Pow] to [float]s
    only. *)
type arith = Add | Sub | Mul | Div | Mod | Pow

type compare = [ `Eq | `Ne | `Lt | `Le | `Gt | `Ge ]
(** The type for comparisons: [`Lt] is [<], and so on. *)

type logic = And | Or

(** The type for reductions from values ['a] to a value ['b]. *)
type (_, _) reduction =
  | Count : ('a, int) reduction
  | Sum : ('a, 'a) reduction
  | Min : ('a, 'a) reduction
  | Max : ('a, 'a) reduction
  | First : ('a, 'a) reduction
  | Last : ('a, 'a) reduction
  | Only : ('a, 'a) reduction
  | Mean : ('a, float) reduction
  | Std : ('a, float) reduction
  | Var : ('a, float) reduction
  | Median : ('a, float) reduction
  | Quantile : float -> ('a, float) reduction
  | Ewm : float -> ('a, float) reduction
  | N_unique : ('a, int) reduction
  | Arg_min : ('a, int) reduction
  | Arg_max : ('a, int) reduction
  | Collect : ('a, 'a array) reduction

type ('e, 's) ext = {
  type_ : Type.ext Type.t;  (** The extension type, an [Ext] type. *)
  storage : 's Type.t;  (** Its storage, the [storage] of [type_]. *)
  ordered : bool;  (** [true] iff storage order is value order. *)
  dec : 's -> 'e;
  enc : 'e -> 's;
}
(** The type for extension declarations, made by [Ext.v]. *)

(** The type for the typings of bound expressions of values ['a]. *)
type 'a typing =
  | Column : 'a Type.t -> 'a typing  (** Stored as this type. *)
  | Extension : ('a, 's) ext -> 'a typing
      (** Stored as the declaration's extension type. *)
  | Value : 'a typing
      (** OCaml values without a column type: a [const], a [$] result or an
          {!option}, which only {!( $ )} and [Query.values] take. *)

type packed = Packed : ('a, 's) t -> packed  (** An expression of any type. *)

(** The type for the nodes of expressions of values ['a]. Shapes are erased: a
    node's operands have any shape. *)
type 'a node =
  | Handle : 'a Kind.t * string -> 'a node  (** A handle. *)
  | Ext_handle : ('a, 's) ext * string -> 'a node  (** An extension handle. *)
  | Read : 'a Type.t * string -> 'a node
      (** A column read at its own type, by {!keep} and {!each}. *)
  | Lit : 'a Kind.t * 'a -> 'a node  (** A literal of a kind. *)
  | Null : 'a node
  | Rows : int node
  | Int : arith * (int, 's0) t * (int, 's1) t -> int node
  | Float : arith * (float, 's0) t * (float, 's1) t -> float node
  | Compare : compare * ('a, 's0) t * ('a, 's1) t -> bool node
  | Logic : logic * (bool, 's0) t * (bool, 's1) t -> bool node
  | Not : (bool, 's) t -> bool node
  | If : (bool, 's0) t * ('a, 's1) t * ('a, 's2) t -> 'a node
  | Is_null : ('a, 's) t -> bool node
  | Coalesce : ('a, 's) t list -> 'a node
  | Is_in : 'a list * ('a, 's) t -> bool node
  | Cut : 'a array * ('a, 's) t -> int node
      (** Edges as given, never mutated; sorted and distinct once bound at a
          column type. *)
  | Cast : 'a Type.t * ('b, 's) t -> 'a node
  | Lift : fn * ('a, 's) t -> 'a node
  | Lift2 : fn2 * ('a, 's0) t * ('a, 's1) t -> 'a node
  | Nx_unary : Nx_backend.unary * ('a, 's) t -> 'a node
      (** An elementwise nx operation that a lift performs, with nx's semantics,
          as are the four below. *)
  | Nx_binary : Nx_backend.binary * ('a, 's0) t * ('a, 's1) t -> 'a node
  | Nx_compare : Nx_backend.compare * ('a, 's0) t * ('a, 's1) t -> bool node
  | Nx_where : (bool, 's0) t * ('a, 's1) t * ('a, 's2) t -> 'a node
  | Nx_cast : 'a Type.t * ('b, 's) t -> 'a node
  | Reduce : ('a, 'b) reduction * ('a, 's) t -> 'b node
  | Over : { by : string list; order : Order.t list; e : ('a, 's) t } -> 'a node
  | Rolling : Window.t * ('a, 's) t -> 'a node
  | Shift : int * ('a, 's) t -> 'a node
  | Rank : ('a, 's) t -> int node
  | Const : 'a -> 'a node
  | App : ('a -> 'b, 's0) t * ('a, 's1) t -> 'b node
  | Option : ('a, 's) t -> 'a option node
  | Of_option : ('a option, 's) t -> 'a node
  | Store : 'a Type.t * ('a, 's) t -> 'a node
  | Batch :
      (('a, 'b) Nx.t -> ('c, 'd) Nx.t) * (('a, 'b) Nx.t, 's) t
      -> ('c, 'd) Nx.t node
  | Record_outs : 's out list -> Record.t node  (** A record as written. *)
  | Fields : (string * packed) list -> Record.t node  (** A bound record. *)
  | Field : 'a Kind.t * string * (Record.t, 's) t -> 'a node
  | Storage : ('e, 'a) ext * ('e, 's) t -> 'a node
  | Wrap : ('a, 'st) ext * ('st, 's) t -> 'a node
  | Text : 'a text_op * (string, 's) t -> 'a node
  | Calendar : 'a calendar_op -> 'a node

(** Text operations, each applied to the text operand of {!Text}. *)
and 'a text_op =
  | Length : int text_op
  | Slice : { offset : int; length : int } -> string text_op
  | Lower : string text_op
  | Upper : string text_op
  | Matches : Str.pattern -> bool text_op
  | Parse : 'a Type.t -> 'a text_op

(** Temporal operations. *)
and 'a calendar_op =
  | Add_span : ('a, 's0) t * (Time.span, 's1) t -> 'a calendar_op
  | Diff : ('a, 's0) t * ('a, 's1) t -> Time.span calendar_op
  | Part : Temporal.field * Tz.zone option * ('a, 's) t -> int calendar_op
  | Floor : Tz.zone option * Time.step * ('a, 's) t -> 'a calendar_op
  | Offset : Tz.zone option * Time.step * ('a, 's) t -> 'a calendar_op
  | Localize : {
      zone : Tz.zone;
      ambiguous : Temporal.policy;
      gap : Temporal.policy;
      a : (Time.instant, 's) t;
    }
      -> Time.instant calendar_op
  | Windows : {
      zone : Tz.zone option;
      every : Time.step;
      period : Time.step;
      a : ('a, 's) t;
    }
      -> 'a array calendar_op
  | Parse_with : string * 'a Type.t * (string, 's) t -> 'a calendar_op
  | Format_with : string * ('a, 's) t -> string calendar_op

val node : ('a, 's) t -> 'a node
(** [node e] is [e]'s node. *)

val make : 'a node -> ('a, 's) t
(** [make n] is the unbound expression of node [n]. *)

val typed : 'a typing -> 'a node -> ('a, 's) t
(** [typed t n] is the bound expression of node [n] and typing [t], with the
    identity of every live bound expression structurally equal to it. It checks
    nothing: the caller states [n]'s shape and keeps the invariants of
    {{!binding}binding}. Binding, the lift tracer and the optimizer build bound
    expressions with it. *)

val same : ('a, 's0) t -> ('b, 's1) t -> bool
(** [same e0 e1] is [true] iff [e0] and [e1] are one expression: for bound
    expressions, iff they are structurally equal. *)

(** {1:binding Binding}

    A verb binds each expression to the schema of its frame. Binding checks the
    expression, infers the type of each of its values, and rewrites it into a
    {e bound} expression, every node of which has a {!typing} that follows from
    its operands' alone:
    - each record is {!Fields}, its outputs named and bound;
    - {!cut}'s edges are sorted and distinct over a column type;
    - each {!Lift} and {!Lift2} is replaced by the operations it records: [f]
      runs once under [Nx.Op.intercept] on a traced value of the operand's
      dtype, and the constants it creates become literals. An [f] that performs
      any other operation on its argument, or computes in a dtype that talon has
      no type for, is a problem. *)

val bind_predicate :
  Schema.t -> (bool, row) t -> ((bool, row) t, Problem.t list) result
(** [bind_predicate s p] is [Ok b] with [b] the bound form of the unbound
    predicate [p] over a frame of the columns [s], at the type [bool], or
    [Error ps] with every problem in [p], each once, in the order of first
    appearance. The problems are:
    - a missing column, and a handle whose kind does not bind its column;
    - operands whose types do not meet, and an operation that a type does not
      support, such as [sum] of a string;
    - a literal, a {!const} value, an {!is_in} value, a {!cut} edge, or the
      value of an integer operation of literals, that its type does not hold;
    - a [null] that nothing types; an {!option} result, and a [const] or [$]
      result that nothing types, anywhere but as an argument of {!( $ )}; and an
      operand of {!cast}, {!option}, a reduction or a frame that has no column
      type;
    - an order-dependent operation on an extension type whose declaration is not
      ordered, or that is read without its declaration;
    - a lift outside its rules, and one whose function ignores an argument;
    - a {!batch} function whose result on an empty batch is not an empty batch
      of cells.

    An OCaml value takes the type [bool], and an extension's values are a
    problem. An exception that a function of [p] raises when binding applies it
    propagates.

    Raises [Invalid_argument] if [p] holds a bound expression. *)

val typing : ('a, 's) t -> 'a typing
(** [typing b] is the typing of the bound expression [b].

    Raises [Invalid_argument] if [b] is not bound. *)

val read : 'a Type.t -> string -> ('a, 's) t
(** [read ty n] is the bound expression that reads the column [n] of type [ty],
    as {!keep} binds it. *)

val conj : (bool, 's) t -> (bool, 's) t -> (bool, 's) t
(** [conj a b] is the bound [a && b] of the bound predicates [a] and [b]. *)

val bind_out :
  Schema.t -> 's out -> ((string * packed) list, Problem.t list) result
(** [bind_out s o] is [Ok outs] with [outs] the named, bound outputs of [o] over
    a frame of the columns [s], in order, each with a {!Column} or {!Extension}
    typing, or [Error ps] with every problem in [o]: those of {!bind_predicate},
    the missing names of its selectors, a column of {!across} that its kind does
    not bind, and an output without a column type. Collisions between the
    outputs of one verb are the verb's to report. *)

val out_name : 's out -> string option
(** [out_name o] is [Some n] if [o] is [n := e], and [None] otherwise: the name
    an output has as written, before binding. *)

(** {1:analyses Bound expressions}

    These functions take bound expressions and serve the optimizer: each raises
    [Invalid_argument] on an expression that is not bound. *)

val reads : ('a, 's) t -> string list
(** [reads b] is the columns that [b] reads, distinct, in order of first
    appearance: the columns of its handles and reads, of {!over}'s [~by] and
    [~order] keys, and of its time windows' keys. *)

val row_local : ('a, 's) t -> bool
(** [row_local b] is [true] iff [b]'s value at a row depends on that row alone:
    [b] has no {!over}, {!rolling}, {!shift} or {!rank}. A filter, a slice or a
    reordering of the frame therefore does not change the values of a row-local
    expression on the rows it keeps. *)

val can_fail : ('a, 's) t -> bool
(** [can_fail b] is [true] iff evaluating the {{!row_local}row-local} [b] can
    fail a run or call a user function: [b] holds a {!cast} that does not widen
    its operand to a type that contains it, {!Str.parse}, a {!Temporal}
    operation other than {!Temporal.field} and {!Temporal.format}, {!of_option},
    {!( $ )} or {!batch}. *)

val rename : (string -> string) -> ('a, 's) t -> ('a, 's) t
(** [rename f b] is [b] reading the column [f n] wherever it reads the column
    [n]: handles, reads, {!over}'s keys and time windows' keys. Typings are
    kept, so [f] maps each column that [b] reads to one of the same type. *)

val fold_constants : ('a, 's) t -> ('a, 's) t
(** [fold_constants b] is [b] with each operation of literals replaced by the
    literal it computes, with the same typing, where that literal is exactly the
    value that evaluating the operation gives:
    - [+], [-], [*], [/] and [mod] of integers whose result the type holds
      without wrapping, a division by zero being null;
    - [+.], [-.], [*.] and [/.] of [float32] and [float64] values, rounded once
      to their type, unless the result is NaN, whose bits nx's kernels define
      (as they define [**] and nx lifts, which are kept);
    - comparisons, by talon's total order at their operands' common type;
    - [&&], [||] and [not], [is_null], [if_] and [coalesce];
    - {!store} of a literal of its type.

    A literal is read as the value it has at its type ({!Type.value}), so an
    operation of [float16] literals, arithmetic or comparison, is kept.

    Besides, with one operand a literal: [a && false] and [a || true] become the
    literal, [a && true] and [a || false] become [a], [if_] with a literal
    condition becomes its branch, and [coalesce] drops its null literals and
    ends at its first literal that is not null. A branch or an operand replaces
    its operation only where it has the operation's typing. Each rewrite gives
    the same value on every row, null included. *)
