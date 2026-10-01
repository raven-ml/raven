(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The reference interpreter: plans run one row at a time.

    A plan is written in the small language below, which {!query} translates to
    the verbs and {!run} evaluates one row at a time over decoded [option]
    values, comparing them with {!Talon_next.Type.compare_value}. The operands
    of an operation meet at their common type ({!Talon_next.Type.common}), and a
    literal or [Null] stands only beside an operand that reads a column, so that
    it takes that operand's type. Extension columns are read as their storage.
*)

open Talon_next

type iop = Add | Sub | Mul | Div | Mod
type fop = Fadd | Fsub | Fmul | Fdiv
type cmp = [ `Eq | `Ne | `Lt | `Le | `Gt | `Ge ]

(** The type for expressions of values ['a]. *)
type 'a expr =
  | Col : 'a Type.t * string -> 'a expr
  | Lit : 'a Type.t * 'a -> 'a expr  (** A literal of the type it meets. *)
  | Null : 'a Type.t -> 'a expr
  | Int : iop * int expr * int expr -> int expr
      (** Meeting at [int8] to [int32], [uint8] to [uint32]. *)
  | Float : fop * float expr * float expr -> float expr
      (** Meeting at [float32] and [float64]. *)
  | Cmp : cmp * 'a expr * 'a expr -> bool expr
  | And : bool expr * bool expr -> bool expr
  | Or : bool expr * bool expr -> bool expr
  | Not : bool expr -> bool expr
  | If : bool expr * 'a expr * 'a expr -> 'a expr
  | Is_null : 'a expr -> bool expr
  | Coalesce : 'a expr list -> 'a expr
  | Is_in : 'a list * 'a expr -> bool expr
  | Store : 'a Type.t * 'a expr -> 'a expr
      (** At a type that contains the operand's. *)

(** The type for outputs. *)
type out = Out : string * 'a expr -> out | Keep of string list

(** The type for plans. *)
type plan =
  | Table of Talon_next.t
  | Select of out list * plan
  | Derive of out list * plan
  | Filter of bool expr * plan
  | Slice of { offset : int; length : int; plan : plan }
  | Append of plan * plan  (** [Append (q, rest)] is [q |> append rest]. *)

val literal : 'a Type.t -> ('a -> ('a, 's) Expr.t) option
(** [literal ty] makes the literals of [ty], if [Expr] writes them. *)

val type_of : 'a expr -> 'a Type.t
(** [type_of e] is the type of [e]'s values. *)

val expr : 'a expr -> ('a, Expr.row) Expr.t
(** [expr e] is [e] in [Expr]. *)

val query : plan -> Query.t
(** [query p] is [p] built with the verbs. *)

val schema : plan -> (string * Type.any) list
(** [schema p] is the columns of [p]'s rows. *)

(** The type for a column of values. *)
type column = Column : 'a Type.t * 'a option array -> column

val decode : Column.t -> column
(** [decode c] is [c]'s values, an extension's as its storage's. *)

val run : plan -> (string * column) list
(** [run p] is [p]'s rows, column by column, as {!decode} reads them. *)

val values : 'a expr -> plan -> ('a array, int) result
(** [values e p] is [e] on each of [p]'s rows, or [Error r] for the first row
    [r] where [e] is null. *)
