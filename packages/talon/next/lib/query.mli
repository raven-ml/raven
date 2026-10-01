(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Queries: descriptions of tables to compute.

    [Talon_next.Query] documents the verbs, their problems and their plans. This
    interface adds the {{!repr}representation} of plans, which serves the
    optimizer and the engine. *)

type t

val of_table : Table.t -> t
val of_source : Source.t -> t
val schema : t -> Schema.t
val select : Expr.row Expr.out list -> t -> t
val derive : Expr.row Expr.out list -> t -> t
val filter : (bool, Expr.row) Expr.t -> t -> t
val sort : Order.t list -> t -> t
val slice : offset:int -> length:int -> t -> t
val aggregate : by:string list -> Expr.agg Expr.out list -> t -> t

val join :
  ?kind:Join.kind ->
  ?each_left:Join.count ->
  ?each_right:Join.count ->
  on:Join.cond ->
  t ->
  t ->
  t

val append : t -> t -> t
val unnest : string list -> t -> t
val equal : t -> t -> bool
val pp : Format.formatter -> t -> unit

(** {1:repr Representation}

    A query is its last step and the schema of its rows. Each step holds its
    arguments bound to its inputs' schemas ({!Expr.bind_out},
    {!Expr.bind_predicate}, {!Join.check}), so a step that the optimizer or the
    engine builds is a step like any other. Steps are not shared unless a plan
    shares them: the optimizer makes equal subplans one value, and the engine
    runs a step that a plan reaches several times once. {!equal} compares
    sources physically, with equal requests. *)

(** The type for steps. *)
type node =
  | Of_table of Table.t  (** The rows of a table. *)
  | Of_source of {
      source : Source.t;
      columns : string list;
          (** The columns read, distinct, in the source's schema order. *)
      filters : (bool, Expr.row) Expr.t list;
          (** The conjuncts handed to the source, bound to its schema, in plan
              order: those it answered [Exact], which no step applies again, and
              those it answered [Inexact], which a filter above applies again.
              Each is a {!Source.Pred.t}, as [Talon_next.Query.optimize]
              translates it. *)
      limit : int option;
          (** [Some n] when no more than the source's first [n] rows are read;
              only when every conjunct on the source is [Exact]. *)
    }
      (** The rows of a source, read with a request. {!of_source} reads every
          column, with no conjunct and no limit. *)
  | Select of { outputs : (string * Expr.packed) list; input : t }
  | Derive of { outputs : (string * Expr.packed) list; input : t }
  | Filter of { predicate : (bool, Expr.row) Expr.t; input : t }
  | Sort of { keys : Order.t list; input : t }
  | Slice of { offset : int; length : int; input : t }
  | Aggregate of {
      by : string list;
      outputs : (string * Expr.packed) list;
      input : t;
    }
  | Join of {
      kind : Join.kind;
      each_left : Join.count;
      each_right : Join.count;
      on : Join.cond;
      left : t;
      right : t;
    }
  | Append of { input : t; rest : t }
  | Unnest of { columns : string list; input : t }

val node : t -> node
(** [node q] is [q]'s last step. *)

val inputs : t -> t list
(** [inputs q] is the inputs of [q]'s last step, in order: none for a table or a
    source, [[left; right]] for a join, [[input; rest]] for an append, and
    [[input]] for the others. *)

val map_inputs : (t -> t) -> node -> node
(** [map_inputs f n] is [n] with [f] applied to each of its inputs. *)

val make : node -> t
(** [make n] is the query whose last step is [n], with the schema that [n]'s
    verb resolves from [n]'s arguments and its inputs' schemas: for a join,
    {!Join.columns}. It checks nothing. The caller keeps the invariants of the
    verbs: [n]'s outputs and predicate are bound to its input's schema, or to
    one with more columns of the same types, its outputs have column typings and
    distinct names, its join condition has passed {!Join.check}, its keys name
    columns of its input, and an [Of_source] names columns of its source and
    holds conjuncts bound to its source's schema. The optimizer and the engine
    build steps with it. *)

val same : t -> t -> bool
(** [same q0 q1] is {!equal} with tables compared physically, so it reads no
    data. The optimizer makes such plans one value. *)
