(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Queries: descriptions of tables to compute.

    [Talon_next.Query] documents the verbs, their problems and their plans. *)

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
