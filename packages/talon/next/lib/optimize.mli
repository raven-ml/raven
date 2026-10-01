(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Plan optimization.

    This module is internal: [Talon_next] exports {!query} as [Query.optimize].
*)

val query : Query.t -> Query.t
(** [query q] is [Talon_next.Query.optimize q], whose documentation is the
    contract. Running a query optimizes it with [query] first. *)

val pred : (bool, 's) Expr.t -> Source.Pred.t option
(** [pred c] is the source predicate that the bound conjunct [c] is, as
    [Talon_next.Query.optimize] translates it, or [None]. The run builds a
    source's request with it. *)
