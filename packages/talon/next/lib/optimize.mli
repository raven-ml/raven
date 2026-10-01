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
