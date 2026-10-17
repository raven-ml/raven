(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The 22 decision-support queries in talon, with their validation
    parameters, over the snappy Parquet files [decision_data.py] writes,
    scanned inside the timed region. The queries are derived from the TPC-H
    specification; results from them are not comparable to published TPC-H
    results.

    Talon does not reorder joins, so each query states its joins in a committed
    order: the order of [decision.py]'s Polars queries. Correlated subqueries
    become aggregates over a partition ({!Talon.Expr.over}) or aggregates joined
    back; [exists] and [not exists] become semi and anti joins; [like] patterns
    become {!Talon.Expr.Str} patterns. *)

val workload : data:string -> string -> Workload.t
(** [workload ~data size] is the workload [decision/size] over the data under
    [data], for [size] [sf0.1], [sf1] or [sf10].

    Raises [Invalid_argument] on another size. *)
