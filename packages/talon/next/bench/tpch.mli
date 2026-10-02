(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The 22 TPC-H queries in talon, with the specification's validation
    parameters, over the snappy Parquet files [tpch_data.py] writes, scanned
    inside the timed region.

    Talon does not reorder joins, so each query states its joins in a committed
    order: the order of [tpch.py]'s Polars queries. Correlated subqueries become
    aggregates over a partition ({!Talon_next.Expr.over}) or aggregates joined
    back; [exists] and [not exists] become semi and anti joins; [like] patterns
    become {!Talon_next.Expr.Str} patterns. *)

val workload : data:string -> string -> Workload.t
(** [workload ~data size] is the workload [tpch/size] over the data under
    [data], for [size] [sf0.1], [sf1] or [sf10].

    Raises [Invalid_argument] on another size. *)
