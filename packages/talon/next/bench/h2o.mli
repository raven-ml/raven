(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** The H2O db-benchmark questions in talon: ten group-by questions and five
    joins, over the CSV files [h2o_data.py] writes, loaded with [h2o.py]'s
    declared schema: identifiers as [string], other integers as [int32] and
    measures as [float64]. *)

val groupby : data:string -> string -> Workload.t
(** [groupby ~data size] is the group-by workload [groupby/size] over the data
    under [data], for [size] [1e6], [1e7] or [1e8].

    Raises [Invalid_argument] on another size. *)

val join : data:string -> string -> Workload.t
(** [join ~data size] is the join workload [join/size], like {!groupby}. *)
