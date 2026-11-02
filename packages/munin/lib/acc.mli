(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Statistics as samples arrive ({!Metric.section-stats}).

    A value is immutable, so a history's statistics stay as they were when a
    later fold adds samples. The writer and every reader add a key's samples in
    log order, so they compute the same bits. *)

type t

val empty : t
val add : t -> Metric.sample -> t

val stats : t -> Metric.stats option
(** [stats a] is [None] when [a] holds no sample. *)
