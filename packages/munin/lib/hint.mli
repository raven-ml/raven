(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** [stats.json], an ended run's statistics hint
    ({!Store_format.section-stats}). *)

type key = {
  kind : [ `Metric | `Reading ];
  goal : Metric.goal option;
  stats : Metric.stats;
}

val encode : length:int -> (string * key) list -> string

val decode : string -> (int * (string * key) list) option
(** [decode s] is the log length and keys of the hint [s], if it parses, is of
    this {!Store_format.version} and follows these {!Store_format.rules}. *)

val equal : (string * key) list -> (string * key) list -> bool
(** [equal a b] is [true] iff [a] and [b] hold the same keys and values, floats
    compared bit for bit, since they read back exactly. *)
