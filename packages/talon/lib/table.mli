(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Tables.

    [Talon] documents tables. This interface adds the functions the run builds
    and reads batches with. A batch is a table of one batch: the run's streams
    carry tables, and {!Talon.batches} and {!Talon.of_batches} split and join
    them. *)

type t

val v : ?rows:int -> (string * Column.t) list -> t
val of_batches : t list -> t
val batches : t -> t list
val schema : t -> Schema.t
val rows : t -> int
val column : t -> string -> Column.t
val take : Nx.int64_t -> t -> t
val to_tensor : ('a, 'b) Nx.dtype -> string list -> t -> ('a, 'b) Nx.t
val equal : t -> t -> bool

(** {1:batches Batches} *)

val batch : Schema.t -> rows:int -> Column.t array -> t
(** [batch s ~rows cs] is the table of one batch of [rows] rows with the columns
    [cs], of the columns [s], in order. It checks nothing: each column has
    [rows] rows and its column's type. A batch keeps its row count, so a batch
    without columns has rows. *)

val columns : t -> Column.t array
(** [columns b] is the columns of the one-batch table [b], in schema order.

    Raises [Invalid_argument] if [b] has more than one batch. *)

val concat : t -> t
(** [concat t] is [t]'s rows as one batch: [t] itself when it is one batch, and
    otherwise each column's parts one after the other ({!Column.concat}). A
    table without rows gives one batch without rows. *)

val canonical : t -> t
(** [canonical t] is the one-batch table [t] with canonical columns
    ({!Column.canonical}): [t] itself when its columns are, one copy of each
    other column otherwise. [run]'s result and [column]'s column are made so,
    which keeps their layouts independent of batching. *)
