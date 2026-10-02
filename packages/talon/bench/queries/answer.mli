(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Answers: their canonical form, their committed files, and how two compare.

    This is [answer.py]'s contract, so that talon's answers check against the
    files DuckDB's answers were committed as. An answer is committed thinned:
    the rows at positions 0, s, 2s, … of its canonical form, where the stride s
    keeps at most {!limit} rows. [answers/index.csv] records each answer's full
    row count and stride. *)

val limit : int
(** [limit] is [200], the most rows a committed answer keeps. *)

val canonical : ordered:bool -> Talon.t -> Talon.t
(** [canonical ~ordered t] is [t] with its integer columns as [int64], its float
    columns as [float64] and its categorical columns as [string]. Unless
    [ordered], its rows are sorted by every column in turn, ascending, nulls
    last. *)

val stride : int -> int
(** [stride rows] is the stride that thins [rows] rows to at most {!limit}. *)

val thin : int -> Talon.t -> Talon.t
(** [thin s t] is the rows of [t] at positions 0, [s], 2[s], …. *)

val compare : Talon.t -> Talon.t -> string list
(** [compare expected got] is the differences between two canonical answers,
    empty when they agree: their schemas, else their row counts, else one line
    per column whose values differ, with the number of rows that differ and the
    first of them. Floats agree within a relative [1e-9] or an absolute [1e-12],
    nulls with nulls and NaNs with NaNs; every other value agrees by key
    identity ({!Talon.Type.compare_value}). *)

val check : expected:Talon.t -> rows:int -> stride:int -> Talon.t -> string list
(** [check ~expected ~rows ~stride got] is the differences between the canonical
    answer [got] and the committed answer [expected] of [rows] rows thinned by
    [stride]: [got]'s row count, else {!compare} of [expected] and [got]
    thinned. *)

(** {1:files Files} *)

val path : string -> string -> string
(** [path root id] is the file of the answer to the question [id] under [root]:
    [root/groupby-1e6/q01.csv] for [groupby/1e6/q01]. *)

val read_index : string -> (string * (int * int)) list
(** [read_index root] is each committed answer's id with its full row count and
    stride, from [root/index.csv], in the file's order.

    Raises [Failure] if the index cannot be read. *)

val read : Talon.Schema.t -> string -> Talon.t
(** [read schema file] is the CSV answer [file], its columns read as [schema]'s.

    Raises [Failure] if [file] cannot be read as [schema]. *)

val write : string -> Talon.t -> unit
(** [write file t] writes [t] to [file] as CSV, with a header, as {!read} reads
    it back.

    Raises [Failure] if [file] cannot be written. *)
