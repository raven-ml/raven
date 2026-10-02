(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Workloads: tables, and the questions asked of them.

    This is [workload.py]'s contract on talon's side. A table either loads
    before timing (H2O's CSV files, read into memory with a declared schema) or
    is scanned inside the timed region (TPC-H's Parquet files). *)

(** The type for tables. *)
type table =
  | Csv of string * (string * Talon.Type.any) list
      (** [Csv (file, columns)] is the CSV [file], read as [columns] and loaded
          before timing. *)
  | Parquet of string
      (** [Parquet file] is the Parquet [file], scanned inside the timed region.
      *)

type question = {
  name : string;  (** The question's name, as [q01]. *)
  query : (string -> Talon.Query.t) -> Talon.Query.t;
      (** [query tables] is the question, over the workload's tables by name. *)
}
(** The type for questions. *)

type t = {
  id : string;  (** The case id prefix, as [groupby/1e7] or [tpch/sf1]. *)
  tables : (string * table) list;  (** The tables, by name. *)
  questions : question list;  (** The questions, in order. *)
  ordered : bool;
      (** Whether answers come in an order the question fixes. Unordered answers
          are compared after sorting their rows. *)
}
(** The type for workloads. *)

val missing : t -> string list
(** [missing w] is the files of [w]'s tables that do not exist. *)

val load : t -> string -> Talon.Query.t
(** [load w] reads [w]'s CSV tables into memory and maps its Parquet files, and
    is the query of each table by name.

    Raises [Failure] if a file is missing or does not read, and the returned
    function raises [Invalid_argument] on a name that is not a table of [w]. *)
