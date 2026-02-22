(*---------------------------------------------------------------------------
  Copyright (c) 2026 The Raven authors. All rights reserved.
  SPDX-License-Identifier: ISC
  ---------------------------------------------------------------------------*)

(** Notebook documents.

    A document is an ordered sequence of {!Cell.t} values. Operations maintain
    cell ordering and identity. *)

(** {1:documents Documents} *)

type t
(** The type for notebook documents. *)

val empty : unit -> t
(** [empty ()] is a document with no cells. *)

val of_cells : ?metadata:(string * string) list -> Cell.t list -> t
(** [of_cells ?metadata cs] is a document containing [cs] in order with the
    given [metadata] (defaults to [[]]). *)

(** {1:accessors Accessors} *)

val cells : t -> Cell.t list
(** [cells d] is the ordered list of cells in [d]. *)

val length : t -> int
(** [length d] is the number of cells in [d]. *)

val metadata : t -> (string * string) list
(** [metadata d] is the document-level metadata of [d]. *)

val set_metadata : (string * string) list -> t -> t
(** [set_metadata m d] is [d] with metadata replaced by [m]. *)

val nth : int -> t -> Cell.t option
(** [nth i d] is the [i]th cell (zero-indexed), or [None]. *)

val find : Cell.id -> t -> Cell.t option
(** [find id d] is the cell with identifier [id] in [d], or [None]. *)

val find_index : Cell.id -> t -> int option
(** [find_index id d] is the zero-based index of cell [id] in [d]. *)

(** {1:modifications Modifications} *)

val insert : pos:int -> Cell.t -> t -> t
(** [insert ~pos c d] inserts [c] at position [pos]. Cells at [pos] and beyond
    shift right. [pos] is clamped to [[0, length d]]. *)

val remove : Cell.id -> t -> t
(** [remove id d] removes the cell with identifier [id] from [d]. Returns [d]
    unchanged if [id] is not found. *)

val replace : Cell.id -> Cell.t -> t -> t
(** [replace id c d] replaces the cell identified by [id] with [c]. Returns [d]
    unchanged if [id] is not found. *)

val move : Cell.id -> pos:int -> t -> t
(** [move id ~pos d] moves the cell [id] to position [pos]. *)

val update : Cell.id -> (Cell.t -> Cell.t) -> t -> t
(** [update id f d] applies [f] to the cell identified by [id]. *)

val clear_all_outputs : t -> t
(** [clear_all_outputs d] clears outputs from all code cells. *)
